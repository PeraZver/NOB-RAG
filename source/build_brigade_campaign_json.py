from __future__ import annotations

import argparse
import json
import sys
import time
from json import JSONDecodeError
from pathlib import Path

from campaign_utils import (
    batch_filename,
    batch_raw_response_filename,
    build_final_campaign_document,
    chunk_records_into_batches,
    extract_json_object,
    format_batch_for_prompt,
    load_reference_template,
    merge_event_records,
)
from rag_env import load_local_env, resolve_provider_model
from rag_providers import generate_with_provider
from rag_utils import infer_book_title, load_metadata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract brigade campaign events from chunked book text in resumable batches."
    )
    parser.add_argument(
        "mode",
        choices=["extract", "consolidate", "run"],
        help="extract batches, consolidate existing batch outputs, or do both",
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Directory containing chunks.jsonl and metadata.json.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="LLM provider used for extraction batches.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional provider model override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks to send to the model in one extraction batch.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="First batch number to process.",
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=None,
        help="Optional last batch number to process.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate extraction batch files. Defaults to <book_dir>/campaign_work.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final JSON output path. Defaults to <book_dir>/brigade_campaign.json.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional reference JSON template. Defaults to <book_dir>/11th_dalmatian_claude-opus-4-6.json if present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run extraction for batches that already have saved output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="How many repair attempts to make if the provider returns malformed JSON.",
    )
    return parser.parse_args()


def resolve_template_path(book_dir: Path, explicit_template: Path | None) -> Path | None:
    if explicit_template is not None:
        return explicit_template.resolve()
    default_template = book_dir / "11th_dalmatian_claude-opus-4-6.json"
    return default_template if default_template.exists() else None


def build_extraction_prompts(book_title: str, brigade_name: str, batch_text: str) -> tuple[str, str]:
    system_prompt = (
        "You extract structured campaign events from a wartime brigade monograph. "
        "Return only valid JSON. "
        "Focus on this brigade's formation, movements, assaults, evacuations, liberations, "
        "defensive combats, and other brigade actions. "
        "If an event is a whole-division operation and no specific brigades are named, you may include it "
        "as participation by the brigade when the source clearly treats it as a division-wide action. "
        "Exclude pure orders, planning directives, commendations, honorary titles, meetings, and unrelated units. "
        "Use the best date the source supports. Use ISO dates YYYY-MM-DD when possible. "
        "Find approximate GPS coordinates for the named place or central coordinates for a larger area. "
        "Be conservative and grounded in the text."
    )
    user_prompt = (
        f"Book title: {book_title}\n"
        f"Target brigade: {brigade_name}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "movements": [\n'
        "    {\n"
        '      "date": "YYYY-MM-DD",\n'
        '      "place": "string",\n'
        '      "coordinates": {"lat": 0.0, "lng": 0.0},\n'
        '      "operation": "string",\n'
        '      "division": "string",\n'
        '      "notes": "string",\n'
        '      "source_chunk_ids": [1],\n'
        '      "source_pages": [12, 13]\n'
        "    }\n"
        "  ],\n"
        '  "notes": ["short batch-level observations or exclusions"]\n'
        "}\n\n"
        "Important rules:\n"
        "1. Only include events tied to the target brigade or clear whole-division actions that include it.\n"
        "2. Keep one record per dated action or movement.\n"
        "3. If the text gives a date range, pick the best single date and explain briefly in notes.\n"
        "4. If the place is a route or larger area, use the best central approximate coordinates.\n"
        "5. Always include source_chunk_ids and source_pages from the evidence.\n\n"
        "Chunks:\n"
        f"{batch_text}"
    )
    return system_prompt, user_prompt


def build_json_repair_prompts(response_text: str) -> tuple[str, str]:
    system_prompt = (
        "You repair malformed JSON. "
        "Return only valid JSON and preserve the original structure and content as closely as possible. "
        "Do not add commentary, markdown fences, or new events."
    )
    user_prompt = (
        "The following text was supposed to be JSON but failed to parse. "
        "Repair it into valid JSON.\n\n"
        f"{response_text}"
    )
    return system_prompt, user_prompt


def parse_with_repair(
    provider: str,
    model: str,
    response_text: str,
    max_retries: int,
) -> tuple[dict, str, int]:
    last_error: Exception | None = None
    current_text = response_text

    for attempt in range(max_retries + 1):
        try:
            parsed = extract_json_object(current_text)
            return parsed, current_text, attempt
        except (JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt >= max_retries:
                break
            repair_system_prompt, repair_user_prompt = build_json_repair_prompts(current_text)
            current_text = generate_with_provider(
                provider=provider,
                model=model,
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
                max_output_tokens=3600,
            )

    assert last_error is not None
    raise last_error


def run_extraction(args: argparse.Namespace, book_dir: Path, work_dir: Path, template: dict) -> None:
    chunks_path = book_dir / "chunks.jsonl"
    batches = chunk_records_into_batches(chunks_path=chunks_path, batch_size=args.batch_size)
    book_title = infer_book_title(book_dir)
    brigade_name = template.get("brigade_name", "11th dalmatian brigade")
    model = resolve_provider_model(args.provider, args.model)
    total_batches = len(batches)
    overall_start = time.perf_counter()

    work_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting extraction for {book_title} with {total_batches} batches "
        f"(batch_size={args.batch_size}, provider={args.provider}, model={model})"
    )

    for batch in batches:
        if batch.batch_id < args.start_batch:
            continue
        if args.end_batch is not None and batch.batch_id > args.end_batch:
            continue

        batch_path = work_dir / batch_filename(batch.batch_id)
        raw_response_path = work_dir / batch_raw_response_filename(batch.batch_id)
        if batch_path.exists() and not args.overwrite:
            elapsed = time.perf_counter() - overall_start
            print(
                f"[{batch.batch_id}/{total_batches}] Skipping {batch_path.name} "
                f"(already exists, elapsed {format_seconds(elapsed)})"
            )
            continue

        batch_start = time.perf_counter()
        print(
            f"[{batch.batch_id}/{total_batches}] Extracting "
            f"{len(batch.records)} chunks spanning pages {batch.source_pages[:1] or ['?']}"
            f"{'...' if len(batch.source_pages) > 1 else ''}"
        )
        system_prompt, user_prompt = build_extraction_prompts(
            book_title=book_title,
            brigade_name=brigade_name,
            batch_text=format_batch_for_prompt(batch),
        )
        response_text = generate_with_provider(
            provider=args.provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3200,
        )
        raw_response_path.write_text(response_text, encoding="utf-8")
        try:
            parsed, final_response_text, repair_attempts = parse_with_repair(
                provider=args.provider,
                model=model,
                response_text=response_text,
                max_retries=args.max_retries,
            )
        except (JSONDecodeError, ValueError) as error:
            batch_elapsed = time.perf_counter() - batch_start
            print(
                f"[{batch.batch_id}/{total_batches}] Failed to parse provider JSON after "
                f"{args.max_retries} repair attempts in {format_seconds(batch_elapsed)}"
            )
            print(f"  Raw response saved to {raw_response_path.name}")
            print("  Tip: retry this batch with a smaller --batch-size, such as 4 or even 2.")
            raise error

        payload = {
            "batch_id": batch.batch_id,
            "chunk_ids": batch.chunk_ids,
            "source_pages": batch.source_pages,
            "provider": args.provider,
            "model": model,
            "movements": parsed.get("movements", []),
            "notes": parsed.get("notes", []),
            "raw_response": final_response_text,
            "repair_attempts": repair_attempts,
        }
        batch_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        batch_elapsed = time.perf_counter() - batch_start
        overall_elapsed = time.perf_counter() - overall_start
        print(
            f"[{batch.batch_id}/{total_batches}] Saved {batch_path.name} "
            f"in {format_seconds(batch_elapsed)} "
            f"(total {format_seconds(overall_elapsed)}, repair_attempts={repair_attempts})"
        )

    total_elapsed = time.perf_counter() - overall_start
    print(f"Extraction pass finished in {format_seconds(total_elapsed)}")


def run_consolidation(book_dir: Path, work_dir: Path, output_path: Path, template: dict) -> None:
    start_time = time.perf_counter()
    batch_files = sorted(work_dir.glob("campaign_batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {work_dir}")
    print(f"Consolidating {len(batch_files)} batch files from {work_dir}")

    all_events: list[dict] = []
    all_notes: list[str] = []
    for batch_file in batch_files:
        payload = json.loads(batch_file.read_text(encoding="utf-8"))
        all_events.extend(payload.get("movements", []))
        all_notes.extend(str(note) for note in payload.get("notes", []))

    merged_events = merge_event_records(all_events)
    metadata = load_metadata(book_dir)
    source_label = metadata.get("source_pdf", str(book_dir / "chunks.jsonl"))
    final_document = build_final_campaign_document(
        template=template,
        merged_events=merged_events,
        top_notes=all_notes,
        source_label=source_label,
    )
    output_path.write_text(
        json.dumps(final_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start_time
    print(f"Wrote final campaign JSON -> {output_path}")
    print(f"Movements: {len(final_document['movements'])}")
    print(f"Consolidation finished in {format_seconds(elapsed)}")


def main() -> None:
    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")

    load_local_env(book_dir)
    work_dir = (args.work_dir or (book_dir / "campaign_work")).resolve()
    output_path = (args.output or (book_dir / "brigade_campaign.json")).resolve()
    template_path = resolve_template_path(book_dir=book_dir, explicit_template=args.template)
    template = load_reference_template(template_path)

    if args.mode in {"extract", "run"}:
        run_extraction(args=args, book_dir=book_dir, work_dir=work_dir, template=template)

    if args.mode in {"consolidate", "run"}:
        run_consolidation(
            book_dir=book_dir,
            work_dir=work_dir,
            output_path=output_path,
            template=template,
        )


if __name__ == "__main__":
    main()
