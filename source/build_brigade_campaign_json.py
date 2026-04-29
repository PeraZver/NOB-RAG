from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from campaign_utils import (
    batch_filename,
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


def run_extraction(args: argparse.Namespace, book_dir: Path, work_dir: Path, template: dict) -> None:
    chunks_path = book_dir / "chunks.jsonl"
    batches = chunk_records_into_batches(chunks_path=chunks_path, batch_size=args.batch_size)
    book_title = infer_book_title(book_dir)
    brigade_name = template.get("brigade_name", "11th dalmatian brigade")
    model = resolve_provider_model(args.provider, args.model)

    work_dir.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        if batch.batch_id < args.start_batch:
            continue
        if args.end_batch is not None and batch.batch_id > args.end_batch:
            continue

        batch_path = work_dir / batch_filename(batch.batch_id)
        if batch_path.exists() and not args.overwrite:
            print(f"Skipping batch {batch.batch_id}: {batch_path.name} already exists")
            continue

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
        parsed = extract_json_object(response_text)
        payload = {
            "batch_id": batch.batch_id,
            "chunk_ids": batch.chunk_ids,
            "source_pages": batch.source_pages,
            "provider": args.provider,
            "model": model,
            "movements": parsed.get("movements", []),
            "notes": parsed.get("notes", []),
            "raw_response": response_text,
        }
        batch_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved batch {batch.batch_id} -> {batch_path.name}")


def run_consolidation(book_dir: Path, work_dir: Path, output_path: Path, template: dict) -> None:
    batch_files = sorted(work_dir.glob("campaign_batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {work_dir}")

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
    print(f"Wrote final campaign JSON -> {output_path}")
    print(f"Movements: {len(final_document['movements'])}")


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
