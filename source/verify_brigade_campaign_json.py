from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from campaign_utils import (
    build_final_campaign_document,
    build_verification_groups,
    collect_events_from_payloads,
    extract_json_object,
    load_batch_payloads,
    load_reference_template,
    merge_event_records,
    verification_filename,
)
from rag_env import load_local_env, resolve_provider_model
from rag_providers import generate_with_provider
from rag_utils import load_metadata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and refine extracted brigade campaign events in resumable groups."
    )
    parser.add_argument(
        "mode",
        choices=["verify", "consolidate", "run"],
        help="verify extracted groups, consolidate verified groups, or do both",
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Directory containing campaign_work and metadata.json.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="LLM provider used for verification groups.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional provider model override.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory containing campaign_batch_*.json. Defaults to <book_dir>/campaign_work.",
    )
    parser.add_argument(
        "--verify-dir",
        type=Path,
        default=None,
        help="Directory for verification group outputs. Defaults to <book_dir>/campaign_verify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final verified JSON output path. Defaults to <book_dir>/brigade_campaign_verified.json.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional reference JSON template. Defaults to <book_dir>/11th_dalmatian_claude-opus-4-6.json if present.",
    )
    parser.add_argument(
        "--start-group",
        type=int,
        default=1,
        help="First verification group number to process.",
    )
    parser.add_argument(
        "--end-group",
        type=int,
        default=None,
        help="Optional last verification group number to process.",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=6,
        help="Maximum number of candidate events to send in one verification group.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run verification for groups that already have saved output.",
    )
    return parser.parse_args()


def resolve_template_path(book_dir: Path, explicit_template: Path | None) -> Path | None:
    if explicit_template is not None:
        return explicit_template.resolve()
    source_template = Path(__file__).resolve().parent / "brigade_campaign_template.json"
    if source_template.exists():
        return source_template
    default_template = book_dir / "11th_dalmatian_claude-opus-4-6.json"
    return default_template if default_template.exists() else None


def build_verification_prompts(brigade_name: str, events: list[dict]) -> tuple[str, str]:
    system_prompt = (
        "You are a careful verifier of extracted WWII campaign events. "
        "You receive candidate event records that may overlap, duplicate each other, have fuzzy dates, "
        "or have approximate place coordinates. "
        "Return only valid JSON. "
        "Your job is to merge duplicates intelligently, tighten dates when the evidence supports it, "
        "normalize place naming, and provide the best approximate coordinates you can for each verified event. "
        "Do not invent events beyond what the candidate records support."
    )

    rendered_events = json.dumps(events, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Target brigade: {brigade_name}\n\n"
        "Review these candidate events.\n"
        "Tasks:\n"
        "1. Merge duplicate or near-duplicate events that describe the same brigade action.\n"
        "2. Tighten the date if several candidates clearly point to the same better date.\n"
        "3. Normalize place names and improve approximate coordinates.\n"
        "4. Preserve source_chunk_ids and source_pages by unioning them across merged events.\n"
        "5. Keep only events that still look relevant to the brigade.\n\n"
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
        '  "notes": ["brief verifier notes"]\n'
        "}\n\n"
        "Candidate events:\n"
        f"{rendered_events}"
    )
    return system_prompt, user_prompt


def run_verification(
    args: argparse.Namespace,
    work_dir: Path,
    verify_dir: Path,
    template: dict,
) -> None:
    payloads = load_batch_payloads(work_dir)
    if not payloads:
        raise FileNotFoundError(f"No campaign batch files found in {work_dir}")

    events, _notes = collect_events_from_payloads(payloads)
    groups = build_verification_groups(events, max_group_size=args.max_group_size)
    model = resolve_provider_model(args.provider, args.model)
    brigade_name = template.get("brigade_name") or book_dir.name

    verify_dir.mkdir(parents=True, exist_ok=True)

    for group_id, group in enumerate(groups, start=1):
        if group_id < args.start_group:
            continue
        if args.end_group is not None and group_id > args.end_group:
            continue

        group_path = verify_dir / verification_filename(group_id)
        if group_path.exists() and not args.overwrite:
            print(f"Skipping group {group_id}: {group_path.name} already exists")
            continue

        if len(group) == 1:
            payload = {
                "group_id": group_id,
                "provider": None,
                "model": None,
                "input_events": group,
                "movements": group,
                "notes": ["Skipped provider verification because the group contains a single event."],
                "raw_response": None,
            }
            group_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved verification group {group_id} -> {group_path.name} (single-event passthrough)")
            continue

        system_prompt, user_prompt = build_verification_prompts(
            brigade_name=brigade_name,
            events=group,
        )
        response_text = generate_with_provider(
            provider=args.provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2800,
        )
        parsed = extract_json_object(response_text)
        payload = {
            "group_id": group_id,
            "provider": args.provider,
            "model": model,
            "input_events": group,
            "movements": parsed.get("movements", []),
            "notes": parsed.get("notes", []),
            "raw_response": response_text,
        }
        group_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved verification group {group_id} -> {group_path.name}")


def run_consolidation(
    book_dir: Path,
    verify_dir: Path,
    output_path: Path,
    template: dict,
) -> None:
    payloads = load_batch_payloads(verify_dir, pattern="campaign_verify_*.json")
    if not payloads:
        raise FileNotFoundError(f"No verification files found in {verify_dir}")

    events, notes = collect_events_from_payloads(payloads)
    merged_events = merge_event_records(events)
    metadata = load_metadata(book_dir)
    source_label = metadata.get("source_pdf", str(book_dir / "chunks.jsonl"))
    final_document = build_final_campaign_document(
        template=template,
        merged_events=merged_events,
        top_notes=notes,
        source_label=source_label,
    )
    output_path.write_text(json.dumps(final_document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote verified campaign JSON -> {output_path}")
    print(f"Movements: {len(final_document['movements'])}")


def main() -> None:
    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")

    load_local_env(book_dir)
    work_dir = (args.work_dir or (book_dir / "campaign_work")).resolve()
    verify_dir = (args.verify_dir or (book_dir / "campaign_verify")).resolve()
    output_path = (args.output or (book_dir / "brigade_campaign_verified.json")).resolve()
    template_path = resolve_template_path(book_dir=book_dir, explicit_template=args.template)
    template = load_reference_template(template_path)

    if args.mode in {"verify", "run"}:
        run_verification(args=args, work_dir=work_dir, verify_dir=verify_dir, template=template)

    if args.mode in {"consolidate", "run"}:
        run_consolidation(
            book_dir=book_dir,
            verify_dir=verify_dir,
            output_path=output_path,
            template=template,
        )


if __name__ == "__main__":
    main()
