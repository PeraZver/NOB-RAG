from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from campaigns.campaign_utils import extract_json_object
from rag.rag_env import load_local_env, resolve_provider_model
from rag.rag_providers import generate_with_provider

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polish verified brigade campaign JSON by shortening operation labels and tightening notes."
    )
    parser.add_argument("book_dir", type=Path, help="Directory containing brigade_campaign_verified.json.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional input JSON path. Defaults to <book_dir>/brigade_campaign_verified.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to overwriting <book_dir>/brigade_campaign_verified.json.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="LLM provider used for polishing event wording.",
    )
    parser.add_argument("--model", default=None, help="Optional provider model override.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of events to polish in one provider request.",
    )
    return parser.parse_args()


def _trim_note(text: str, max_chars: int = 500) -> str:
    value = " ".join(str(text).split()).strip()
    if not value:
        return value

    sentence_parts = re.split(r"(?<=[.!?])\s+", value)
    meta_markers = [
        "estimated",
        "estimate",
        "chapter range",
        "source text",
        "context indicates",
        "best date",
        "placed at",
        "placed in",
        "exact date not given",
        "date reflects",
        "date retained",
        "interpreted as",
        "unsupported",
        "candidate",
        "roman numeral date",
    ]
    filtered_parts = []
    for sentence in sentence_parts:
        normalized = sentence.lower()
        if any(marker in normalized for marker in meta_markers):
            continue
        filtered_parts.append(sentence.strip())

    value = " ".join(part for part in filtered_parts if part).strip() or value
    if len(value) <= max_chars:
        return value
    clipped = value[: max_chars - 3].rstrip(" ,;:-")
    return f"{clipped}..."


def _clean_operation(text: str) -> str:
    value = " ".join(str(text).split()).strip().strip("\"'")
    if len(value) <= 100:
        return value
    for separator in [" – ", ": ", "; "]:
        if separator in value:
            candidate = value.split(separator, 1)[0].strip()
            if candidate:
                return candidate
    return value[:100].rstrip(" ,;:-")


def build_polish_prompts(brigade_name: str, events: list[dict]) -> tuple[str, str]:
    system_prompt = (
        "You polish already-verified WWII brigade campaign records. "
        "Return only valid JSON. "
        "Do not add or remove events. "
        "Rewrite only the operation and notes fields for each input item."
    )

    rendered_events = []
    for index, event in enumerate(events, start=1):
        rendered_events.append(
            {
                "index": index,
                "date": event.get("date", ""),
                "place": event.get("place", ""),
                "operation": event.get("operation", ""),
                "division": event.get("division", ""),
                "notes": event.get("notes", ""),
            }
        )

    user_prompt = (
        f"Target brigade: {brigade_name}\n\n"
        "Rewrite each event using these rules:\n"
        "1. Keep the event meaning, date, place, and division intact.\n"
        "2. operation must be short and title-like, usually 2-8 words.\n"
        "3. operation examples: 'Assault on Brač', 'Liberation of Šibenik', 'Mostar Operation'.\n"
        "4. Avoid long sentence-style operation labels.\n"
        "5. notes must be concise and factual, maximum 500 characters.\n"
        "6. Do not mention parsing choices, source ambiguity, estimation method, page references, or reconciliation commentary.\n"
        "7. Keep language neutral and consistent with the existing English JSON style.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "events": [\n'
        "    {\n"
        '      "index": 1,\n'
        '      "operation": "string",\n'
        '      "notes": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Events:\n"
        f"{json.dumps(rendered_events, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def polish_events(
    provider: str,
    model: str,
    brigade_name: str,
    events: list[dict],
    batch_size: int,
) -> list[dict]:
    polished: list[dict] = []

    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        system_prompt, user_prompt = build_polish_prompts(brigade_name=brigade_name, events=batch)
        response_text = generate_with_provider(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3200,
        )
        parsed = extract_json_object(response_text)
        rewritten = parsed.get("events", [])
        by_index = {int(item["index"]): item for item in rewritten if "index" in item}

        for index, event in enumerate(batch, start=1):
            updated = dict(event)
            candidate = by_index.get(index, {})
            updated["operation"] = _clean_operation(candidate.get("operation", event.get("operation", "")))
            updated["notes"] = _trim_note(candidate.get("notes", event.get("notes", "")))
            polished.append(updated)

        print(
            f"Polished events {start + 1}-{start + len(batch)} of {len(events)} "
            f"(provider={provider}, model={model})"
        )

    return polished


def main() -> None:
    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")

    load_local_env(book_dir)
    model = resolve_provider_model(args.provider, args.model)
    input_path = (args.input or (book_dir / "brigade_campaign_verified.json")).resolve()
    output_path = (args.output or input_path).resolve()

    document = json.loads(input_path.read_text(encoding="utf-8"))
    brigade_name = document.get("brigade_name") or book_dir.name
    events = document.get("movements", [])
    document["movements"] = polish_events(
        provider=args.provider,
        model=model,
        brigade_name=brigade_name,
        events=events,
        batch_size=args.batch_size,
    )
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote polished verified campaign JSON -> {output_path}")
    print(f"Movements: {len(document['movements'])}")


if __name__ == "__main__":
    main()
