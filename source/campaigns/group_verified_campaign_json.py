from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

from campaigns.campaign_utils import extract_json_object
from rag.rag_env import load_local_env, resolve_provider_model
from rag.rag_providers import generate_with_provider

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group verified brigade campaign events into broader map-display operations."
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Directory containing brigade_campaign_verified.json.",
    )
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
        help="Optional output JSON path. Defaults to <book_dir>/brigade_campaign_grouped.json.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="LLM provider used for grouping.",
    )
    parser.add_argument("--model", default=None, help="Optional provider model override.")
    parser.add_argument(
        "--major-ops-file",
        type=Path,
        default=None,
        help=(
            "Optional major operations JSON file. Defaults to "
            "source/campaigns/operations.json."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=12000,
        help="Maximum output tokens for the grouping provider call.",
    )
    return parser.parse_args()


def _event_digest(index: int, event: dict) -> dict:
    return {
        "index": index,
        "date": str(event.get("date", "")).strip(),
        "place": str(event.get("place", "")).strip(),
        "operation": str(event.get("operation", "")).strip(),
        "division": str(event.get("division", "")).strip(),
        "notes": str(event.get("notes", "")).strip(),
    }


def _normalize_text(value: str) -> str:
    lowered = str(value or "").lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _tokenize_region(value: str) -> set[str]:
    stopwords = {
        "area",
        "sector",
        "valley",
        "line",
        "coast",
        "island",
        "islands",
        "region",
        "central",
        "western",
        "eastern",
        "northern",
        "southern",
        "approaches",
        "road",
        "route",
        "mountain",
        "river",
        "canyon",
        "polje",
        "west",
        "east",
        "north",
        "south",
    }
    tokens = {token for token in _normalize_text(value).split() if len(token) >= 4}
    return {token for token in tokens if token not in stopwords}


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def _resolve_major_ops_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()
    campaigns_dir = Path(__file__).resolve().parent
    preferred = campaigns_dir / "operations.json"
    if preferred.exists():
        return preferred
    return campaigns_dir / "yugoslavia_major_operations_initial.json"


def _load_major_operations(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"major operations file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Major operations file must be a JSON array.")

    operations: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        operation_name = str(item.get("operation_name", "")).strip()
        start_date = str(item.get("start_date", "")).strip()
        end_date = str(item.get("end_date", "")).strip()
        regions_affected = item.get("regions_affected", [])

        if not operation_name or not start_date or not end_date:
            continue
        if not isinstance(regions_affected, list):
            regions_affected = []

        operations.append(
            {
                "operation_name": operation_name,
                "start_date": start_date,
                "end_date": end_date,
                "regions_affected": [str(region).strip() for region in regions_affected if str(region).strip()],
            }
        )
    if not operations:
        raise ValueError(f"No valid operations found in {path}")
    return operations


def _event_best_major_operation(event: dict, operations: list[dict]) -> str | None:
    raw_date = str(event.get("date", "")).strip()
    if not raw_date:
        return None

    event_date = _parse_iso_date(raw_date)
    event_place = str(event.get("place", "")).strip()
    event_place_norm = _normalize_text(event_place)
    event_place_tokens = _tokenize_region(event_place)

    best_name: str | None = None
    best_score = 0
    best_span = 10**9

    for operation in operations:
        op_start = _parse_iso_date(operation["start_date"])
        op_end = _parse_iso_date(operation["end_date"])
        if event_date < op_start or event_date > op_end:
            continue

        score = 0
        for region in operation.get("regions_affected", []):
            region_norm = _normalize_text(region)
            if not region_norm:
                continue

            if region_norm in event_place_norm:
                score = max(score, 6 + len(region_norm.split()))
                continue

            overlap = len(event_place_tokens & _tokenize_region(region))
            if overlap > 0:
                score = max(score, overlap)

        if score <= 0:
            continue

        span_days = (op_end - op_start).days
        if score > best_score or (score == best_score and span_days < best_span):
            best_name = operation["operation_name"]
            best_score = score
            best_span = span_days

    return best_name


def _split_group_by_major_operation(
    event_indexes: list[int],
    event_major_ops: list[str | None],
) -> list[tuple[list[int], str | None]]:
    if not event_indexes:
        return []

    runs: list[tuple[list[int], str | None]] = []
    current_indexes = [event_indexes[0]]
    current_operation = event_major_ops[event_indexes[0] - 1]
    previous_index = event_indexes[0]

    for index in event_indexes[1:]:
        operation_name = event_major_ops[index - 1]
        if operation_name != current_operation or index != previous_index + 1:
            runs.append((current_indexes, current_operation))
            current_indexes = [index]
            current_operation = operation_name
        else:
            current_indexes.append(index)
        previous_index = index

    runs.append((current_indexes, current_operation))
    return runs


def _filter_operations_for_events(events: list[dict], operations: list[dict]) -> list[dict]:
    event_dates: list[date] = []
    for event in events:
        raw_date = str(event.get("date", "")).strip()
        if raw_date:
            event_dates.append(_parse_iso_date(raw_date))

    if not event_dates:
        return operations

    min_event_date = min(event_dates)
    max_event_date = max(event_dates)
    filtered: list[dict] = []
    for operation in operations:
        op_start = _parse_iso_date(operation["start_date"])
        op_end = _parse_iso_date(operation["end_date"])
        if op_end < min_event_date or op_start > max_event_date:
            continue
        filtered.append(operation)

    return filtered or operations


def build_grouping_prompts(
    brigade_name: str,
    events: list[dict],
    major_operations: list[dict],
) -> tuple[str, str]:
    system_prompt = (
        "You group already-verified WWII brigade campaign events into map-display groups. "
        "Return only valid JSON. "
        "Do not change the underlying facts of the input events. "
        "Your task is to decide which nearby events belong to one short, coherent combat sequence in the same place and time. "
        "Favor smaller, local, tactically coherent groups over broad campaign summaries."
    )

    rendered_events = json.dumps(
        [_event_digest(index=index, event=event) for index, event in enumerate(events, start=1)],
        ensure_ascii=False,
        indent=2,
    )
    rendered_operations = json.dumps(major_operations, ensure_ascii=False, indent=2)

    user_prompt = (
        f"Target brigade: {brigade_name}\n\n"
        "You are given a chronological list of verified events. Group them into map-display entries.\n\n"
        "You are also given a catalog of major Yugoslav WWII operations with operation_name, start_date, end_date, and regions_affected.\n\n"
        "Grouping rules:\n"
        "1. For each event, first test major-operation matching: an event matches a major operation when its date is within that operation's start_date/end_date and its place/region context overlaps one or more of regions_affected.\n"
        "2. If events match the same major operation and are reasonably close in sequence, group them together and set operation exactly to that major operation's operation_name.\n"
        "3. If an event does not match any major operation by both date and region, use a local tactical group with a short operation title.\n"
        "4. Prefer small local groups. Group neighboring events only when they are close in date and happen in the same immediate area or on the same objective.\n"
        "5. Same broader campaign is not enough. Do not merge events just because they belong to the same offensive, month, or front.\n"
        "6. Split groups whenever the brigade shifts to a distinct town, island, ridge, road axis, landing site, fortress, or urban battlefield.\n"
        "7. Do not create overlapping groups. Every input index must appear exactly once.\n"
        "8. Keep groups chronologically ordered.\n"
        "9. notes must be a short factual summary in English, built from grouped event notes, including involved units, casualties, and key achievements when those details are present in the source notes.\n"
        "10. If a group contains just one input event, return it as a single-item group.\n"
        "11. Be strict in late-war phases: do not merge Istria actions with Trieste actions unless both date and region clearly match the same catalog operation.\n\n"
        "Example of the intended behavior:\n"
        "- Multiple events from 1943-09-13 through 1943-09-26 in the Split, Klis, Solin, Mosor area can be grouped as 'Battle for Split 1943' if they describe the same operational episode.\n\n"
        "- But late-April Istria landings, fighting at Paz, and urban fighting in Trieste should be separate groups if the brigade has clearly shifted objective and place.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "groups": [\n'
        "    {\n"
        '      "event_indexes": [1, 2],\n'
        '      "operation": "string",\n'
        '      "notes": "string"\n'
        "    }\n"
        "  ],\n"
        '  "notes": ["brief grouping notes"]\n'
        "}\n\n"
        "Major operations catalog:\n"
        f"{rendered_operations}\n\n"
        "Input events:\n"
        f"{rendered_events}"
    )
    return system_prompt, user_prompt


def _pick_group_coordinates(events: list[dict]) -> dict:
    lat_values: list[float] = []
    lng_values: list[float] = []
    for event in events:
        coordinates = event.get("coordinates") or {}
        lat = coordinates.get("lat")
        lng = coordinates.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            lat_values.append(float(lat))
            lng_values.append(float(lng))

    if not lat_values or not lng_values:
        return {"lat": None, "lng": None}

    return {
        "lat": round(sum(lat_values) / len(lat_values), 6),
        "lng": round(sum(lng_values) / len(lng_values), 6),
    }


def _pick_division(events: list[dict]) -> str:
    for event in events:
        division = str(event.get("division", "")).strip()
        if division:
            return division
    return ""


def _merge_places(events: list[dict]) -> str:
    places: list[str] = []
    for event in events:
        place = " ".join(str(event.get("place", "")).split()).strip()
        if place and place not in places:
            places.append(place)
    return " | ".join(places)


def _clean_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip(" ,;:-") + "..."


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if sentence.strip()
    ]


def _pick_unique(snippets: list[str], limit: int = 1) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        normalized = _normalize_text(snippet)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        picked.append(" ".join(snippet.split()))
        if len(picked) >= limit:
            break
    return picked


def _build_group_note(member_events: list[dict], fallback_text: str, start_date: date, end_date: date) -> str:
    note_texts = [
        str(event.get("notes", "")).strip()
        for event in member_events
        if str(event.get("notes", "")).strip()
    ]
    sentences: list[str] = []
    for note_text in note_texts:
        sentences.extend(_split_sentences(note_text))

    if not sentences and fallback_text:
        sentences = _split_sentences(fallback_text)

    combined = " ".join(note_texts)
    lead = _clean_text((note_texts[0] if note_texts else (sentences[0] if sentences else "")), max_chars=300)

    unit_matches = re.findall(
        r"\b(?:\d+(?:st|nd|rd|th)?\s+)?(?:[A-Z][\w-]*\s+){0,2}(?:Battalion|Company|Brigade|Regiment|Division|Detachment|Platoon)s?\b",
        combined,
        flags=re.IGNORECASE,
    )
    units = _pick_unique(unit_matches, limit=3)

    casualty_matches = re.findall(
        r"\b\d+\s+(?:killed|wounded|captured|dead|prisoners?)\b",
        combined,
        flags=re.IGNORECASE,
    )
    if re.search(r"\bno losses\b", combined, flags=re.IGNORECASE):
        casualty_matches.append("no losses")
    casualties = _pick_unique(casualty_matches, limit=3)

    outcome_markers = ["liberat", "captur", "seiz", "destroy", "repuls", "withdrew", "retook", "held", "overran"]
    outcome_candidates = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in outcome_markers)
    ]
    outcomes = _pick_unique(outcome_candidates, limit=1)
    lead_lower = lead.lower()
    lead_has_units = any(marker in lead_lower for marker in ["battalion", "company", "brigade", "regiment", "division", "platoon", "detachment"])
    lead_has_casualties = any(marker in lead_lower for marker in ["killed", "wounded", "captured", "dead", "losses", "prisoner", "casualt"])
    lead_has_outcomes = any(marker in lead_lower for marker in ["liberat", "captur", "seiz", "destroy", "repuls", "withdrew", "retook", "held", "overran"])

    parts: list[str] = []
    if lead:
        parts.append(lead)
    if units and not lead_has_units:
        units_text = ", ".join(units)
        if _normalize_text(units_text) not in _normalize_text(lead):
            parts.append(f"Involved units: {units_text}.")
    if casualties and not lead_has_casualties:
        parts.append(f"Casualties: {', '.join(casualties)}.")
    if outcomes and not lead_has_outcomes:
        if _normalize_text(outcomes[0]) not in _normalize_text(lead):
            parts.append(f"Key outcome: {_clean_text(outcomes[0], max_chars=180)}")

    note = " ".join(parts).strip()
    if start_date != end_date:
        note = f"{note} Grouped span: {start_date.isoformat()} to {end_date.isoformat()}.".strip()
    return _clean_text(note, max_chars=650)


def _validate_groups(raw_groups: list[dict], event_count: int) -> list[list[int]]:
    validated: list[list[int]] = []
    seen: set[int] = set()

    for raw_group in raw_groups:
        indexes = raw_group.get("event_indexes", [])
        if not isinstance(indexes, list):
            raise ValueError("Provider returned a group with non-list event_indexes.")

        normalized_indexes: list[int] = []
        for value in indexes:
            index = int(value)
            if index < 1 or index > event_count:
                raise ValueError(f"Provider returned out-of-range event index: {index}")
            if index in seen:
                raise ValueError(f"Provider returned duplicate event index across groups: {index}")
            seen.add(index)
            normalized_indexes.append(index)

        if not normalized_indexes:
            raise ValueError("Provider returned an empty group.")

        ordered_indexes = sorted(normalized_indexes)
        validated.append(ordered_indexes)

    missing = [index for index in range(1, event_count + 1) if index not in seen]
    if missing:
        raise ValueError(f"Provider did not assign every event to a group. Missing indexes: {missing}")

    validated.sort(key=lambda group: group[0])
    return validated


def build_grouped_document(
    document: dict,
    parsed: dict,
    brigade_name: str,
    major_operations: list[dict],
) -> dict:
    events = list(document.get("movements", []))
    raw_groups = parsed.get("groups", [])
    validated_groups = _validate_groups(raw_groups=raw_groups, event_count=len(events))
    event_major_ops = [_event_best_major_operation(event=event, operations=major_operations) for event in events]
    by_first_index = {
        sorted(int(value) for value in group.get("event_indexes", []))[0]: group
        for group in raw_groups
        if group.get("event_indexes")
    }

    grouped_movements: list[dict] = []
    for event_indexes in validated_groups:
        source_group = by_first_index[event_indexes[0]]
        split_runs = _split_group_by_major_operation(event_indexes=event_indexes, event_major_ops=event_major_ops)

        for run_indexes, matched_operation_name in split_runs:
            member_events = [events[index - 1] for index in run_indexes]
            start_date = min(_parse_iso_date(event.get("date", "")) for event in member_events)
            end_date = max(_parse_iso_date(event.get("date", "")) for event in member_events)

            operation_name = _clean_text(source_group.get("operation", ""), max_chars=120)
            if matched_operation_name:
                operation_name = matched_operation_name

            notes = _build_group_note(
                member_events=member_events,
                fallback_text=source_group.get("notes", ""),
                start_date=start_date,
                end_date=end_date,
            )

            grouped_movements.append(
                {
                    "date": start_date.isoformat(),
                    "place": _merge_places(member_events),
                    "coordinates": _pick_group_coordinates(member_events),
                    "operation": operation_name,
                    "division": _pick_division(member_events),
                    "notes": notes,
                }
            )

    return {
        "brigade_name": document.get("brigade_name") or brigade_name,
        "brigade_id": document.get("brigade_id", ""),
        "notes": " | ".join(
            " ".join(str(note).split()).strip()
            for note in parsed.get("notes", [])
            if str(note).strip()
        ),
        "movements": grouped_movements,
    }


def main() -> None:
    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")

    load_local_env(book_dir)
    model = resolve_provider_model(args.provider, args.model)

    input_path = (args.input or (book_dir / "brigade_campaign_verified.json")).resolve()
    output_path = (args.output or (book_dir / "brigade_campaign_grouped.json")).resolve()

    document = json.loads(input_path.read_text(encoding="utf-8"))
    brigade_name = document.get("brigade_name") or book_dir.name
    events = list(document.get("movements", []))
    if not events:
        raise ValueError(f"No movements found in {input_path}")

    major_ops_path = _resolve_major_ops_path(args.major_ops_file)
    major_operations = _filter_operations_for_events(
        events=events,
        operations=_load_major_operations(major_ops_path),
    )

    system_prompt, user_prompt = build_grouping_prompts(
        brigade_name=brigade_name,
        events=events,
        major_operations=major_operations,
    )
    response_text = generate_with_provider(
        provider=args.provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=args.max_output_tokens,
    )
    try:
        parsed = extract_json_object(response_text)
    except Exception:
        raw_output_path = output_path.with_suffix(output_path.suffix + ".raw.txt")
        raw_output_path.write_text(response_text, encoding="utf-8")
        print(f"Saved raw provider response for debugging -> {raw_output_path}")
        raise
    grouped_document = build_grouped_document(
        document=document,
        parsed=parsed,
        brigade_name=brigade_name,
        major_operations=major_operations,
    )
    output_path.write_text(json.dumps(grouped_document, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote grouped campaign JSON -> {output_path}")
    print(f"Major operations considered: {len(major_operations)} from {major_ops_path}")
    print(f"Input movements: {len(events)}")
    print(f"Grouped movements: {len(grouped_document['movements'])}")


if __name__ == "__main__":
    main()