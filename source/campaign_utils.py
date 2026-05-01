from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rag_utils import iter_jsonl, normalize_search_text


@dataclass
class ChunkBatch:
    batch_id: int
    records: list[dict]

    @property
    def chunk_ids(self) -> list[int]:
        return [int(record["chunk_id"]) for record in self.records]

    @property
    def source_pages(self) -> list[int]:
        pages: list[int] = []
        for record in self.records:
            pages.extend(int(page) for page in record.get("source_pages", []))
        return sorted(set(pages))


def load_reference_template(template_path: Path | None) -> dict:
    if template_path is None or not template_path.exists():
        return {
            "brigade_id": None,
            "brigade_name": "",
            "movements": [],
            "notes": "",
            "source": "",
        }
    return json.loads(template_path.read_text(encoding="utf-8"))


def chunk_records_into_batches(chunks_path: Path, batch_size: int) -> list[ChunkBatch]:
    records = list(iter_jsonl(chunks_path))
    batches: list[ChunkBatch] = []
    for index in range(0, len(records), batch_size):
        batch_id = (index // batch_size) + 1
        batches.append(ChunkBatch(batch_id=batch_id, records=records[index : index + batch_size]))
    return batches


def format_batch_for_prompt(batch: ChunkBatch) -> str:
    lines: list[str] = []
    for record in batch.records:
        pages = ",".join(str(page) for page in record.get("source_pages", []))
        lines.extend(
            [
                f"CHUNK {record['chunk_id']}",
                f"PAGES: {pages}",
                "TEXT:",
                record.get("text", ""),
                "",
            ]
        )
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            return json.loads(candidate)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in provider response.")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return json.loads(candidate)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_event(event: dict) -> dict:
    coordinates = event.get("coordinates") or {}
    source_chunk_ids = sorted({int(item) for item in event.get("source_chunk_ids", [])})
    source_pages = sorted({int(item) for item in event.get("source_pages", [])})
    normalized = {
        "date": str(event.get("date", "")).strip(),
        "place": str(event.get("place", "")).strip(),
        "coordinates": {
            "lat": _coerce_float(coordinates.get("lat")),
            "lng": _coerce_float(coordinates.get("lng")),
        },
        "operation": str(event.get("operation", "")).strip(),
        "division": str(event.get("division", "")).strip(),
        "notes": str(event.get("notes", "")).strip(),
        "source_chunk_ids": source_chunk_ids,
        "source_pages": source_pages,
    }
    return normalized


def event_identity_key(event: dict) -> tuple[str, str, str]:
    return (
        event.get("date", "").strip(),
        normalize_search_text(event.get("place", "")),
        normalize_search_text(event.get("operation", "")),
    )


def _operation_priority(event: dict) -> int:
    operation_text = normalize_search_text(event.get("operation", ""))
    notes_text = normalize_search_text(event.get("notes", ""))
    combined = f"{operation_text} {notes_text}".strip()

    if "formation" in combined or "formiranje" in combined:
        return 0
    if "reorganisation" in combined or "reorganization" in combined or "redesignated" in combined:
        return 1
    return 2


def event_sort_key(event: dict) -> tuple[str, int, str, str]:
    return (
        event.get("date", "").strip(),
        _operation_priority(event),
        event.get("place", "").strip(),
        event.get("operation", "").strip(),
    )


def merge_event_records(events: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}

    for raw_event in events:
        event = normalize_event(raw_event)
        if not event["date"] or not event["place"] or not event["operation"]:
            continue

        key = event_identity_key(event)
        existing = merged.get(key)
        if existing is None:
            merged[key] = event
            continue

        if len(event["notes"]) > len(existing["notes"]):
            existing["notes"] = event["notes"]
        if not existing["division"] and event["division"]:
            existing["division"] = event["division"]
        if existing["coordinates"]["lat"] is None and event["coordinates"]["lat"] is not None:
            existing["coordinates"] = event["coordinates"]
        existing["source_chunk_ids"] = sorted(
            set(existing["source_chunk_ids"]) | set(event["source_chunk_ids"])
        )
        existing["source_pages"] = sorted(
            set(existing["source_pages"]) | set(event["source_pages"])
        )

    return sorted(merged.values(), key=event_sort_key)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_batch_payloads(work_dir: Path, pattern: str = "campaign_batch_*.json") -> list[dict]:
    return [load_json(path) for path in sorted(work_dir.glob(pattern))]


def collect_events_from_payloads(payloads: list[dict]) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    notes: list[str] = []
    for payload in payloads:
        events.extend(payload.get("movements", []))
        notes.extend(str(note) for note in payload.get("notes", []))
    return events, notes


def strip_event_to_template(event: dict) -> dict:
    return {
        "date": event["date"],
        "place": event["place"],
        "coordinates": {
            "lat": event["coordinates"]["lat"],
            "lng": event["coordinates"]["lng"],
        },
        "operation": event["operation"],
        "division": event["division"],
        "notes": event["notes"],
    }


def build_final_campaign_document(
    template: dict,
    merged_events: list[dict],
    top_notes: list[str],
    source_label: str,
) -> dict:
    final_notes = " | ".join(note for note in top_notes if note.strip())
    return {
        "brigade_id": template.get("brigade_id", 11),
        "brigade_name": template.get("brigade_name", "11th dalmatian brigade"),
        "movements": [strip_event_to_template(event) for event in merged_events],
        "notes": final_notes or template.get("notes", ""),
        "source": source_label or template.get("source", ""),
    }


def batch_filename(batch_id: int) -> str:
    return f"campaign_batch_{batch_id:04d}.json"


def batch_raw_response_filename(batch_id: int) -> str:
    return f"campaign_batch_{batch_id:04d}.raw.txt"


def infer_total_batches(chunks_path: Path, batch_size: int) -> int:
    record_count = sum(1 for _ in iter_jsonl(chunks_path))
    return math.ceil(record_count / batch_size)


def verification_filename(group_id: int) -> str:
    return f"campaign_verify_{group_id:04d}.json"


def parse_isoish_date(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def rough_date_distance_days(left: str, right: str) -> int | None:
    left_parts = parse_isoish_date(left)
    right_parts = parse_isoish_date(right)
    if left_parts is None or right_parts is None:
        return None
    ly, lm, ld = left_parts
    ry, rm, rd = right_parts
    return abs((ly * 372 + lm * 31 + ld) - (ry * 372 + rm * 31 + rd))


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_search_text(left), normalize_search_text(right)).ratio()


def events_look_related(left: dict, right: dict) -> bool:
    place_similarity = text_similarity(left.get("place", ""), right.get("place", ""))
    operation_similarity = text_similarity(left.get("operation", ""), right.get("operation", ""))
    date_distance = rough_date_distance_days(left.get("date", ""), right.get("date", ""))

    if place_similarity >= 0.9 and operation_similarity >= 0.75:
        return True
    if place_similarity >= 0.8 and operation_similarity >= 0.9:
        return True
    if date_distance is not None and date_distance <= 5 and place_similarity >= 0.75:
        return True
    if date_distance is not None and date_distance <= 3 and operation_similarity >= 0.8:
        return True
    return False


def build_verification_groups(events: list[dict], max_group_size: int = 6) -> list[list[dict]]:
    normalized_events = [normalize_event(event) for event in events]
    normalized_events.sort(key=lambda item: (item["date"], item["place"], item["operation"]))

    groups: list[list[dict]] = []
    used_indices: set[int] = set()

    for index, event in enumerate(normalized_events):
        if index in used_indices:
            continue

        group = [event]
        used_indices.add(index)

        for other_index in range(index + 1, len(normalized_events)):
            if other_index in used_indices:
                continue
            other_event = normalized_events[other_index]
            if any(events_look_related(existing, other_event) for existing in group):
                group.append(other_event)
                used_indices.add(other_index)
                if len(group) >= max_group_size:
                    break

        groups.append(group)

    return groups
