from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rag.rag_utils import iter_jsonl, normalize_search_text


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


def _event_text_blob(event: dict) -> str:
    parts = [
        str(event.get("operation", "")),
        str(event.get("notes", "")),
        str(event.get("place", "")),
        str(event.get("division", "")),
    ]
    return normalize_search_text(" ".join(part for part in parts if part).strip())


def _event_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_search_text(text), flags=re.UNICODE))


def _pick_better_text(current: str, candidate: str) -> str:
    current = str(current or "").strip()
    candidate = str(candidate or "").strip()
    if not current:
        return candidate
    if not candidate:
        return current
    if len(candidate) > len(current):
        return candidate
    return current


def _combine_notes(left: str, right: str, max_chars: int = 700) -> str:
    snippets: list[str] = []
    for value in (left, right):
        text = " ".join(str(value or "").split()).strip()
        if text and text not in snippets:
            snippets.append(text)
    combined = " ".join(snippets).strip()
    if len(combined) <= max_chars:
        return combined
    return combined[: max_chars - 3].rstrip(" ,;:-") + "..."


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _starts_with_any(text: str, prefixes: list[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _looks_like_movement_title(text: str) -> bool:
    patterns = [
        r"^(?:\d+(?:st|nd|rd|th)\s+)?(?:battalion|company|platoon|patrol)\b.*\b("
        r"transported|moved|returned|withdrawn|deployed|transferred|evacuated|embarked|marched"
        r")\b",
        r"^(?:units of|evacuation of|movement of|transfer of|withdrawal of|withdrawal to|return from|return to)\b",
        r"^(?:brigade|units)\b.*\b(returned|withdrawn|evacuated|moved|transported|transferred)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _looks_like_small_unit_movement(text: str) -> bool:
    small_unit_terms = [
        "battalion",
        "bataljon",
        "company",
        "ceta",
        "platoon",
        "vod",
        "patrol",
        "patrola",
    ]
    movement_terms = [
        "deploy",
        "deployed",
        "deployment",
        "withdrawn",
        "withdrew",
        "transfer",
        "transferred",
        "return",
        "returned",
        "embark",
        "embarked",
        "disposition",
        "reserve",
        "concentrat",
        "relief",
        "redistribution",
        "march",
        "movement",
        "position",
        "holding positions",
        "stationed",
        "garrisoned",
    ]
    return _contains_any(text, small_unit_terms) and _contains_any(text, movement_terms)


def _is_combat_event(event: dict) -> bool:
    text = _event_text_blob(event)
    combat_markers = [
        "battle",
        "combat",
        "fighting",
        "fight",
        "raid",
        "assault",
        "attack",
        "counterattack",
        "landing",
        "landed",
        "repulse",
        "repelled",
        "defen",
        "offensive",
        "operation",
        "liberation",
        "capture",
        "captured",
        "ambush",
        "skirmish",
        "napad",
        "borba",
        "borbe",
        "desant",
        "iskrc",
        "odbran",
        "oslobod",
        "protivnapad",
        "prepad",
        "juris",
    ]
    return _contains_any(text, combat_markers)


def _canonical_place_tokens(place: str) -> set[str]:
    stopwords = {
        "island",
        "otok",
        "area",
        "near",
        "sector",
        "coast",
        "mainland",
        "the",
        "and",
        "of",
        "at",
    }
    return {
        token
        for token in _event_tokens(place)
        if len(token) >= 3 and token not in stopwords and not token.isdigit()
    }


def places_look_same_area(left: str, right: str) -> bool:
    left_norm = normalize_search_text(left).strip()
    right_norm = normalize_search_text(right).strip()
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = _canonical_place_tokens(left)
    right_tokens = _canonical_place_tokens(right)
    if not left_tokens or not right_tokens:
        return False

    overlap = left_tokens & right_tokens
    if overlap:
        return True

    similarity = SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio()
    return similarity >= 0.82


def same_day_same_area(left: dict, right: dict) -> bool:
    if left.get("date", "").strip() != right.get("date", "").strip():
        return False
    return places_look_same_area(left.get("place", ""), right.get("place", ""))


def should_merge_sequential_events(left: dict, right: dict) -> bool:
    if not same_day_same_area(left, right):
        return False
    if not (_is_combat_event(left) and _is_combat_event(right)):
        return False

    operation_similarity = text_similarity(left.get("operation", ""), right.get("operation", ""))
    notes_similarity = text_similarity(left.get("notes", ""), right.get("notes", ""))
    shared_operation_tokens = _event_tokens(left.get("operation", "")) & _event_tokens(right.get("operation", ""))

    if operation_similarity >= 0.55 or notes_similarity >= 0.55:
        return True
    if shared_operation_tokens:
        return True
    if "casualt" in _event_text_blob(left) or "casualt" in _event_text_blob(right):
        return True
    return False


def merge_two_events(left: dict, right: dict) -> dict:
    merged = normalize_event(left)
    candidate = normalize_event(right)

    merged["operation"] = _pick_better_text(merged["operation"], candidate["operation"])
    merged["notes"] = _combine_notes(merged["notes"], candidate["notes"])
    merged["division"] = _pick_better_text(merged["division"], candidate["division"])

    if merged["coordinates"]["lat"] is None and candidate["coordinates"]["lat"] is not None:
        merged["coordinates"] = candidate["coordinates"]
    elif candidate["coordinates"]["lat"] is not None and merged["coordinates"]["lat"] is not None:
        if len(candidate["place"]) > len(merged["place"]):
            merged["coordinates"] = candidate["coordinates"]

    merged["place"] = _pick_better_text(merged["place"], candidate["place"])
    merged["source_chunk_ids"] = sorted(set(merged["source_chunk_ids"]) | set(candidate["source_chunk_ids"]))
    merged["source_pages"] = sorted(set(merged["source_pages"]) | set(candidate["source_pages"]))
    return merged


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


def is_brigade_formation_event(event: dict) -> bool:
    operation_text = normalize_search_text(event.get("operation", ""))
    notes_text = normalize_search_text(event.get("notes", ""))
    combined = f"{operation_text} {notes_text}".strip()
    if "formation" not in combined and "formiranje" not in combined:
        return False
    if "division" in combined or "divizija" in combined:
        if "redesignated" in combined or "redesignation" in combined:
            return False
        if "formation of 26th division" in combined or "formed the 26th division" in combined:
            return False
    if "brigade" in combined or "brigada" in combined:
        return True
    return False


def is_relevant_campaign_event(event: dict) -> bool:
    if is_brigade_formation_event(event):
        return True

    operation_text = normalize_search_text(str(event.get("operation", "")).strip())
    combined = _event_text_blob(event)

    excluded_phrases = [
        "meeting",
        "staff established",
        "headquarters",
        "command handover",
        "appointed brigade commander",
        "replaced by",
        "reorganisation of",
        "reorganization of",
        "redesignated",
        "organisation of island defences",
        "organization of island defences",
        "administrative order",
        "order of the",
        "order issued by",
        "directive",
        "staff with seat",
        "formed the 26th division",
        "formation of 26th division",
        "meeting of coastal command",
        "battalion into four companies",
        "strength report",
        "disposition report",
        "operational hq established",
        "dressing station",
        "hospital",
        "headquarters established",
        "deputy commander",
    ]
    if any(phrase in combined for phrase in excluded_phrases):
        return False

    admin_markers = [
        "handover",
        "appointed",
        "seat in",
        "staff",
        "meeting",
        "reorganisation",
        "reorganization",
        "redesignation",
        "redesignated",
        "directive",
        "order",
        "hospital",
        "strength",
        "disposition",
        "redistribution",
    ]
    if any(marker in combined for marker in admin_markers):
        return False

    excluded_noncombat_terms = [
        "deployment",
        "deployed to",
        "transfer to mainland",
        "transfer to the mainland",
        "transferred to",
        "relocation",
        "concentration of",
        "strength report",
        "disposition",
        "reserve",
        "withdrawn from",
        "stationed at",
        "holding positions",
        "evacuated",
        "embarked",
        "return to",
        "march to",
    ]
    if _contains_any(combined, excluded_noncombat_terms) and not _is_combat_event(event):
        return False

    operation_movement_markers = [
        "rotation of",
        "returned to",
        "return to",
        "withdrawal to",
        "withdrawn to",
        "withdrawn from",
        "transported by",
        "transferred to",
        "transfer to",
        "deployment to",
        "deployed to",
        "concentration of",
        "relocation",
        "redistribution",
        "holding positions",
        "reserve",
        "organis",
        "organiz",
    ]
    if _contains_any(operation_text, operation_movement_markers) and not _is_combat_event(
        {"operation": operation_text}
    ):
        return False

    operation_noncombat_prefixes = [
        "units of",
        "returned to",
        "return to",
        "withdrawal to",
        "withdrawn to",
        "withdrawn from",
        "rotation of",
        "transported by",
        "moved from",
        "movement from",
        "transfer to",
        "transferred to",
        "deployed in",
        "deployed to",
        "concentration of",
    ]
    if _starts_with_any(operation_text, operation_noncombat_prefixes):
        return False

    if _looks_like_movement_title(operation_text):
        return False

    if _looks_like_small_unit_movement(combined) and not _is_combat_event(event):
        return False

    if not _is_combat_event({"operation": operation_text, "notes": operation_text}):
        return False

    return True


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

    sorted_events = sorted(merged.values(), key=event_sort_key)
    if not sorted_events:
        return []

    sequentially_merged: list[dict] = [sorted_events[0]]
    for event in sorted_events[1:]:
        previous = sequentially_merged[-1]
        if should_merge_sequential_events(previous, event):
            sequentially_merged[-1] = merge_two_events(previous, event)
            continue
        sequentially_merged.append(event)

    return sequentially_merged


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
    fallback_document: dict | None = None,
) -> dict:
    final_notes = " | ".join(note for note in top_notes if note.strip())
    fallback_document = fallback_document or {}
    brigade_id = template.get("brigade_id")
    if brigade_id is None:
        brigade_id = fallback_document.get("brigade_id")

    brigade_name = template.get("brigade_name") or fallback_document.get("brigade_name", "")
    notes_value = final_notes or template.get("notes", "") or fallback_document.get("notes", "")
    source_value = source_label or template.get("source", "") or fallback_document.get("source", "")
    return {
        "brigade_id": brigade_id,
        "brigade_name": brigade_name,
        "movements": [strip_event_to_template(event) for event in merged_events],
        "notes": notes_value,
        "source": source_value,
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
    if should_merge_sequential_events(left, right):
        return True

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
    current_group: list[dict] = []

    for index, event in enumerate(normalized_events):
        previous_event = normalized_events[index - 1] if index > 0 else None
        next_event = normalized_events[index + 1] if index + 1 < len(normalized_events) else None

        related_to_neighbors = any(
            neighbor is not None and events_look_related(event, neighbor)
            for neighbor in (previous_event, next_event)
        )
        related_to_group = any(events_look_related(event, existing) for existing in current_group)

        if current_group and (
            len(current_group) >= max_group_size or (not related_to_group and not related_to_neighbors)
        ):
            groups.append(current_group)
            current_group = []

        current_group.append(event)

    if current_group:
        groups.append(current_group)

    return groups
