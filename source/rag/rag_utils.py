from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_page_number(record: dict) -> int:
    if "page_number" in record:
        return int(record["page_number"])
    if "page" in record:
        return int(record["page"])
    raise KeyError("Record is missing 'page_number'/'page'")


def load_pages_map(pages_jsonl_path: Path) -> dict[int, str]:
    pages: dict[int, str] = {}
    for record in iter_jsonl(pages_jsonl_path):
        pages[get_page_number(record)] = record.get("text", "")
    return pages


def load_metadata(book_dir: Path) -> dict:
    metadata_path = book_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def infer_book_title(book_dir: Path) -> str:
    metadata = load_metadata(book_dir)
    title = metadata.get("book_title")
    if isinstance(title, str) and title.strip():
        return title
    return book_dir.name


def slugify_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().replace("_", "-")
    ascii_text = re.sub(r"[^a-z0-9-]+", "-", ascii_text)
    ascii_text = ascii_text.strip("-")
    if not ascii_text:
        ascii_text = "book"
    if len(ascii_text) < 3:
        ascii_text = f"{ascii_text}-rag"
    return ascii_text[:63]


def chroma_collection_name(book_title: str) -> str:
    name = slugify_name(book_title)
    if not name[0].isalnum():
        name = f"b-{name}"
    if not name[-1].isalnum():
        name = f"{name}0"
    return name[:63]


def parse_source_pages(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [int(item) for item in value.split(",") if item.strip()]
    return []


def format_page_citation(pages: list[int]) -> str:
    if not pages:
        return "page unknown"

    ranges: list[tuple[int, int]] = []
    start = pages[0]
    end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append((start, end))
        start = page
        end = page
    ranges.append((start, end))

    labels = [
        f"{range_start}-{range_end}" if range_start != range_end else str(range_start)
        for range_start, range_end in ranges
    ]
    prefix = "pp." if len(labels) > 1 or ranges[0][0] != ranges[0][1] else "p."
    return f"{prefix} {', '.join(labels)}"


def preview_text(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def normalize_search_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def tokenize_search_text(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_search_text(text), flags=re.UNICODE)


def infer_chunk_features(chunk_id: int, source_pages: list[int], text: str) -> dict[str, object]:
    normalized_text = normalize_search_text(text)
    first_page = source_pages[0] if source_pages else -1
    page_span = len(source_pages)
    compact = " ".join(text.split())
    snippet = compact[:1200]

    section_terms = [
        "prva glava",
        "druga glava",
        "treca glava",
        "cetvrta glava",
        "peta glava",
        "sesta glava",
        "sedma glava",
        "osma glava",
        "deveta glava",
        "deseta glava",
        "jedanaesta glava",
        "dvanaesta glava",
        "predgovor",
        "rijec autora",
        "uvodne napomene",
        "sadrzaj",
    ]
    structure_terms = [
        "osnivanje",
        "formiranje",
        "brigada",
        "stab",
        "staba",
        "komanda",
        "komandant",
        "komesara",
        "komesar",
        "bataljon",
        "ceta",
        "divizije",
        "korpusa",
    ]
    chronology_terms = [
        "1941",
        "1942",
        "1943",
        "1944",
        "1945",
        "oktobra",
        "septembra",
        "januara",
        "marta",
        "aprila",
        "maja",
        "juna",
        "jula",
        "augusta",
        "novembra",
        "decembra",
    ]
    operational_terms = [
        "napad",
        "borba",
        "borbe",
        "dejstva",
        "operacija",
        "oslobodenje",
        "oslobodenja",
        "desant",
        "iskrcavanje",
        "odbrana",
        "obrana",
        "prodor",
        "povlacenje",
        "evakuacija",
        "evakuirane",
        "vis",
        "brac",
        "korcula",
        "peljesac",
        "split",
        "sibenik",
        "knin",
        "mostar",
        "bihac",
        "gospic",
        "krk",
        "trst",
        "celovec",
    ]

    is_contents = "sadrzaj" in normalized_text and "strana" in normalized_text
    has_section_heading = any(term in normalized_text for term in section_terms)
    has_structure_terms = sum(term in normalized_text for term in structure_terms) >= 3
    has_chronology_terms = sum(term in normalized_text for term in chronology_terms) >= 3
    has_operational_terms = sum(term in normalized_text for term in operational_terms) >= 3
    is_early_section = 0 < first_page <= 80
    is_appendix_like = first_page >= 470
    looks_list_like = snippet.count(";") >= 8 or snippet.count(" - ") >= 8

    return {
        "first_page": first_page,
        "page_span": page_span,
        "is_contents": is_contents,
        "has_section_heading": has_section_heading,
        "has_structure_terms": has_structure_terms,
        "has_chronology_terms": has_chronology_terms,
        "has_operational_terms": has_operational_terms,
        "is_early_section": is_early_section,
        "is_appendix_like": is_appendix_like,
        "looks_list_like": looks_list_like,
        "chunk_position_ratio": chunk_id,
    }
