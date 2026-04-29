from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group pages.jsonl records into overlapping word chunks."
    )
    parser.add_argument(
        "pages_jsonl",
        type=Path,
        help="Path to the input pages.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to chunks.jsonl next to pages.jsonl.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=800,
        help="Target number of words per chunk.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=120,
        help="Number of overlapping words between consecutive chunks.",
    )
    return parser.parse_args()


def unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def write_chunk(output_file, chunk_id: int, word_entries: list[tuple[str, int]]) -> None:
    text = " ".join(word for word, _page_number in word_entries)
    source_pages = unique_in_order([page_number for _word, page_number in word_entries])
    record = {
        "chunk_id": chunk_id,
        "source_pages": source_pages,
        "text": text,
        "word_count": len(word_entries),
    }
    json.dump(record, output_file, ensure_ascii=False)
    output_file.write("\n")


def chunk_pages_jsonl(
    pages_jsonl_path: Path,
    output_path: Path,
    chunk_words: int,
    overlap_words: int,
) -> Path:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    step_words = chunk_words - overlap_words
    buffer: list[tuple[str, int]] = []
    chunk_id = 1

    with pages_jsonl_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            page_record = json.loads(line)
            page_number = page_record["page_number"]
            page_words = page_record.get("text", "").split()

            for word in page_words:
                buffer.append((word, page_number))

            while len(buffer) >= chunk_words:
                write_chunk(output_file, chunk_id, buffer[:chunk_words])
                chunk_id += 1
                buffer = buffer[step_words:]

        if buffer and (chunk_id == 1 or len(buffer) > overlap_words):
            write_chunk(output_file, chunk_id, buffer)

    return output_path


def main() -> None:
    args = parse_args()
    pages_jsonl_path = args.pages_jsonl.resolve()

    if not pages_jsonl_path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {pages_jsonl_path}")

    output_path = args.output.resolve() if args.output else pages_jsonl_path.with_name(
        "chunks.jsonl"
    )
    chunk_pages_jsonl(
        pages_jsonl_path=pages_jsonl_path,
        output_path=output_path,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    print(f"Wrote chunks to: {output_path}")


if __name__ == "__main__":
    main()
