from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate chunks.jsonl and print a chunk-size summary."
    )
    parser.add_argument(
        "chunks_jsonl",
        type=Path,
        help="Path to the input chunks.jsonl file.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=200,
        help="Flag chunks below this word count threshold.",
    )
    parser.add_argument(
        "--expected-overlap",
        type=int,
        default=120,
        help="Expected word overlap between adjacent chunks.",
    )
    return parser.parse_args()


def validate_chunks(chunks_jsonl_path: Path, min_words: int, expected_overlap: int) -> None:
    total_chunks = 0
    total_words = 0
    min_chunk_words: int | None = None
    max_chunk_words: int | None = None
    short_chunks: list[tuple[int, int, list[int]]] = []
    empty_text_chunks: list[tuple[int, list[int]]] = []
    word_count_mismatches: list[tuple[int, int, int]] = []
    chunk_id_gaps: list[tuple[int, int]] = []
    overlap_issues: list[tuple[int, int, int, int]] = []
    previous_record: dict[str, object] | None = None
    expected_chunk_id = 1

    with chunks_jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            chunk_id = record["chunk_id"]
            word_count = record["word_count"]
            source_pages = record.get("source_pages", [])
            text = record.get("text", "")
            actual_word_count = len(text.split()) if text else 0

            total_chunks += 1
            total_words += word_count

            if min_chunk_words is None or word_count < min_chunk_words:
                min_chunk_words = word_count
            if max_chunk_words is None or word_count > max_chunk_words:
                max_chunk_words = word_count

            if word_count < min_words:
                short_chunks.append((chunk_id, word_count, source_pages))
            if not text.strip():
                empty_text_chunks.append((chunk_id, source_pages))
            if word_count != actual_word_count:
                word_count_mismatches.append((chunk_id, word_count, actual_word_count))
            if chunk_id != expected_chunk_id:
                chunk_id_gaps.append((expected_chunk_id, chunk_id))
                expected_chunk_id = chunk_id

            if previous_record is not None and expected_overlap > 0:
                previous_text = str(previous_record["text"])
                previous_chunk_id = int(previous_record["chunk_id"])
                previous_words = previous_text.split()
                current_words = text.split()
                overlap_size = min(
                    expected_overlap,
                    len(previous_words),
                    len(current_words),
                )
                if overlap_size > 0:
                    if previous_words[-overlap_size:] != current_words[:overlap_size]:
                        overlap_issues.append(
                            (
                                previous_chunk_id,
                                chunk_id,
                                overlap_size,
                                min(len(previous_words), len(current_words)),
                            )
                        )

            expected_chunk_id += 1
            previous_record = {"chunk_id": chunk_id, "text": text}

    avg_chunk_words = (total_words / total_chunks) if total_chunks else 0.0

    print(f"Total chunks: {total_chunks}")
    print(f"Min chunk size (words): {min_chunk_words or 0}")
    print(f"Max chunk size (words): {max_chunk_words or 0}")
    print(f"Avg chunk size (words): {avg_chunk_words:.2f}")

    if short_chunks:
        print(f"Chunks under {min_words} words:")
        for chunk_id, word_count, source_pages in short_chunks:
            print(
                f"  chunk_id={chunk_id} word_count={word_count} "
                f"source_pages={source_pages}"
            )
    else:
        print(f"No chunks under {min_words} words.")

    if empty_text_chunks:
        print("Chunks with empty text:")
        for chunk_id, source_pages in empty_text_chunks:
            print(f"  chunk_id={chunk_id} source_pages={source_pages}")
    else:
        print("No chunks with empty text.")

    if word_count_mismatches:
        print("Chunks with word_count mismatches:")
        for chunk_id, stored_word_count, actual_word_count in word_count_mismatches:
            print(
                f"  chunk_id={chunk_id} stored_word_count={stored_word_count} "
                f"actual_word_count={actual_word_count}"
            )
    else:
        print("No word_count mismatches.")

    if chunk_id_gaps:
        print("Non-sequential chunk_id values:")
        for expected_id, actual_id in chunk_id_gaps:
            print(f"  expected_chunk_id={expected_id} actual_chunk_id={actual_id}")
    else:
        print("All chunk_id values are sequential.")

    if overlap_issues:
        print(f"Chunks with broken {expected_overlap}-word overlap:")
        for previous_chunk_id, chunk_id, checked_overlap, comparable_words in overlap_issues:
            print(
                f"  previous_chunk_id={previous_chunk_id} chunk_id={chunk_id} "
                f"checked_overlap={checked_overlap} comparable_words={comparable_words}"
            )
    else:
        print(f"All adjacent chunks preserve the expected {expected_overlap}-word overlap.")


def main() -> None:
    args = parse_args()
    chunks_jsonl_path = args.chunks_jsonl.resolve()

    if not chunks_jsonl_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_jsonl_path}")

    validate_chunks(chunks_jsonl_path, args.min_words, args.expected_overlap)


if __name__ == "__main__":
    main()
