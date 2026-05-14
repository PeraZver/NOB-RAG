from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FOOTNOTE_START_RE = re.compile(
    r"^\s*(?:\^?\s*)?(?:\d{1,2}|I\d{1,2})[a-z]?[)\.'\"]\s+[A-ZČĆŽŠĐ]"
)
INLINE_FOOTNOTE_RE = re.compile(
    r"(?P<main>.{40,}?[\.\!\?\"'])\s*(?P<footnote>(?:\d{1,2}|I\d{1,2})[a-z]?[)\.'\"]\s+[A-ZČĆŽŠĐ].*)$"
)
PURE_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
UPPER_HEADING_RE = re.compile(r"^[A-ZČĆŽŠĐ0-9 .,:;()\-\"«»]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean pages.jsonl for better retrieval and chunking."
    )
    parser.add_argument("pages_jsonl", type=Path, help="Path to the input pages.jsonl file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to cleaned_pages.jsonl next to pages.jsonl.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path. Defaults to cleaning_report.json next to pages.jsonl.",
    )
    return parser.parse_args()


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped.split()) <= 8 and UPPER_HEADING_RE.fullmatch(stripped):
        return True
    return False


def looks_like_footnote_start(line: str, line_index: int, total_lines: int) -> bool:
    if line_index < total_lines // 2:
        return False
    return bool(FOOTNOTE_START_RE.match(line))


def split_main_and_footnotes(lines: list[str]) -> tuple[list[str], list[str]]:
    main_lines: list[str] = []
    footnote_lines: list[str] = []
    in_footnotes = False

    for index, line in enumerate(lines):
        if PURE_PAGE_NUMBER_RE.match(line):
            continue

        inline_match = INLINE_FOOTNOTE_RE.match(line)
        if inline_match and not in_footnotes:
            main_part = inline_match.group("main").rstrip()
            footnote_part = inline_match.group("footnote").strip()
            if main_part:
                main_lines.append(main_part)
            if footnote_part:
                footnote_lines.append(footnote_part)
                in_footnotes = True
            continue

        if not in_footnotes and looks_like_footnote_start(line, index, len(lines)):
            in_footnotes = True

        if in_footnotes:
            footnote_lines.append(line)
        else:
            main_lines.append(line)

    return main_lines, footnote_lines


def clean_paragraph_lines(lines: list[str]) -> tuple[str, int, int]:
    cleaned_parts: list[str] = []
    current = ""
    dehyphenations = 0
    merged_lines = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                cleaned_parts.append(current.strip())
                current = ""
            continue

        if is_heading_line(line):
            if current:
                cleaned_parts.append(current.strip())
                current = ""
            cleaned_parts.append(line)
            continue

        if not current:
            current = line
            continue

        if current.endswith("-") and line[:1].islower():
            current = current[:-1] + line
            dehyphenations += 1
            merged_lines += 1
            continue

        current = current + " " + line
        merged_lines += 1

    if current:
        cleaned_parts.append(current.strip())

    cleaned_text = "\n\n".join(part for part in cleaned_parts if part)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text, dehyphenations, merged_lines


def clean_pages_jsonl(
    pages_jsonl_path: Path,
    output_path: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    report = {
        "pages_processed": 0,
        "pages_with_footnotes_removed": 0,
        "total_footnote_lines_removed": 0,
        "total_dehyphenations": 0,
        "total_line_merges": 0,
    }

    with pages_jsonl_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            record = json.loads(line)
            raw_text = record.get("text", "")
            raw_lines = raw_text.splitlines()
            main_lines, footnote_lines = split_main_and_footnotes(raw_lines)
            cleaned_text, dehyphenations, merged_lines = clean_paragraph_lines(main_lines)

            cleaned_record = {
                "page_number": record["page_number"],
                "text": cleaned_text,
                "word_count": count_words(cleaned_text),
            }
            if footnote_lines:
                cleaned_record["footnotes"] = "\n".join(line.strip() for line in footnote_lines if line.strip())

            json.dump(cleaned_record, output_file, ensure_ascii=False)
            output_file.write("\n")

            report["pages_processed"] += 1
            report["total_dehyphenations"] += dehyphenations
            report["total_line_merges"] += merged_lines
            report["total_footnote_lines_removed"] += len(footnote_lines)
            if footnote_lines:
                report["pages_with_footnotes_removed"] += 1

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, report_path


def main() -> None:
    args = parse_args()
    pages_jsonl_path = args.pages_jsonl.resolve()

    if not pages_jsonl_path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {pages_jsonl_path}")

    output_path = args.output.resolve() if args.output else pages_jsonl_path.with_name(
        "cleaned_pages.jsonl"
    )
    report_path = args.report.resolve() if args.report else pages_jsonl_path.with_name(
        "cleaning_report.json"
    )
    output_path, report_path = clean_pages_jsonl(
        pages_jsonl_path=pages_jsonl_path,
        output_path=output_path,
        report_path=report_path,
    )
    print(f"Wrote cleaned pages to: {output_path}")
    print(f"Wrote cleaning report to: {report_path}")


if __name__ == "__main__":
    main()
