from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber


def slugify_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("_", " ")
    ascii_name = re.sub(r"[^\w\s.-]", "", ascii_name)
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip(" .")
    return ascii_name or "book"


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def extract_pdf_to_jsonl(pdf_path: Path, output_root: Path) -> tuple[Path, Path]:
    book_title = pdf_path.stem
    book_dir = output_root / slugify_filename(book_title)
    book_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = book_dir / "pages.jsonl"
    metadata_path = book_dir / "metadata.json"

    with pdfplumber.open(pdf_path) as pdf, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        page_count = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            record = {
                "page_number": page_number,
                "text": text,
                "word_count": count_words(text),
            }
            json.dump(record, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    metadata = {
        "source_pdf": str(pdf_path.resolve()),
        "book_title": book_title,
        "page_count": page_count,
        "output_jsonl": str(jsonl_path.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return jsonl_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a PDF to JSONL one page at a time with pdfplumber."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./AI Processing"),
        help="Root folder for extracted output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf_path.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    jsonl_path, metadata_path = extract_pdf_to_jsonl(pdf_path, args.output_root)
    print(f"Wrote JSONL to: {jsonl_path}")
    print(f"Wrote metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
