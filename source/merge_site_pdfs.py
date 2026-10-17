#!/usr/bin/env python3
"""
merge_site_pdfs.py

Inspect a web page, find all linked PDF files, download them, merge them
(in the order they appear on the page) into a single PDF, and save it in
the current working directory.

Usage:
    python merge_site_pdfs.py https://znaci.org/00001/72.htm
    python merge_site_pdfs.py https://znaci.org/00001/72.htm -o book.pdf
    python merge_site_pdfs.py https://znaci.org/00001/72.htm --keep-parts

Requirements:
    pip install requests beautifulsoup4 pypdf
"""

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pdf-merger/1.0)"}
TIMEOUT = 60


def fetch_html(url: str) -> tuple[str, str]:
    """Download the page and return (html_text, final_url)."""
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    # Many older sites (e.g. Balkan ones) don't declare encoding properly.
    if r.encoding is None or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text, r.url


def find_pdf_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Return [(absolute_url, link_text)] for every PDF link, in page order,
    with duplicates removed."""
    soup = BeautifulSoup(html, "html.parser")
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absolute = urljoin(base_url, href)
        path = urlparse(absolute).path.lower()
        if not path.endswith(".pdf"):
            continue
        absolute = absolute.split("#")[0]
        if absolute in seen:
            continue
        seen.add(absolute)
        text = " ".join(a.get_text(" ", strip=True).split())
        links.append((absolute, text))
    return links


def download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download url to dest. Returns True if a valid PDF was saved."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            with open(dest, "rb") as f:
                if f.read(5) != b"%PDF-":
                    print(f"   ! not a real PDF, skipping: {url}")
                    return False
            return True
        except requests.RequestException as e:
            print(f"   ! attempt {attempt}/{retries} failed: {e}")
            time.sleep(1.5 * attempt)
    return False


def safe_name(s: str, fallback: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", s).strip(" ._")
    return s or fallback


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and merge all PDFs linked from a page.")
    ap.add_argument("url", help="Page to inspect")
    ap.add_argument("-o", "--output", help="Output filename (default: derived from page URL)")
    ap.add_argument("--keep-parts", action="store_true",
                    help="Also keep the individual PDFs in a subfolder")
    ap.add_argument("--no-bookmarks", action="store_true",
                    help="Don't add a bookmark (outline entry) per source PDF")
    args = ap.parse_args()

    print(f"Inspecting {args.url} ...")
    try:
        html, final_url = fetch_html(args.url)
    except requests.RequestException as e:
        print(f"Could not load page: {e}")
        return 1

    links = find_pdf_links(html, final_url)
    if not links:
        print("No PDF links found on that page.")
        return 1
    print(f"Found {len(links)} PDF link(s).")

    stem = Path(urlparse(final_url).path).stem or "merged"
    out_name = args.output or f"{stem}_merged.pdf"
    if not out_name.lower().endswith(".pdf"):
        out_name += ".pdf"
    out_path = Path.cwd() / out_name

    parts_dir = Path.cwd() / f"{stem}_parts" if args.keep_parts else None
    if parts_dir:
        parts_dir.mkdir(exist_ok=True)

    writer = PdfWriter()
    merged_count = 0

    with tempfile.TemporaryDirectory() as tmp:
        for i, (url, text) in enumerate(links, 1):
            fname = unquote(Path(urlparse(url).path).name) or f"part_{i}.pdf"
            target = (parts_dir or Path(tmp)) / fname
            print(f"[{i}/{len(links)}] {fname}")

            if not download(url, target):
                print("   -> skipped")
                continue

            try:
                reader = PdfReader(str(target))
                if reader.is_encrypted:
                    reader.decrypt("")
                start_page = len(writer.pages)
                for page in reader.pages:
                    writer.add_page(page)
            except Exception as e:  # corrupt / unreadable PDF
                print(f"   ! could not read PDF ({e}), skipped")
                continue

            if not args.no_bookmarks:
                title = text or Path(fname).stem
                writer.add_outline_item(title[:200], start_page)
            merged_count += 1

        if merged_count == 0:
            print("Nothing could be downloaded/merged.")
            return 1

        with open(out_path, "wb") as f:
            writer.write(f)

    print(f"\nMerged {merged_count}/{len(links)} PDFs "
          f"({len(writer.pages)} pages) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
