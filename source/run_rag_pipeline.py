#!/usr/bin/env python3
"""
Run the full PDF -> chunks -> Chroma pipeline, stopping on the first error.

Usage (from your project root, where `text_processing` and `rag` are importable):

    python run_pipeline.py ..\\pdf_dir\\pdf_file.pdf --max-pages 20
    python run_pipeline.py ..\\pdf_dir\\pdf_file.pdf --max-pages 20 --output-root ..

Optional:
    --chunk-dir PATH   Use this chunk directory instead of auto-detecting it
                       (auto-detect = the newest pages.jsonl created under
                       --output-root by step 1).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, module_args):
    """Run `python -m <module> ...`; abort the whole pipeline on failure."""
    cmd = [sys.executable, "-m", *[str(a) for a in module_args]]
    print(f"\n=== {name} ===")
    print("> " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: step '{name}' failed with exit code {result.returncode}. "
              f"Pipeline stopped.", file=sys.stderr)
        sys.exit(result.returncode)


def detect_chunk_dir(output_root, started_at):
    """Find the directory of the newest pages.jsonl written since `started_at`."""
    candidates = [
        p for p in Path(output_root).rglob("pages.jsonl")
        if p.stat().st_mtime >= started_at - 2  # small tolerance for clock granularity
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.parent


def main():
    ap = argparse.ArgumentParser(description="PDF -> chunks -> Chroma pipeline")
    ap.add_argument("pdf", help="Path to the source PDF")
    ap.add_argument("--max-pages", type=int, required=True,
                    help="Number of pages to process")
    ap.add_argument("--output-root", default="..",
                    help="Output root passed to extract_pdf_to_jsonl (default: ..)")
    ap.add_argument("--chunk-dir", default=None,
                    help="Chunk directory (skip auto-detection)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for campaign extraction (default: 8)")
    ap.add_argument("--provider", default="openai",
                    help="LLM provider for campaign steps (default: openai)")
    args = ap.parse_args()

    started_at = time.time()

    # 1. Extract PDF pages -> pages.jsonl (creates the chunk directory)
    run_step("Extract PDF", [
        "text_processing.extract_pdf_to_jsonl", args.pdf,
        "--output-root", args.output_root,
        "--max-pages", args.max_pages,
    ])

    # Determine chunk directory
    if args.chunk_dir:
        chunk_dir = Path(args.chunk_dir)
    else:
        chunk_dir = detect_chunk_dir(args.output_root, started_at)
        if chunk_dir is None:
            print("ERROR: could not find a pages.jsonl created by step 1 under "
                  f"'{args.output_root}'. Re-run with --chunk-dir.", file=sys.stderr)
            sys.exit(1)
    print(f"\nUsing chunk directory: {chunk_dir}")

    # 2. Clean, chunk, validate
    run_step("Clean pages", ["text_processing.clean_pages_jsonl",
                             chunk_dir / "pages.jsonl"])
    run_step("Chunk pages", ["text_processing.chunk_pages_jsonl",
                             chunk_dir / "cleaned_pages.jsonl"])
    run_step("Validate chunks", ["text_processing.validate_chunks_jsonl",
                                 chunk_dir / "chunks.jsonl"])

    # 3. Build Chroma indexes
    run_step("Build Chroma index", ["rag.build_chroma_index", chunk_dir, "--reset"])
    run_step("Build shared Chroma index", ["rag.build_shared_chroma_index"])

    # 4. Campaign extraction, verification, grouping
    run_step("Build campaign JSON", [
        "campaigns.build_brigade_campaign_json", "run", chunk_dir,
        "--batch-size", args.batch_size, "--provider", args.provider,
    ])
    run_step("Verify campaign JSON", [
        "campaigns.verify_brigade_campaign_json", "run", chunk_dir,
        "--provider", args.provider,
    ])
    run_step("Group verified campaign JSON", [
        "campaigns.group_verified_campaign_json", chunk_dir,
        "--provider", args.provider,
    ])

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
