from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rag.rag_env import load_local_env, resolve_provider_model
from rag.rag_profiles import resolve_query_profile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an interactive local RAG session that keeps models and Chroma loaded."
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Directory containing chunks.jsonl and the persistent Chroma index.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=os.getenv("RAG_PROVIDER", "anthropic"),
        help="Answer-generation provider to use after retrieval.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="SentenceTransformer model name used for querying.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional provider model override.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional Chroma collection name override.",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Optional persistent Chroma directory. Defaults to <book_dir>/chroma_db.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of retrieved chunks to send to the model.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=40,
        help="Number of chunks to collect before reranking.",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip cross-encoder reranking and use hybrid retrieval order directly.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster retrieval preset: smaller candidate set, fewer final chunks, no reranking.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use a deeper retrieval preset: larger candidate set, more final chunks, reranking on.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source chunks after each answer.",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Skip the provider call and print only the retrieved source chunks.",
    )
    return parser.parse_args()


def main() -> None:
    from rag.rag_engine import BookRAGEngine

    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")
    load_local_env(book_dir)
    profile = resolve_query_profile(args.fast, args.deep)

    top_k = args.top_k
    candidate_k = args.candidate_k
    skip_rerank = args.skip_rerank
    if profile is not None:
        top_k = profile.top_k
        candidate_k = profile.candidate_k
        skip_rerank = profile.skip_rerank

    engine = BookRAGEngine(
        book_dir=book_dir,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        collection_name=args.collection,
        db_dir=Path(args.db_dir).resolve() if args.db_dir else None,
    )
    engine.preload(include_reranker=not skip_rerank)

    resolved_model = None if args.retrieve_only else resolve_provider_model(args.provider, args.model)

    print(f"Loaded book: {engine.title}")
    if args.retrieve_only:
        print("Interactive retrieve-only mode. Press Ctrl+C to stop.")
    else:
        print(
            f"Interactive answer mode with provider={args.provider}, model={resolved_model}. "
            "Press Ctrl+C to stop."
        )
    if profile is not None:
        print(
            f"Query profile: {profile.name} "
            f"(top_k={top_k}, candidate_k={candidate_k}, skip_rerank={skip_rerank})"
        )

    try:
        while True:
            question = input("\nQuestion> ").strip()
            if not question:
                continue

            if args.retrieve_only:
                retrieval = engine.retrieve_only(
                    question=question,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    skip_rerank=skip_rerank,
                )
                for summary in retrieval.source_summaries:
                    print(summary)
                continue

            result = engine.answer_question(
                question=question,
                provider=args.provider,
                model=resolved_model,
                top_k=top_k,
                candidate_k=candidate_k,
                skip_rerank=skip_rerank,
            )
            print()
            print(result.answer.strip())
            if args.show_sources:
                print("\nRetrieved sources:")
                for summary in result.retrieval.source_summaries:
                    print(summary)
    except (KeyboardInterrupt, EOFError):
        print("\nStopping interactive session.")


if __name__ == "__main__":
    main()
