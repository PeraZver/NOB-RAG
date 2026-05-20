from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.rag_utils import (
    IndexedBook,
    build_chunk_doc_id,
    discover_books,
    iter_jsonl,
    parse_source_pages,
    resolve_books_root,
    shared_collection_name,
    shared_db_dir,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one shared Chroma index across all discovered book folders."
    )
    parser.add_argument(
        "--books-root",
        type=Path,
        default=None,
        help="Root directory containing book folders. Defaults to the directory next to source/.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--collection",
        default=shared_collection_name(),
        help="Chroma collection name for the shared library.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Persistent Chroma directory. Defaults to <books_root>/shared_chroma_db.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for chunk ingestion.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the shared collection from scratch.",
    )
    return parser.parse_args()


def load_embedder(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def get_existing_ids(collection, page_size: int = 1000) -> set[str]:
    existing_ids: set[str] = set()
    offset = 0

    while True:
        result = collection.get(include=[], limit=page_size, offset=offset)
        ids = result.get("ids", [])
        if not ids:
            break
        existing_ids.update(ids)
        if len(ids) < page_size:
            break
        offset += page_size

    return existing_ids


def flush_batch(collection, embedder, batch: list[dict]) -> int:
    if not batch:
        return 0

    documents = [item["text"] for item in batch]
    embeddings = embedder.encode(documents, batch_size=len(documents)).tolist()
    ids = [item["doc_id"] for item in batch]
    metadatas = [item["metadata"] for item in batch]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(batch)


def build_shared_index(
    books_root: Path,
    embedding_model: str,
    collection_name: str,
    db_dir: Path,
    batch_size: int,
    reset: bool,
) -> tuple[int, int, int]:
    books = discover_books(books_root)
    if not books:
        raise FileNotFoundError(f"No book folders with chunks.jsonl found in {books_root}")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    embedder = load_embedder(embedding_model)
    existing_ids = set() if reset else get_existing_ids(collection)

    indexed = 0
    skipped = 0
    total_books = 0
    batch: list[dict] = []

    for book in books:
        total_books += 1
        for record in iter_jsonl(book.chunks_path):
            chunk_id = int(record["chunk_id"])
            doc_id = build_chunk_doc_id(book.slug, chunk_id)
            if doc_id in existing_ids:
                skipped += 1
                continue
            source_pages = parse_source_pages(record.get("source_pages", []))
            batch.append(
                {
                    "doc_id": doc_id,
                    "text": record["text"],
                    "metadata": {
                        "doc_id": doc_id,
                        "book_title": book.title,
                        "book_slug": book.slug,
                        "book_dir": str(book.path),
                        "source_pdf": book.source_pdf or "",
                        "chunk_id": chunk_id,
                        "word_count": int(record["word_count"]),
                        "source_pages": ",".join(str(page) for page in source_pages),
                        "page_start": int(source_pages[0]) if source_pages else -1,
                        "page_end": int(source_pages[-1]) if source_pages else -1,
                    },
                }
            )
            if len(batch) >= batch_size:
                indexed += flush_batch(collection, embedder, batch)
                batch = []

    indexed += flush_batch(collection, embedder, batch)
    return total_books, indexed, skipped


def main() -> None:
    args = parse_args()
    books_root = (args.books_root or resolve_books_root()).resolve()
    db_dir = (args.db_dir or shared_db_dir(books_root)).resolve()
    total_books, indexed, skipped = build_shared_index(
        books_root=books_root,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        db_dir=db_dir,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Books scanned: {total_books}")
    print(f"Indexed {indexed} chunks")
    print(f"Skipped {skipped} existing chunks")
    print(f"Collection: {args.collection}")
    print(f"Shared Chroma DB: {db_dir}")


if __name__ == "__main__":
    main()
