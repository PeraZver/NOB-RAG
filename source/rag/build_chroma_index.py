from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.rag_utils import chroma_collection_name, infer_book_title, iter_jsonl, parse_source_pages

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local persistent Chroma index from chunks.jsonl."
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Directory containing chunks.jsonl and metadata.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="SentenceTransformer model name.",
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
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for chunk ingestion.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before indexing.",
    )
    return parser.parse_args()


def flush_batch(collection, embedder, batch: list[dict]) -> int:
    if not batch:
        return 0

    documents = [item["text"] for item in batch]
    embeddings = embedder.encode(documents, batch_size=len(documents)).tolist()
    ids = [f"chunk-{item['chunk_id']}" for item in batch]
    metadatas = []

    for item in batch:
        source_pages = parse_source_pages(item.get("source_pages", []))
        metadatas.append(
            {
                "chunk_id": int(item["chunk_id"]),
                "word_count": int(item["word_count"]),
                "source_pages": ",".join(str(page) for page in source_pages),
                "page_start": int(source_pages[0]) if source_pages else -1,
                "page_end": int(source_pages[-1]) if source_pages else -1,
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(batch)


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


def load_embedder(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def build_index(
    book_dir: Path,
    embedding_model: str,
    collection_name: str | None,
    db_dir: Path | None,
    batch_size: int,
    reset: bool,
) -> tuple[str, Path, int]:
    chunks_path = book_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    title = infer_book_title(book_dir)
    collection_name = collection_name or chroma_collection_name(title)
    db_dir = db_dir or (book_dir / "chroma_db")
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
    batch: list[dict] = []

    for record in iter_jsonl(chunks_path):
        chunk_id = f"chunk-{record['chunk_id']}"
        if chunk_id in existing_ids:
            skipped += 1
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            indexed += flush_batch(collection, embedder, batch)
            batch = []

    indexed += flush_batch(collection, embedder, batch)
    return collection_name, db_dir, indexed, skipped


def main() -> None:
    args = parse_args()
    book_dir = args.book_dir.resolve()
    if not book_dir.exists():
        raise FileNotFoundError(f"book directory not found: {book_dir}")

    collection_name, db_dir, indexed, skipped = build_index(
        book_dir=book_dir,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        db_dir=Path(args.db_dir).resolve() if args.db_dir else None,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Indexed {indexed} chunks")
    print(f"Skipped {skipped} existing chunks")
    print(f"Collection: {collection_name}")
    print(f"Chroma DB: {db_dir}")


if __name__ == "__main__":
    main()
