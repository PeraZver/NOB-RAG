from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.rag_engine import BookRAGEngine
from rag.rag_env import load_local_env, resolve_provider_model
from rag.rag_profiles import resolve_query_profile
from rag.rag_utils import infer_book_title, load_metadata, slugify_name


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "webapp_static"


@dataclass(frozen=True)
class BookRecord:
    slug: str
    title: str
    path: Path
    chunk_count: int | None
    source_pdf: str | None


class ChatRequest(BaseModel):
    book_slug: str
    question: str = Field(min_length=1)
    provider: str = Field(default=os.getenv("RAG_PROVIDER", "anthropic"))
    profile: str = Field(default="fast")
    retrieve_only: bool = False
    model: str | None = None


def _count_chunks(chunks_path: Path) -> int | None:
    try:
        with chunks_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def resolve_books_root() -> Path:
    explicit_root = os.getenv("RAG_BOOKS_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return SOURCE_DIR.parent.resolve()


def _is_book_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name != "source"
        and (path / "chunks.jsonl").exists()
        and (path / "chroma_db").is_dir()
    )


@lru_cache(maxsize=1)
def discover_books() -> tuple[BookRecord, ...]:
    books: list[BookRecord] = []
    books_root = resolve_books_root()
    if not books_root.exists():
        return tuple()

    for path in sorted(books_root.iterdir(), key=lambda item: item.name.lower()):
        if not _is_book_dir(path):
            continue

        metadata = load_metadata(path)
        title = infer_book_title(path)
        source_pdf = metadata.get("source_pdf")
        books.append(
            BookRecord(
                slug=slugify_name(title),
                title=title,
                path=path.resolve(),
                chunk_count=_count_chunks(path / "chunks.jsonl"),
                source_pdf=str(source_pdf) if source_pdf else None,
            )
        )
    return tuple(books)


def get_book_by_slug(book_slug: str) -> BookRecord:
    for book in discover_books():
        if book.slug == book_slug:
            return book
    raise HTTPException(status_code=404, detail=f"Unknown book: {book_slug}")


@lru_cache(maxsize=8)
def get_engine(book_dir: str, embedding_model: str, reranker_model: str) -> BookRAGEngine:
    book_path = Path(book_dir)
    return BookRAGEngine(
        book_dir=book_path,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )


def create_app() -> FastAPI:
    load_local_env()
    app = FastAPI(title="Book RAG Web UI", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc) or "Internal Server Error"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/books")
    async def list_books() -> dict[str, object]:
        books = [
            {
                "slug": book.slug,
                "title": book.title,
                "chunk_count": book.chunk_count,
                "source_pdf": book.source_pdf,
            }
            for book in discover_books()
        ]
        return {"books": books}

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, object]:
        book = get_book_by_slug(request.book_slug)
        provider = request.provider.lower().strip()
        if provider not in {"openai", "anthropic"}:
            raise HTTPException(status_code=400, detail="provider must be 'openai' or 'anthropic'")

        profile_name = request.profile.lower().strip()
        if profile_name not in {"default", "fast", "deep"}:
            raise HTTPException(status_code=400, detail="profile must be default, fast, or deep")

        fast = profile_name == "fast"
        deep = profile_name == "deep"
        profile = resolve_query_profile(fast, deep)
        top_k = 8
        candidate_k = 40
        skip_rerank = False
        if profile is not None:
            top_k = profile.top_k
            candidate_k = profile.candidate_k
            skip_rerank = profile.skip_rerank

        engine = get_engine(
            book_dir=str(book.path),
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
        )

        def run_query() -> dict[str, object]:
            load_local_env(book.path)
            started = perf_counter()

            if request.retrieve_only:
                retrieval = engine.retrieve_only(
                    question=request.question,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    skip_rerank=skip_rerank,
                )
                duration_ms = round((perf_counter() - started) * 1000, 1)
                return {
                    "book": {"slug": book.slug, "title": retrieval.title},
                    "answer": "Retrieved sources only.",
                    "chunk_ids": [candidate["chunk_id"] for candidate in retrieval.candidates],
                    "timing_ms": duration_ms,
                    "mode": "retrieve-only",
                    "provider": None,
                    "model": None,
                }

            resolved_model = resolve_provider_model(provider, request.model)
            result = engine.answer_question(
                question=request.question,
                provider=provider,
                model=resolved_model,
                top_k=top_k,
                candidate_k=candidate_k,
                skip_rerank=skip_rerank,
            )
            duration_ms = round((perf_counter() - started) * 1000, 1)
            return {
                "book": {"slug": book.slug, "title": result.retrieval.title},
                "answer": result.answer.strip(),
                "chunk_ids": [candidate["chunk_id"] for candidate in result.retrieval.candidates],
                "timing_ms": duration_ms,
                "mode": "answer",
                "provider": provider,
                "model": resolved_model,
            }

        try:
            return await run_in_threadpool(run_query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
