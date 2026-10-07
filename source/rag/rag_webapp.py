from __future__ import annotations

import json
import os
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
from rag.rag_utils import (
    IndexedBook,
    discover_books,
    resolve_books_root,
    shared_collection_name,
    shared_db_dir,
)


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "webapp_static"

CAMPAIGN_FILES = {
    "default": "brigade_campaign.json",
    "grouped": "brigade_campaign_grouped.json",
    "verified": "brigade_campaign_verified.json",
}


class ChatRequest(BaseModel):
    book_slug: str
    question: str = Field(min_length=1)
    provider: str = Field(default=os.getenv("RAG_PROVIDER", "anthropic"))
    profile: str = Field(default="fast")
    retrieve_only: bool = False
    model: str | None = None


def display_parts(book: IndexedBook) -> dict[str, str]:
    # Prefer curated metadata; fall back to splitting "Author - Title".
    author = str(book.metadata.get("author") or "").strip()
    title = str(book.metadata.get("display_title") or "").strip()
    if not title:
        head, sep, tail = book.title.partition(" - ")
        author, title = (head.strip(), tail.strip()) if sep else ("", book.title)
    return {"author": author, "display_title": title}


def available_books() -> tuple[IndexedBook, ...]:
    return discover_books(resolve_books_root(SOURCE_DIR))


def shared_index_available() -> bool:
    books_root = resolve_books_root(SOURCE_DIR)
    db_dir = shared_db_dir(books_root)
    if not db_dir.exists():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        client.get_collection(shared_collection_name())
        return True
    except Exception:
        return False


def get_book_by_slug(book_slug: str) -> IndexedBook:
    for book in available_books():
        if book.slug == book_slug:
            return book
    raise HTTPException(status_code=404, detail=f"Unknown book: {book_slug}")


@lru_cache(maxsize=8)
def get_engine(
    book_dir: str | None,
    embedding_model: str,
    reranker_model: str,
    db_dir: str | None = None,
    collection_name: str | None = None,
    books_root: str | None = None,
    query_book_slug: str | None = None,
    title: str | None = None,
) -> BookRAGEngine:
    book_path = Path(book_dir) if book_dir else None
    return BookRAGEngine(
        book_dir=book_path,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        db_dir=Path(db_dir) if db_dir else None,
        collection_name=collection_name,
        books_root=Path(books_root) if books_root else None,
        query_book_slug=query_book_slug,
        title=title,
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

    @app.middleware("http")
    async def no_cache_ui(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

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
                **display_parts(book),
                "chunk_count": book.chunk_count,
                "source_pdf": book.source_pdf,
                "has_book_index": (book.path / "chroma_db").is_dir(),
            }
            for book in available_books()
        ]
        return {
            "books": books,
            "shared_index_available": shared_index_available(),
            "books_root": str(resolve_books_root(SOURCE_DIR)),
        }

    @app.get("/api/campaign/{book_slug}")
    async def campaign(book_slug: str, variant: str = "default") -> dict[str, object]:
        if variant not in CAMPAIGN_FILES:
            raise HTTPException(status_code=400, detail="variant must be default, grouped, or verified")
        book = get_book_by_slug(book_slug)
        path = book.path / CAMPAIGN_FILES[variant]
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No {variant} campaign file for this book.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"Could not read campaign file: {exc}") from exc

        points = []
        for movement in data.get("movements") or []:
            coords = movement.get("coordinates") or {}
            lat, lng = coords.get("lat"), coords.get("lng")
            if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
                continue
            points.append(
                {
                    "date": movement.get("date") or "",
                    "place": movement.get("place") or "",
                    "operation": movement.get("operation") or "",
                    "lat": lat,
                    "lng": lng,
                }
            )
        points.sort(key=lambda point: point["date"] or "9999")
        if not points:
            raise HTTPException(status_code=404, detail=f"The {variant} campaign file has no points with coordinates.")
        return {"variant": variant, "brigade_name": data.get("brigade_name") or "", "points": points}

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, object]:
        using_shared = shared_index_available()
        book = None if request.book_slug == "__all__" else get_book_by_slug(request.book_slug)
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

        if request.book_slug == "__all__" and not using_shared:
            raise HTTPException(
                status_code=400,
                detail="The shared index is not available yet. Build it with python -m rag.build_shared_chroma_index --reset.",
            )
        if not using_shared and book is not None and not (book.path / "chroma_db").is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The selected book does not have a per-book Chroma index yet: {book.title}. "
                    "Run python -m rag.build_chroma_index \"..\\<Book Folder>\" --reset or build the shared library index."
                ),
            )

        books_root = resolve_books_root(SOURCE_DIR)
        if using_shared:
            selected_title = "All indexed books" if book is None else book.title
            engine = get_engine(
                book_dir=None,
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                db_dir=str(shared_db_dir(books_root)),
                collection_name=shared_collection_name(),
                books_root=str(books_root),
                query_book_slug=None if book is None else book.slug,
                title=selected_title,
            )
        else:
            assert book is not None
            engine = get_engine(
                book_dir=str(book.path),
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                title=book.title,
            )

        def run_query() -> dict[str, object]:
            load_local_env(book.path if book is not None else None)
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
                    "book": {"slug": request.book_slug, "title": retrieval.title},
                    "answer": "Retrieved sources only.",
                    "chunk_refs": [
                        (
                            f"{candidate['metadata'].get('book_slug')}:{candidate['chunk_id']}"
                            if request.book_slug == "__all__"
                            else str(candidate["chunk_id"])
                        )
                        for candidate in retrieval.candidates
                    ],
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
                "book": {"slug": request.book_slug, "title": result.retrieval.title},
                "answer": result.answer.strip(),
                "chunk_refs": [
                    (
                        f"{candidate['metadata'].get('book_slug')}:{candidate['chunk_id']}"
                        if request.book_slug == "__all__"
                        else str(candidate["chunk_id"])
                    )
                    for candidate in result.retrieval.candidates
                ],
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
