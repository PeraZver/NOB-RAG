from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_providers import answer_with_provider
from rag_retrieval import RetrievalResult, RetrievalSession


@dataclass
class AnswerResult:
    answer: str
    retrieval: RetrievalResult


class BookRAGEngine:
    def __init__(
        self,
        book_dir: Path,
        embedding_model: str,
        reranker_model: str,
        collection_name: str | None = None,
        db_dir: Path | None = None,
    ) -> None:
        self.retrieval = RetrievalSession(
            book_dir=book_dir,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            collection_name=collection_name,
            db_dir=db_dir,
        )

    @property
    def title(self) -> str:
        return self.retrieval.title

    def preload(self, include_reranker: bool = True) -> None:
        self.retrieval.preload(include_reranker=include_reranker)

    def retrieve_only(
        self,
        question: str,
        top_k: int,
        candidate_k: int,
        skip_rerank: bool,
    ) -> RetrievalResult:
        return self.retrieval.retrieve(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            skip_rerank=skip_rerank,
        )

    def answer_question(
        self,
        question: str,
        provider: str,
        model: str,
        top_k: int,
        candidate_k: int,
        skip_rerank: bool,
    ) -> AnswerResult:
        retrieval = self.retrieve_only(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            skip_rerank=skip_rerank,
        )
        answer = answer_with_provider(
            provider=provider,
            model=model,
            title=retrieval.title,
            question=question,
            context=retrieval.context,
        )
        return AnswerResult(answer=answer, retrieval=retrieval)
