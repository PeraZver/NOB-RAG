from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag.rag_utils import (
    chroma_collection_name,
    format_page_citation,
    infer_book_title,
    infer_chunk_features,
    iter_jsonl,
    normalize_search_text,
    parse_source_pages,
    preview_text,
    tokenize_search_text,
)


@dataclass
class RetrievalResult:
    title: str
    context: str
    source_summaries: list[str]
    candidates: list[dict]


def build_context(candidates: list[dict]) -> tuple[str, list[str]]:
    blocks: list[str] = []
    source_summaries: list[str] = []

    for rank, candidate in enumerate(candidates, start=1):
        document = candidate["text"]
        metadata = candidate["metadata"]
        distance = candidate["distance"]
        source_pages = parse_source_pages(metadata.get("source_pages", ""))
        citation = format_page_citation(source_pages)
        chunk_id = metadata["chunk_id"]
        blocks.append(
            "\n".join(
                [
                    f"Source {rank}",
                    f"chunk_id: {chunk_id}",
                    f"pages: {citation}",
                    f"distance: {distance:.4f}",
                    document,
                ]
            )
        )
        source_summaries.append(
            f"[{rank}] chunk_id={chunk_id} {citation} :: {preview_text(document)}"
        )

    return "\n\n".join(blocks), source_summaries


@lru_cache(maxsize=4)
def load_embedder(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _load_reranker_components(
    model_name: str,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()
    return tokenizer, model


def lexical_score(question: str, text: str) -> float:
    question_tokens = [token for token in tokenize_search_text(question) if len(token) >= 3]
    if not question_tokens:
        return 0.0

    expanded_terms = {
        "osnovana": ["osnivanje", "formiranje", "formirana", "postrojili", "postrojavanje"],
        "osnovan": ["osnivanje", "formiranje", "formiran"],
        "osnivanje": ["osnovana", "formiranje", "formirana", "postrojavanje"],
        "formirana": ["formiranje", "osnovana", "osnivanje", "postrojavanje"],
        "formiran": ["formiranje", "osnovan", "osnivanje"],
        "formiranje": ["formirana", "osnivanje", "osnovana", "postrojavanje"],
        "brigada": ["brigade", "biokovske", "biokovska"],
        "where": ["gdje", "kozici", "biokovu"],
        "when": ["kada", "oktobra", "listopada"],
        "formed": ["formirana", "formiranje", "osnovana", "osnivanje"],
        "founded": ["osnovana", "osnivanje", "formirana", "formiranje"],
        "established": ["osnovana", "osnivanje", "formirana", "formiranje"],
    }

    search_terms: list[str] = list(question_tokens)
    for token in question_tokens:
        search_terms.extend(expanded_terms.get(token, []))

    text_tokens = set(tokenize_search_text(text))
    token_overlap = sum(1 for token in search_terms if token in text_tokens)

    normalized_question = normalize_search_text(question)
    normalized_text = normalize_search_text(text)
    bigrams = [
        f"{search_terms[index]} {search_terms[index + 1]}"
        for index in range(len(search_terms) - 1)
    ]
    bigram_hits = sum(1 for bigram in bigrams if bigram in normalized_text)
    phrase_bonus = 2.0 if len(normalized_question) >= 12 and normalized_question in normalized_text else 0.0
    return float(token_overlap + (1.5 * bigram_hits) + phrase_bonus)


def classify_query_intent(question: str) -> dict[str, bool]:
    normalized = normalize_search_text(question)
    tokens = set(tokenize_search_text(question))

    formation_terms = {
        "osnovana",
        "osnovan",
        "osnivanje",
        "formirana",
        "formiran",
        "formiranje",
        "postrojavanje",
        "postrojili",
        "founded",
        "formed",
        "established",
    }
    chronology_terms = {
        "kada",
        "when",
        "timeline",
        "chronology",
        "tokom",
        "nakon",
        "prije",
        "1941",
        "1942",
        "1943",
        "1944",
        "1945",
    }
    commander_terms = {
        "komandant",
        "komandanti",
        "komesar",
        "komesari",
        "stab",
        "staba",
        "command",
        "commander",
        "staff",
    }
    structure_terms = {
        "brigada",
        "bataljon",
        "bataljona",
        "ceta",
        "cete",
        "odred",
        "divizija",
        "divizije",
        "korpus",
        "korpusa",
        "sastav",
        "structure",
        "unit",
        "organization",
    }
    campaign_terms = {
        "napad",
        "napadi",
        "borba",
        "borbe",
        "bitka",
        "bitke",
        "operacija",
        "operacije",
        "dejstva",
        "pokret",
        "kretanje",
        "movement",
        "movements",
        "location",
        "locations",
        "battle",
        "battles",
        "campaign",
        "campaigns",
        "assault",
        "attack",
        "attacks",
        "incursion",
        "evacuated",
        "evacuation",
        "desant",
        "iskrcavanje",
        "oslobodenje",
        "oslobadanje",
        "advance",
        "advancee",
        "vis",
        "brac",
        "korcula",
        "peljesac",
        "split",
        "sibenik",
        "knin",
        "mostar",
        "bihac",
        "gospic",
        "krk",
        "trst",
        "istria",
        "istra",
        "slovenia",
        "sloveniji",
    }

    return {
        "formation": bool(tokens & formation_terms),
        "chronology": bool(tokens & chronology_terms),
        "command": bool(tokens & commander_terms),
        "structure": bool(tokens & structure_terms),
        "campaign": bool(tokens & campaign_terms),
        "brach_topic": "brac" in normalized or "braÄu" in question.lower() or "braÄ" in question.lower(),
    }


def metadata_boost(question: str, metadata: dict[str, object], text: str) -> float:
    intent = classify_query_intent(question)
    features = infer_chunk_features(
        chunk_id=int(metadata["chunk_id"]),
        source_pages=parse_source_pages(metadata.get("source_pages", "")),
        text=text,
    )

    boost = 0.0

    if intent["formation"]:
        if features["has_section_heading"]:
            boost += 0.35
        if features["is_early_section"]:
            boost += 0.45
        if features["has_structure_terms"]:
            boost += 0.25
        if "kozic" in normalize_search_text(text) or "biokov" in normalize_search_text(text):
            boost += 0.5

    if intent["chronology"]:
        if features["has_chronology_terms"]:
            boost += 0.35
        if features["is_early_section"]:
            boost += 0.2
        if features["is_contents"]:
            boost += 0.25

    if intent["command"]:
        if features["has_structure_terms"]:
            boost += 0.35
        if features["is_appendix_like"]:
            boost += 0.2
        if "komand" in normalize_search_text(text) or "komesar" in normalize_search_text(text):
            boost += 0.35

    if intent["structure"]:
        if features["has_structure_terms"]:
            boost += 0.35
        if features["is_contents"]:
            boost += 0.25
        if features["has_section_heading"]:
            boost += 0.15

    if intent["campaign"]:
        if features["has_operational_terms"]:
            boost += 0.55
        if features["has_section_heading"]:
            boost += 0.15
        if not features["is_appendix_like"]:
            boost += 0.15
        if features["is_appendix_like"]:
            boost -= 0.35
        if features["looks_list_like"]:
            boost -= 0.15
        normalized_text = normalize_search_text(text)
        location_terms = [
            "vis",
            "brac",
            "korcula",
            "peljesac",
            "split",
            "sibenik",
            "knin",
            "mostar",
            "bihac",
            "gospic",
            "krk",
            "trst",
            "celovec",
            "istra",
            "sloven",
            "biokov",
        ]
        if any(term in normalized_text for term in location_terms):
            boost += 0.2

    if intent["brach_topic"]:
        normalized_text = normalize_search_text(text)
        if "brac" in normalized_text:
            boost += 0.5
        if "desant" in normalized_text or "napad" in normalized_text:
            boost += 0.3

    if features["is_contents"]:
        boost += 0.05

    return boost


def get_semantic_candidates(collection, query_embedding: list[float], limit: int) -> dict[int, dict]:
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    candidates: dict[int, dict] = {}

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        chunk_id = int(metadata["chunk_id"])
        candidates[chunk_id] = {
            "chunk_id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": float(distance),
            "semantic_rank": rank,
        }
    return candidates


def get_lexical_candidates(chunks_path: Path, question: str, limit: int) -> dict[int, dict]:
    scored: list[dict] = []
    for record in iter_jsonl(chunks_path):
        score = lexical_score(question, record.get("text", ""))
        if score <= 0:
            continue
        scored.append(
            {
                "chunk_id": int(record["chunk_id"]),
                "text": record["text"],
                "metadata": {
                    "chunk_id": int(record["chunk_id"]),
                    "word_count": int(record["word_count"]),
                    "source_pages": ",".join(str(page) for page in record.get("source_pages", [])),
                },
                "lexical_score": score,
            }
        )

    scored.sort(key=lambda item: item["lexical_score"], reverse=True)
    return {item["chunk_id"]: item for item in scored[:limit]}


def merge_candidates(
    question: str,
    semantic_candidates: dict[int, dict],
    lexical_candidates: dict[int, dict],
    candidate_k: int,
) -> list[dict]:
    merged: dict[int, dict] = {}

    max_semantic_rank = max(
        (candidate["semantic_rank"] for candidate in semantic_candidates.values()),
        default=1,
    )
    max_lexical_score = max(
        (candidate["lexical_score"] for candidate in lexical_candidates.values()),
        default=1.0,
    )

    for chunk_id, candidate in semantic_candidates.items():
        semantic_score = 1.0 - ((candidate["semantic_rank"] - 1) / max(1, max_semantic_rank - 1))
        merged[chunk_id] = {
            **candidate,
            "semantic_score": semantic_score,
            "lexical_score": 0.0,
            "metadata_boost": metadata_boost(question, candidate["metadata"], candidate["text"]),
        }

    for chunk_id, candidate in lexical_candidates.items():
        lexical_component = candidate["lexical_score"] / max_lexical_score
        if chunk_id in merged:
            merged[chunk_id]["lexical_score"] = lexical_component
        else:
            merged[chunk_id] = {
                **candidate,
                "distance": 1.0,
                "semantic_rank": max_semantic_rank + 1,
                "semantic_score": 0.0,
                "lexical_score": lexical_component,
                "metadata_boost": metadata_boost(question, candidate["metadata"], candidate["text"]),
            }

    merged_items = list(merged.values())
    for item in merged_items:
        item["combined_score"] = (
            (0.35 * item["semantic_score"])
            + (0.45 * item["lexical_score"])
            + (0.20 * item["metadata_boost"])
        )

    merged_items.sort(key=lambda item: item["combined_score"], reverse=True)
    return merged_items[:candidate_k]


def rerank_candidates(
    question: str,
    candidates: list[dict],
    reranker_model: str,
    top_k: int,
) -> list[dict]:
    if not candidates:
        return []

    tokenizer, model = _load_reranker_components(reranker_model)
    pairs = [[question, candidate["text"]] for candidate in candidates]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float().tolist()

    reranked = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["rerank_score"] = float(score)
        reranked.append(enriched)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


class RetrievalSession:
    def __init__(
        self,
        book_dir: Path,
        embedding_model: str,
        reranker_model: str,
        collection_name: str | None = None,
        db_dir: Path | None = None,
    ) -> None:
        self.book_dir = book_dir.resolve()
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.title = infer_book_title(self.book_dir)
        self.collection_name = collection_name or chroma_collection_name(self.title)
        self.db_dir = (db_dir or (self.book_dir / "chroma_db")).resolve()
        self.chunks_path = self.book_dir / "chunks.jsonl"

        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_collection(self.collection_name)
        self.embedder = load_embedder(self.embedding_model)

    def preload(self, include_reranker: bool = True) -> None:
        if include_reranker:
            _load_reranker_components(self.reranker_model)

    def retrieve(self, question: str, top_k: int, candidate_k: int, skip_rerank: bool) -> RetrievalResult:
        query_embedding = self.embedder.encode(question).tolist()

        semantic_candidates = get_semantic_candidates(
            collection=self.collection,
            query_embedding=query_embedding,
            limit=max(candidate_k * 2, 24),
        )
        lexical_candidates = get_lexical_candidates(
            chunks_path=self.chunks_path,
            question=question,
            limit=max(candidate_k * 2, 24),
        )
        merged_candidates = merge_candidates(
            question=question,
            semantic_candidates=semantic_candidates,
            lexical_candidates=lexical_candidates,
            candidate_k=candidate_k,
        )
        final_candidates = (
            merged_candidates[:top_k]
            if skip_rerank
            else rerank_candidates(
                question=question,
                candidates=merged_candidates,
                reranker_model=self.reranker_model,
                top_k=top_k,
            )
        )
        context, source_summaries = build_context(final_candidates)
        return RetrievalResult(
            title=self.title,
            context=context,
            source_summaries=source_summaries,
            candidates=final_candidates,
        )
