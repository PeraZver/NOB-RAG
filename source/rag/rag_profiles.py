from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryProfile:
    name: str
    top_k: int
    candidate_k: int
    skip_rerank: bool


FAST_PROFILE = QueryProfile(
    name="fast",
    top_k=4,
    candidate_k=12,
    skip_rerank=True,
)

DEEP_PROFILE = QueryProfile(
    name="deep",
    top_k=10,
    candidate_k=60,
    skip_rerank=False,
)


def resolve_query_profile(use_fast: bool, use_deep: bool) -> QueryProfile | None:
    if use_fast and use_deep:
        raise ValueError("Choose only one query profile: --fast or --deep.")
    if use_fast:
        return FAST_PROFILE
    if use_deep:
        return DEEP_PROFILE
    return None
