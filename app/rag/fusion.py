"""Reciprocal Rank Fusion (Cormack et al., 2009) — combine multiple rankings.

Two retrievers rank the same pool differently; RRF merges them by summing
1/(k+rank) across lists, rewarding items ranked high by *several* retrievers
without needing calibrated scores. Standard, score-free, and robust — the usual
way to fuse a lexical and a dense retriever.
"""
from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60, top_k: int | None = None):
    """Fuse ranked id-lists into one ranking. Returns [(id, rrf_score), ...]."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    fused = [(d, round(s, 6)) for d, s in fused]
    return fused[:top_k] if top_k else fused
