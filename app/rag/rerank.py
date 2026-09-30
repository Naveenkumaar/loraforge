"""Second-stage reranker — BM25 (Okapi), a different scoring function.

The first stage (TF-IDF cosine) is tuned for recall; BM25 reorders those
candidates for precision with term-saturation (k1) and length-normalization (b),
which cosine lacks. Using a *different* function for reranking is the point — it
corrects first-stage ranking errors rather than repeating them.
"""
from __future__ import annotations

import math
from collections import Counter

from app.rag.corpus import Chunk
from app.rag.index import tokens


class BM25Reranker:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.by_id = {c.id: c for c in chunks}
        self._toks = {c.id: tokens(c.text) for c in chunks}
        lengths = [len(t) for t in self._toks.values()] or [1]
        self.avgdl = sum(lengths) / len(lengths)
        n = len(chunks) or 1
        df: Counter[str] = Counter()
        for t in self._toks.values():
            for term in set(t):
                df[term] += 1
        # BM25 idf with +0.5 smoothing
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, chunk_id: str) -> float:
        doc = self._toks.get(chunk_id, [])
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in tokens(query):
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            num = tf[term] * (self.k1 + 1)
            den = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * num / den
        return s

    def rerank(self, query: str, candidate_ids: list[str], top_k: int | None = None):
        scored = [(cid, round(self.score(query, cid), 6)) for cid in candidate_ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k] if top_k else scored
