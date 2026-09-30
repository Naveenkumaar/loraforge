"""First-stage retrieval — TF-IDF cosine over chunks (pure Python).

Fast, high-recall candidate generation. IDF is computed once over the whole
chunk set; a query is vectorised the same way and scored by cosine. Swap for a
dense embedding index later — the ``search`` signature stays the same.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.corpus import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")

# minimal stopword list — keep content words dominant so rare function words
# ("how", "do", "make") don't hijack TF-IDF/BM25 scoring on short queries.
_STOP = frozenset(
    "a an the of to in on at for and or but is are was were be been being this "
    "that these those it its as by with from into out up down how do does did "
    "what when where why who which can could would should will may might must "
    "i you he she we they them my your our their me us him her not no yes if "
    "then than so such about over under again more most some any each".split())


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if keep_stop:
        return toks
    filtered = [t for t in toks if t not in _STOP]
    return filtered or toks   # never return empty (all-stopword query)


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        n = len(chunks) or 1
        df: Counter[str] = Counter()
        self._toks: dict[str, list[str]] = {}
        for c in chunks:
            t = tokens(c.text)
            self._toks[c.id] = t
            for term in set(t):
                df[term] += 1
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self._vec = {c.id: self._vector(self._toks[c.id]) for c in chunks}

    def _vector(self, toks: list[str]) -> dict[str, float]:
        if not toks:
            return {}
        tf = Counter(toks)
        default_idf = math.log(len(self.chunks) + 1) + 1.0
        v = {t: (tf[t] / len(toks)) * self._idf.get(t, default_idf) for t in tf}
        norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
        return {t: w / norm for t, w in v.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        return sum(a[t] * b[t] for t in (set(a) & set(b)))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = self._vector(tokens(query))
        scored = [(cid, round(self._cosine(q, v), 6)) for cid, v in self._vec.items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
