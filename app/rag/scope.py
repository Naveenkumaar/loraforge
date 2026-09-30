"""Scope gate — nearest-centroid routing to refuse out-of-domain queries.

A cheap decision layer before the LLM: embed each in-domain class from a few
labelled examples as a centroid, then score a query by max cosine to any
centroid. Below a threshold → refuse (out of scope) without spending an LLM call.
Classic nearest-centroid classification; a cost gate, not a safety net.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _tf(text: str) -> dict[str, float]:
    toks = _tokens(text)
    if not toks:
        return {}
    c = Counter(toks)
    v = {t: c[t] / len(toks) for t in c}
    norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
    return {t: w / norm for t, w in v.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return sum(a[t] * b[t] for t in (set(a) & set(b)))


def _centroid(vectors: list[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for v in vectors:
        for t, w in v.items():
            acc[t] = acc.get(t, 0.0) + w
    if not acc:
        return {}
    n = len(vectors)
    acc = {t: w / n for t, w in acc.items()}
    norm = math.sqrt(sum(w * w for w in acc.values())) or 1.0
    return {t: w / norm for t, w in acc.items()}


@dataclass
class ScopeDecision:
    in_scope: bool
    label: str | None
    score: float


class ScopeGate:
    def __init__(self, threshold: float = 0.12) -> None:
        self.threshold = threshold
        self.centroids: dict[str, dict[str, float]] = {}

    def fit(self, labelled: dict[str, list[str]]) -> "ScopeGate":
        self.centroids = {label: _centroid([_tf(x) for x in examples])
                          for label, examples in labelled.items()}
        return self

    def classify(self, query: str) -> ScopeDecision:
        q = _tf(query)
        best_label, best = None, 0.0
        for label, c in self.centroids.items():
            s = _cosine(q, c)
            if s > best:
                best_label, best = label, s
        return ScopeDecision(in_scope=best >= self.threshold,
                             label=best_label if best >= self.threshold else None,
                             score=round(best, 4))
