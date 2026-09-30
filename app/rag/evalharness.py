"""Retrieval eval harness — recall@k, MRR, nDCG over a labelled set.

Cheap, no-LLM metrics to gate retrieval quality in CI (the 'Tier-1' idea): run
the pipeline over (query -> relevant chunk id) pairs and report aggregate
recall@k, mean reciprocal rank, and nDCG@k. Deterministic and fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.rag.search import SearchConfig, SearchPipeline


def recall_at_k(result_ids: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in result_ids[:k] else 0.0


def reciprocal_rank(result_ids: list[str], gold: str) -> float:
    for i, cid in enumerate(result_ids, start=1):
        if cid == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(result_ids: list[str], gold: str, k: int) -> float:
    # single relevant doc → DCG = 1/log2(rank+1); ideal DCG = 1
    for i, cid in enumerate(result_ids[:k], start=1):
        if cid == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


@dataclass
class EvalReport:
    n: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int

    def as_dict(self) -> dict:
        return {"n": self.n, "k": self.k,
                "recall@k": round(self.recall_at_k, 4),
                "mrr": round(self.mrr, 4),
                "ndcg@k": round(self.ndcg_at_k, 4)}


def run_eval(pipeline: SearchPipeline, evalset: list[tuple[str, str]],
             k: int = 5, config: SearchConfig | None = None) -> EvalReport:
    if not evalset:
        return EvalReport(0, 0.0, 0.0, 0.0, k)
    rec = mrr = ndcg = 0.0
    for query, gold in evalset:
        ids = [r.chunk_id for r in pipeline.search(query, config=config).results]
        rec += recall_at_k(ids, gold, k)
        mrr += reciprocal_rank(ids, gold)
        ndcg += ndcg_at_k(ids, gold, k)
    n = len(evalset)
    return EvalReport(n, rec / n, mrr / n, ndcg / n, k)
