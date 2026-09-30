"""Recursive self-improvement of the retrieval policy (RSI-inspired).

Given a small labelled eval set (query -> the chunk that should rank #1), the
system *evaluates its own configurations* and hill-climbs toward a better one:
it scores the current config on the metric, proposes neighbouring configs
(toggle HyDE/rerank, nudge recall_k / rerank_k / BM25 k1,b), keeps any that
improve the score, and repeats until no neighbour helps. Deterministic — no
randomness — so a run is reproducible.

This is a small, transparent take on the general idea of a system that measures
and improves its own behaviour; it optimises a retrieval policy, not model
weights. Swap the metric or the search space without touching the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.rag.rerank import BM25Reranker
from app.rag.search import SearchConfig, SearchPipeline


def reciprocal_rank(result_ids: list[str], gold_id: str) -> float:
    for i, cid in enumerate(result_ids, start=1):
        if cid == gold_id:
            return 1.0 / i
    return 0.0


def evaluate(pipeline: SearchPipeline, evalset: list[tuple[str, str]],
             config: SearchConfig) -> float:
    """Mean Reciprocal Rank of the pipeline under ``config`` on the eval set."""
    if not evalset:
        return 0.0
    total = 0.0
    for query, gold in evalset:
        out = pipeline.search(query, config=config)
        total += reciprocal_rank([r.chunk_id for r in out.results], gold)
    return total / len(evalset)


@dataclass
class ImproveResult:
    best_config: SearchConfig
    best_score: float
    history: list[dict] = field(default_factory=list)


def _neighbours(cfg: SearchConfig) -> list[SearchConfig]:
    n: list[SearchConfig] = []
    n.append(replace(cfg, use_hyde=not cfg.use_hyde))
    n.append(replace(cfg, use_rerank=not cfg.use_rerank))
    for dk in (-2, 2):
        if 2 <= cfg.recall_k + dk <= 50:
            n.append(replace(cfg, recall_k=cfg.recall_k + dk))
    for dk in (-1, 1):
        if 1 <= cfg.rerank_k + dk <= cfg.recall_k:
            n.append(replace(cfg, rerank_k=cfg.rerank_k + dk))
    return n


def self_improve(pipeline: SearchPipeline, evalset: list[tuple[str, str]],
                 start: SearchConfig | None = None, max_rounds: int = 12,
                 tune_bm25: bool = True) -> ImproveResult:
    """Hill-climb the search config to maximise MRR on the eval set."""
    cfg = start or SearchConfig()
    best = evaluate(pipeline, evalset, cfg)
    history = [{"round": 0, "score": round(best, 4), "config": cfg.__dict__.copy()}]

    for r in range(1, max_rounds + 1):
        improved = False
        for cand in _neighbours(cfg):
            s = evaluate(pipeline, evalset, cand)
            if s > best + 1e-9:
                best, cfg, improved = s, cand, True
        # optionally nudge BM25 params on the shared reranker (part of the policy)
        if tune_bm25:
            for (dk1, db) in [(-0.3, 0), (0.3, 0), (0, -0.1), (0, 0.1)]:
                rr = pipeline.reranker
                old_k1, old_b = rr.k1, rr.b
                rr.k1 = max(0.5, round(old_k1 + dk1, 3))
                rr.b = min(1.0, max(0.0, round(old_b + db, 3)))
                s = evaluate(pipeline, evalset, cfg)
                if s > best + 1e-9:
                    best, improved = s, True
                else:
                    rr.k1, rr.b = old_k1, old_b   # revert
        history.append({"round": r, "score": round(best, 4), "config": cfg.__dict__.copy(),
                        "bm25": {"k1": pipeline.reranker.k1, "b": pipeline.reranker.b}})
        if not improved:
            break
    return ImproveResult(best_config=cfg, best_score=best, history=history)
