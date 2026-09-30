"""The search pipeline: (HyDE) -> TF-IDF recall -> BM25 rerank.

A request flows: optionally rewrite the query with HyDE, pull top-N candidates
from the TF-IDF index (recall), then reorder the top of that list with BM25
(precision). Each stage is optional and swappable, and every result carries the
chunk it came from so answers are traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.hyde import HyDE
from app.rag.index import TfidfIndex
from app.rag.rerank import BM25Reranker


@dataclass
class SearchConfig:
    recall_k: int = 10          # first-stage candidates
    rerank_k: int = 5           # reranked results returned
    use_hyde: bool = True
    use_rerank: bool = True


@dataclass
class Result:
    chunk_id: str
    title: str
    text: str
    score: float


@dataclass
class SearchOutput:
    query: str
    search_key: str
    results: list[Result] = field(default_factory=list)


class SearchPipeline:
    def __init__(self, index: TfidfIndex, reranker: BM25Reranker,
                 hyde: HyDE | None = None, config: SearchConfig | None = None) -> None:
        self.index = index
        self.reranker = reranker
        self.hyde = hyde
        self.cfg = config or SearchConfig()

    def search(self, query: str, config: SearchConfig | None = None) -> SearchOutput:
        cfg = config or self.cfg
        key = self.hyde.search_key(query) if (cfg.use_hyde and self.hyde) else query
        recall = self.index.search(key, top_k=cfg.recall_k)
        ids = [cid for cid, _ in recall]
        if cfg.use_rerank:
            ranked = self.reranker.rerank(query, ids, top_k=cfg.rerank_k)
        else:
            ranked = recall[:cfg.rerank_k]
        out = SearchOutput(query=query, search_key=key)
        for cid, score in ranked:
            c = self.index.by_id[cid]
            out.results.append(Result(cid, c.title, c.text, score))
        return out
