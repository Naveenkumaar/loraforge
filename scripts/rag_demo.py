#!/usr/bin/env python3
"""Offline demo of the RAG stack: scope-gate → HyDE → recall → rerank → eval.

    python scripts/rag_demo.py

Runs on the built-in wiki-style corpus, no network. Shows the decision layer
refusing an out-of-domain query, HyDE improving a sparse query, RRF fusing two
retrievers, the eval harness, and the self-improvement loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.corpus import load_corpus  # noqa: E402
from app.rag.evalharness import run_eval  # noqa: E402
from app.rag.fusion import rrf  # noqa: E402
from app.rag.hyde import HyDE, TemplateModel  # noqa: E402
from app.rag.improve import self_improve  # noqa: E402
from app.rag.index import TfidfIndex  # noqa: E402
from app.rag.rerank import BM25Reranker  # noqa: E402
from app.rag.scope import ScopeGate  # noqa: E402
from app.rag.search import SearchConfig, SearchPipeline  # noqa: E402


def rule(t): print(f"\n\033[1m--- {t} ---\033[0m")


def main() -> int:
    chunks = load_corpus()
    idx = TfidfIndex(chunks)
    rr = BM25Reranker(chunks)
    hint = {"energy from sunlight":
            "Plants convert light energy into chemical energy using chlorophyll, "
            "turning carbon dioxide and water into glucose and oxygen."}
    pipe = SearchPipeline(idx, rr, hyde=HyDE(TemplateModel(hint)))

    rule("Corpus")
    print(f"{len(chunks)} chunks from {len(set(c.title for c in chunks))} articles")

    rule("Scope gate — refuse out-of-domain before any LLM call")
    gate = ScopeGate(threshold=0.12).fit({
        "nature": ["plants photosynthesis sunlight", "bees honey flowers", "glacier ice snow"],
        "earth": ["volcano magma lava", "tides moon ocean"]})
    for q in ["how do bees make honey", "how do I file my taxes"]:
        d = gate.classify(q)
        print(f"  {'ADMIT' if d.in_scope else 'REFUSE'}  ({d.label or '-'}, {d.score})  \"{q}\"")

    rule("HyDE — a sparse query finds the right passage")
    out = pipe.search("energy from sunlight in plants", SearchConfig(use_hyde=True))
    print(f"  top: {out.results[0].title}  (key rewritten: {out.search_key[:60]}...)")

    rule("RRF fusion — combine TF-IDF recall with BM25")
    q = "what makes the sea rise and fall each day"
    tfidf = [cid for cid, _ in idx.search(q, top_k=6)]
    bm25 = [cid for cid, _ in rr.rerank(q, tfidf)]
    fused = rrf([tfidf, bm25], k=60, top_k=3)
    print(f"  fused top: {[idx.by_id[d].title for d, _ in fused]}")

    rule("Eval harness (Tier-1, no LLM)")
    evalset = [("how do bees make honey", "Honeybee#0"),
               ("what causes daily sea level changes", "Tides#0"),
               ("how do plants use sunlight", "Photosynthesis#0"),
               ("how are icebergs formed", "Glacier#0")]
    print(f"  {run_eval(pipe, evalset, k=5, config=SearchConfig(use_hyde=False)).as_dict()}")

    rule("Self-improvement (RSI-style) — hill-climb the retrieval policy")
    res = self_improve(pipe, evalset, start=SearchConfig(), max_rounds=10)
    print(f"  MRR {res.history[0]['score']} -> {round(res.best_score,4)} "
          f"in {len(res.history)-1} rounds; best={res.best_config.__dict__}")

    rule("Done")
    print("All offline. This is the RAG half of loraforge; the LoRA half is scripts/demo.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
