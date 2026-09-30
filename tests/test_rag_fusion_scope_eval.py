"""RRF fusion, scope-gate routing, and the eval harness."""
from app.rag.corpus import load_corpus
from app.rag.evalharness import ndcg_at_k, recall_at_k, run_eval
from app.rag.fusion import rrf
from app.rag.hyde import HyDE, TemplateModel
from app.rag.index import TfidfIndex
from app.rag.rerank import BM25Reranker
from app.rag.scope import ScopeGate
from app.rag.search import SearchConfig, SearchPipeline


# ---- RRF ----
def test_rrf_rewards_agreement_across_rankings():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = rrf([a, b], k=60)
    # y is 1st+2nd, x is 1st+2nd → both top; y edges x (rank sum 1+2 vs 2+1 tie -> id order)
    ids = [d for d, _ in fused]
    assert ids[0] in ("x", "y") and ids[1] in ("x", "y")
    assert set(ids) == {"x", "y", "z", "w"}


def test_rrf_single_list_preserves_order():
    assert [d for d, _ in rrf([["a", "b", "c"]])] == ["a", "b", "c"]


# ---- scope gate ----
def _gate():
    return ScopeGate(threshold=0.12).fit({
        "nature": ["plants photosynthesis sunlight leaves",
                   "bees honey pollination flowers",
                   "glacier ice snow melt"],
        "earth": ["volcano magma eruption lava", "tides moon ocean sea level"],
    })


def test_scope_admits_in_domain():
    d = _gate().classify("how do bees pollinate flowers")
    assert d.in_scope and d.label == "nature"


def test_scope_refuses_out_of_domain():
    d = _gate().classify("how do I file my income tax return")
    assert not d.in_scope and d.label is None


# ---- eval harness ----
def test_metric_helpers():
    assert recall_at_k(["a", "b", "c"], "c", 3) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert round(ndcg_at_k(["a", "b"], "b", 5), 3) == round(1 / __import__("math").log2(3), 3)


def test_run_eval_reports_metrics():
    chunks = load_corpus()
    pipe = SearchPipeline(TfidfIndex(chunks), BM25Reranker(chunks), hyde=HyDE(TemplateModel()))
    evalset = [("how do bees make honey", "Honeybee#0"),
               ("how do plants use sunlight", "Photosynthesis#0")]
    rep = run_eval(pipe, evalset, k=5, config=SearchConfig(use_hyde=False))
    assert rep.n == 2 and rep.k == 5
    assert 0.0 <= rep.recall_at_k <= 1.0
    assert "recall@k" in rep.as_dict()
