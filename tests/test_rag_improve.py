"""The RSI-style self-improvement loop finds a config at least as good as default."""
from app.rag.corpus import load_corpus
from app.rag.hyde import HyDE, TemplateModel
from app.rag.index import TfidfIndex
from app.rag.rerank import BM25Reranker
from app.rag.improve import evaluate, reciprocal_rank, self_improve
from app.rag.search import SearchConfig, SearchPipeline

EVAL = [
    ("how do bees make honey", "Honeybee#0"),
    ("what causes daily sea level changes", "Tides#0"),
    ("how do plants use sunlight", "Photosynthesis#0"),
    ("how are icebergs formed from ice", "Glacier#0"),
]


def _pipeline():
    chunks = load_corpus()
    return SearchPipeline(TfidfIndex(chunks), BM25Reranker(chunks),
                          hyde=HyDE(TemplateModel())), chunks


def test_reciprocal_rank_math():
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_self_improve_never_worse_than_start():
    pipe, _ = _pipeline()
    start = SearchConfig(recall_k=8, rerank_k=5, use_hyde=True, use_rerank=True)
    base = evaluate(pipe, EVAL, start)
    res = self_improve(pipe, EVAL, start=start, max_rounds=8)
    assert res.best_score >= base                 # hill-climb can't regress
    assert res.history[0]["round"] == 0
    assert res.history[-1]["score"] == round(res.best_score, 4)


def test_self_improve_reaches_strong_mrr():
    pipe, _ = _pipeline()
    res = self_improve(pipe, EVAL, start=SearchConfig(), max_rounds=12)
    # on this clean corpus a good policy should rank the gold chunk at/near the top
    assert res.best_score >= 0.75


def test_evaluate_empty_set_is_zero():
    pipe, _ = _pipeline()
    assert evaluate(pipe, [], SearchConfig()) == 0.0
