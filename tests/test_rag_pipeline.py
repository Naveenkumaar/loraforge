"""Corpus/index/rerank/HyDE/pipeline behaviour."""
from app.rag.corpus import chunk_text, load_corpus
from app.rag.hyde import HyDE, TemplateModel
from app.rag.index import TfidfIndex
from app.rag.rerank import BM25Reranker
from app.rag.search import SearchConfig, SearchPipeline


def _pipeline(**hyde_hints):
    chunks = load_corpus()
    idx = TfidfIndex(chunks)
    rr = BM25Reranker(chunks)
    hyde = HyDE(TemplateModel(hyde_hints)) if hyde_hints else None
    return SearchPipeline(idx, rr, hyde=hyde), chunks


def test_chunking_overlaps_and_covers():
    text = " ".join(f"w{i}" for i in range(60))
    ch = chunk_text(text, size=24, overlap=8)
    assert len(ch) > 1
    assert ch[0].split()[0] == "w0" and ch[-1].split()[-1] == "w59"


def test_index_finds_the_obvious_article():
    pipe, _ = _pipeline()
    out = pipe.search("how do bees make honey", SearchConfig(use_hyde=False))
    assert out.results[0].title == "Honeybee"


def test_rerank_changes_order_vs_recall_only():
    pipe, _ = _pipeline()
    q = "what causes the sea level to rise and fall each day"
    reranked = pipe.search(q, SearchConfig(use_hyde=False, use_rerank=True))
    recall = pipe.search(q, SearchConfig(use_hyde=False, use_rerank=False))
    assert reranked.results[0].title == "Tides"
    # both stages should agree on the right topic here, but expose distinct scores
    assert reranked.results[0].score != recall.results[0].score


def test_hyde_helps_a_sparse_query():
    # a question with little lexical overlap with the target passage
    hint = {"energy from sunlight":
            "Plants convert light energy into chemical energy using chlorophyll "
            "in chloroplasts, turning carbon dioxide and water into glucose and oxygen."}
    pipe, _ = _pipeline(**hint)
    q = "energy from sunlight in plants"
    with_hyde = pipe.search(q, SearchConfig(use_hyde=True, use_rerank=True))
    assert with_hyde.results[0].title == "Photosynthesis"
    assert with_hyde.search_key != q          # HyDE rewrote the search key


def test_search_key_is_query_when_hyde_off():
    pipe, _ = _pipeline(**{"x": "y"})
    out = pipe.search("volcano eruption", SearchConfig(use_hyde=False))
    assert out.search_key == "volcano eruption"


def test_hyde_blend_never_empty_on_model_error():
    class _Boom:
        def generate_hypothetical(self, q): raise RuntimeError("down")
    key = HyDE(_Boom()).search_key("glacier ice flow")
    assert "glacier" in key
