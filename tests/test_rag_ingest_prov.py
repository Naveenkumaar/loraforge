"""Multimodal ingestion (one unit shape, context attachment) + provenance."""
import pytest

from app.rag.ingest import (
    ExtractedUnit, attach_context, extract_delimited, ingest,
)
from app.rag.provenance import ProvenanceStore, chunk_id


# ---- ingestion ----
def test_markdown_splits_text_and_tables_in_order():
    md = ("# Fruit prices\n\nHere is the current list.\n\n"
          "| Fruit | Price |\n|---|---|\n| Apple | 2 |\n| Pear | 3 |\n\nThanks.")
    units = ingest(md, "md", name="prices")
    kinds = [(u.order, u.content_type) for u in units]
    assert kinds == [(0, "text"), (1, "text"), (2, "table"), (3, "text")]
    table = [u for u in units if u.content_type == "table"][0]
    assert table.structured_data[0] == ["Fruit", "Price"]      # kept structured
    assert table.structured_data[2] == ["Pear", "3"]


def test_all_formats_emit_the_same_shape():
    for src, fmt in [("hello world", "txt"), ("# h\n\ntext", "md"), ("a,b\n1,2", "csv")]:
        for u in ingest(src, fmt):
            assert isinstance(u, ExtractedUnit)
            assert u.content_type in ("text", "table", "image")


def test_delimited_keeps_grid():
    units = extract_delimited("name,qty\napple,5\npear,9", name="t")
    assert units[0].structured_data == [["name", "qty"], ["apple", "5"], ["pear", "9"]]


def test_context_attached_to_table():
    units = [ExtractedUnit("text", "The table below lists Q3 revenue by region.", 0),
             ExtractedUnit("table", "Region | Rev\nEMEA | 10", 1, structured_data=[["Region", "Rev"]])]
    out = attach_context(units)
    tbl = [u for u in out if u.content_type == "table"][0]
    assert "Q3 revenue by region" in tbl.content          # preceding text prepended
    assert tbl.metadata["context_attached"] is True


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        ingest("x", "pdf")                                # declared extension point, no parser shipped


# ---- provenance ----
def test_chunk_id_is_deterministic_and_idempotent():
    a = chunk_id("doc", 0, "hello")
    b = chunk_id("doc", 0, "hello")
    c = chunk_id("doc", 0, "world")
    assert a == b and a != c
    st = ProvenanceStore()
    x = st.add("doc", 0, "hello")
    y = st.add("doc", 0, "hello")
    assert x.id == y.id and len(st.current()) == 1        # no duplicate


def test_supersede_keeps_audit_and_bumps_version():
    st = ProvenanceStore()
    st.add("doc", 0, "old value")
    new = st.supersede("doc", 0, "new value")
    assert new.version == 2 and new.is_current
    hist = st.history("doc", 0)
    assert len(hist) == 2                                  # old row survives
    assert [h.is_current for h in hist] == [False, True]
    assert len(st.current()) == 1                          # only the new one is current


def test_named_release_snapshots_current():
    st = ProvenanceStore()
    st.add("doc", 0, "a"); st.add("doc", 1, "b")
    r1 = st.release("v1")
    st.supersede("doc", 0, "a2")
    r2 = st.release("v2")
    assert st.get_release("v1") == r1                      # v1 frozen
    assert r1 != r2                                        # v2 reflects the change
    with pytest.raises(KeyError):
        st.get_release("nope")
