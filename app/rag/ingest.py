"""Multimodal ingestion — many source formats behind one unit shape.

Real corpora aren't just .txt: they're markdown, CSV/TSV tables, and (in a full
system) PDF/DOCX/images via OCR. The lesson that matters is architectural: every
extractor, whatever the format, emits **one shape** so everything downstream
(chunking, embedding, indexing) is format-agnostic — and it preserves document
order so a table/row's preceding text can be attached as context before embedding.

This ships dependency-free extractors for text, markdown, and delimited tables
(the formats that need no heavy libraries). Binary formats (PDF/DOCX/images+OCR)
are declared as the same interface with a clear extension point — plug a parser
without changing anything downstream.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ExtractedUnit:
    """One normalized piece of a source document, in authored order."""
    content_type: str                      # "text" | "table" | "image"
    content: str                           # text, or a text rendering of a table/image
    order: int                             # position within the source (document order)
    structured_data: list | None = None    # e.g. a table grid, kept structured (not flattened)
    metadata: dict = field(default_factory=dict)


# ---- individual extractors (each returns list[ExtractedUnit]) --------------

def extract_text(source: str, name: str = "doc") -> list[ExtractedUnit]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", source) if p.strip()]
    return [ExtractedUnit("text", p, i, metadata={"source": name})
            for i, p in enumerate(paras)]


def extract_markdown(source: str, name: str = "doc") -> list[ExtractedUnit]:
    """Split markdown into text blocks and fenced/pipe tables, in order."""
    units: list[ExtractedUnit] = []
    order = 0
    blocks = re.split(r"\n\s*\n", source)
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if _looks_like_md_table(b):
            grid = _parse_md_table(b)
            units.append(ExtractedUnit("table", _render_table(grid), order,
                                       structured_data=grid, metadata={"source": name}))
        else:
            text = re.sub(r"^#+\s*", "", b)          # strip heading marks
            units.append(ExtractedUnit("text", text, order, metadata={"source": name}))
        order += 1
    return units


def extract_delimited(source: str, name: str = "table", delimiter: str = ",") -> list[ExtractedUnit]:
    rows = [r for r in csv.reader(io.StringIO(source), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        return []
    return [ExtractedUnit("table", _render_table(rows), 0,
                          structured_data=rows, metadata={"source": name, "rows": len(rows)})]


# ---- table helpers ---------------------------------------------------------

def _looks_like_md_table(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) >= 2 and "|" in lines[0] and set(lines[1].strip()) <= set("|-: ")


def _parse_md_table(block: str) -> list[list[str]]:
    lines = [l for l in block.splitlines() if l.strip()]
    rows = []
    for i, line in enumerate(lines):
        if i == 1:            # separator row
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _render_table(grid: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in grid)


# ---- the one entry point + context attachment ------------------------------

_EXTRACTORS: dict[str, Callable[..., list[ExtractedUnit]]] = {
    "txt": extract_text, "text": extract_text,
    "md": extract_markdown, "markdown": extract_markdown,
    "csv": extract_delimited, "tsv": lambda s, name="t": extract_delimited(s, name, "\t"),
}


def ingest(source: str, fmt: str, name: str = "doc") -> list[ExtractedUnit]:
    fmt = fmt.lower().lstrip(".")
    if fmt not in _EXTRACTORS:
        raise ValueError(f"no extractor for format {fmt!r}; supported: {sorted(_EXTRACTORS)}")
    return _EXTRACTORS[fmt](source, name=name)


def attach_context(units: list[ExtractedUnit], window: int = 240) -> list[ExtractedUnit]:
    """Prepend the tail of the preceding TEXT unit to each table/image unit.

    A table alone often loses meaning ('what are these columns?'); attaching the
    text that introduced it makes the chunk self-contained before embedding.
    """
    out: list[ExtractedUnit] = []
    last_text = ""
    for u in sorted(units, key=lambda x: x.order):
        if u.content_type == "text":
            last_text = u.content
            out.append(u)
        else:
            prefix = last_text[-window:].strip()
            content = f"{prefix}\n{u.content}" if prefix else u.content
            out.append(ExtractedUnit(u.content_type, content, u.order,
                                     structured_data=u.structured_data,
                                     metadata={**u.metadata, "context_attached": bool(prefix)}))
    return out
