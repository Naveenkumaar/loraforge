"""Provenance & versioning — deterministic chunk ids, supersede, never delete.

Auditable retrieval needs to answer 'where did this come from and is it current?'
Each chunk gets a **deterministic sha256 id** over (source, order, content), so
re-ingesting identical content yields the same id (idempotent). A changed source
**supersedes** the old chunk (marks it not-current) rather than hard-deleting —
the audit row survives. Named **corpus versions** snapshot the current set.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def chunk_id(source: str, order: int, content: str) -> str:
    h = hashlib.sha256(f"{source}\x1f{order}\x1f{content}".encode()).hexdigest()
    return h[:16]


@dataclass
class ProvChunk:
    id: str
    source: str
    order: int
    content: str
    is_current: bool = True
    version: int = 1


class ProvenanceStore:
    def __init__(self) -> None:
        self._chunks: dict[str, ProvChunk] = {}      # id -> chunk (all history)
        self._releases: dict[str, list[str]] = {}    # version name -> current ids

    def add(self, source: str, order: int, content: str) -> ProvChunk:
        cid = chunk_id(source, order, content)
        if cid in self._chunks:
            return self._chunks[cid]                 # idempotent: same content, same id
        self._chunks[cid] = ProvChunk(cid, source, order, content)
        return self._chunks[cid]

    def supersede(self, source: str, order: int, new_content: str) -> ProvChunk:
        """A source position changed: retire the old current chunk, add the new."""
        for c in self._chunks.values():
            if c.source == source and c.order == order and c.is_current:
                c.is_current = False                 # kept for audit, not deleted
                new = self.add(source, order, new_content)
                new.version = c.version + 1
                return new
        return self.add(source, order, new_content)

    def current(self) -> list[ProvChunk]:
        return [c for c in self._chunks.values() if c.is_current]

    def history(self, source: str, order: int) -> list[ProvChunk]:
        return sorted([c for c in self._chunks.values()
                       if c.source == source and c.order == order],
                      key=lambda c: c.version)

    # ---- named, immutable releases ------------------------------------
    def release(self, name: str) -> list[str]:
        ids = sorted(c.id for c in self.current())
        self._releases[name] = ids
        return ids

    def get_release(self, name: str) -> list[str]:
        if name not in self._releases:
            raise KeyError(f"no corpus version {name!r}")
        return list(self._releases[name])
