"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Instead of searching with the short question, ask a model to write a hypothetical
answer and search with *that* — it shares far more vocabulary with the real
passage. Blended with the query so a weak/offline model never hurts recall.

The model is pluggable. Offline, ``TemplateModel`` gives deterministic
hypothetical answers (a keyword-expansion stub) so the demo/tests run with no
network; swap in a real LLM for production quality.
"""
from __future__ import annotations

_HYDE_SYSTEM = (
    "Write a short factual passage that answers the question, as if quoted from "
    "an encyclopedia. State it plainly; do not express uncertainty.")


class TemplateModel:
    """Deterministic offline stand-in: expands a query into a fuller pseudo-answer.

    Not a real generator — it just repeats and lightly expands query terms so the
    pipeline is exercisable offline. Real lift comes from a real LLM.
    """
    backend = "template"

    def __init__(self, hints: dict[str, str] | None = None) -> None:
        self.hints = hints or {}

    def generate_hypothetical(self, query: str) -> str:
        for key, passage in self.hints.items():
            if key.lower() in query.lower():
                return passage
        return query  # no hint → fall back to the query itself


class HyDE:
    def __init__(self, model, blend: bool = True) -> None:
        self.model = model
        self.blend = blend

    def search_key(self, query: str) -> str:
        try:
            hyp = self.model.generate_hypothetical(query)
        except Exception:
            hyp = ""
        hyp = hyp or ""
        return f"{hyp} {query}".strip() if self.blend else (hyp or query)
