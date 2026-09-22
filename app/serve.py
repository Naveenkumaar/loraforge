"""Serving shim — one base model, many adapters, routed per request.

Mirrors how a self-hosted LLM is served with LoRA in production: the expensive
base is loaded once; each request names an adapter; the server hot-swaps it in
and runs the forward pass. A tiny per-model swap cache avoids reloading the same
adapter back-to-back.
"""
from __future__ import annotations

import numpy as np

from app.model import BaseModel
from app.registry import AdapterRegistry


class AdapterServer:
    def __init__(self, base: BaseModel, registry: AdapterRegistry) -> None:
        self.base = base
        self.registry = registry
        self.current: str | None = None

    def infer(self, x, adapter: str | None = None) -> np.ndarray:
        """Run inference, optionally under a named adapter (None = base only)."""
        if adapter != self.current:
            if adapter is None:
                self.base.layer.clear_adapter()
            else:
                self.registry.swap(self.base.layer, adapter)
            self.current = adapter
        return self.base.predict(np.atleast_2d(x))

    def available(self) -> list[str]:
        return self.registry.names()
