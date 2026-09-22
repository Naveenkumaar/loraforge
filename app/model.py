"""A tiny 'base model' whose projection is LoRA-adaptable.

Stands in for a large self-hosted LLM: a single frozen linear projection that all
adapters share. In a real system this would be the attention/MLP projections of
an open model (Qwen/Llama-class) loaded once and served with many adapters.
"""
from __future__ import annotations

import numpy as np

from app.lora import LoRAConfig, LoRALinear


class BaseModel:
    """One frozen projection wrapped in a LoRA layer."""

    def __init__(self, in_dim: int = 6, out_dim: int = 3,
                 config: LoRAConfig | None = None, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        W0 = rng.normal(0, 0.5, size=(out_dim, in_dim))     # frozen base weight
        self.layer = LoRALinear(W0, config=config, seed=seed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.layer(x)

    def with_adapter(self, state: dict | None) -> "BaseModel":
        """Attach an adapter (or detach with None) and return self for chaining."""
        if state is None:
            self.layer.clear_adapter()
        else:
            self.layer.load_adapter(state)
        return self

    def param_counts(self) -> dict:
        return {"base": self.layer.n_base(), "adapter": self.layer.n_trainable()}
