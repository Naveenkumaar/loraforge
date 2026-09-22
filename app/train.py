"""Fit a LoRA adapter — full-batch gradient descent on the adapter only.

The base weight never moves; only ``A``/``B`` are updated. Returns the trained
adapter state (portable JSON-able dict) plus the loss curve, so a caller can
save it to the registry and hot-swap it onto a shared base model later.
"""
from __future__ import annotations

import numpy as np

from app.lora import LoRAConfig, LoRALinear


def train_adapter(W0: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  config: LoRAConfig | None = None, steps: int = 400,
                  lr: float = 0.05, seed: int = 0):
    """Train an adapter on top of a frozen ``W0`` to map ``X -> Y``.

    Returns ``(adapter_state, history)`` where history is the per-step loss.
    """
    layer = LoRALinear(W0, config=config, seed=seed)
    history: list[float] = []
    for _ in range(steps):
        loss, dA, dB = layer.grads_mse(X, Y)
        history.append(loss)
        layer.A -= lr * dA
        layer.B -= lr * dB
    # final loss after the last update
    history.append(layer.grads_mse(X, Y)[0])
    return layer.adapter_state(), history


def evaluate(layer: LoRALinear, X: np.ndarray, Y: np.ndarray) -> float:
    """Mean-squared error of the (base + current adapter) on held-out data."""
    diff = layer(X) - np.asarray(Y, dtype=float)
    return float(np.mean(diff ** 2))
