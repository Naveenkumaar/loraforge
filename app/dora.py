"""DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

DoRA splits a weight into **magnitude** and **direction**: ``W = m · V/‖V‖_c``
(column-wise norm). It freezes the base, applies a LoRA update to the *direction*
only, and learns a separate **magnitude** vector. Decoupling the two lets DoRA
match full fine-tuning more closely than LoRA at the same rank.

Compose with a quantized base (``app.quant``) and you get **QDoRA**: a 4/8-bit
frozen base + a full-precision DoRA adapter. Implemented on small matrices,
trained with a compact numerical optimizer, so it stays exact and testable.
"""
from __future__ import annotations

import numpy as np

from app.lora import LoRAConfig
from app.quant import QuantizedBase


def _colnorm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=0)
    n[n == 0] = 1.0
    return n


class DoRALinear:
    def __init__(self, W0, b=None, config: LoRAConfig | None = None, seed: int = 0) -> None:
        self.W0 = np.asarray(W0, dtype=float)          # (out, in) — frozen
        self.out_dim, self.in_dim = self.W0.shape
        self.b = np.zeros(self.out_dim) if b is None else np.asarray(b, dtype=float)
        self.cfg = config or LoRAConfig()
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 0.01, size=(self.cfg.r, self.in_dim))
        self.B = np.zeros((self.out_dim, self.cfg.r))
        # magnitude initialised to the base's column norms → adapter is a no-op at start
        self.m = _colnorm(self.W0).copy()
        self.enabled = True

    def _direction(self) -> np.ndarray:
        return self.W0 + self.cfg.scaling * (self.B @ self.A)

    def effective_weight(self) -> np.ndarray:
        V = self._direction()
        return V * (self.m / _colnorm(V))              # scale each column to magnitude m

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.enabled:
            return x @ self.W0.T + self.b
        return x @ self.effective_weight().T + self.b

    __call__ = forward
    merged_weight = effective_weight

    def n_trainable(self) -> int:
        return self.A.size + self.B.size + self.m.size

    def adapter_state(self) -> dict:
        return {"kind": "dora", "r": self.cfg.r, "alpha": self.cfg.alpha,
                "A": self.A.tolist(), "B": self.B.tolist(), "m": self.m.tolist()}

    def load_adapter(self, state: dict) -> None:
        self.cfg = LoRAConfig(r=int(state["r"]), alpha=float(state["alpha"]))
        self.A = np.asarray(state["A"], float)
        self.B = np.asarray(state["B"], float)
        self.m = np.asarray(state["m"], float)


def _loss(layer: DoRALinear, X, Y) -> float:
    return float(np.mean((layer(X) - Y) ** 2))


def train_dora(W0, X, Y, config: LoRAConfig | None = None, steps: int = 300,
               lr: float = 0.05, eps: float = 1e-5, seed: int = 0):
    """Fit a DoRA adapter (A, B, m) by numerical gradient descent; base frozen.

    Numerical grads keep the decomposition exact and the code short — fine on the
    small reference matrices; a real impl would use autograd.
    """
    layer = DoRALinear(W0, config=config, seed=seed)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    history = [_loss(layer, X, Y)]
    params = [("A", layer.A), ("B", layer.B), ("m", layer.m)]
    for _ in range(steps):
        for _, P in params:
            flat = P.ravel()
            grad = np.zeros_like(flat)
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps; lp = _loss(layer, X, Y)
                flat[i] = orig - eps; lm = _loss(layer, X, Y)
                flat[i] = orig
                grad[i] = (lp - lm) / (2 * eps)
            flat -= lr * grad
        history.append(_loss(layer, X, Y))
    return layer.adapter_state(), history


def make_quantized_base(W0) -> np.ndarray:
    """Return the dequantized int8 base — the frozen weight QLoRA/QDoRA train on."""
    return QuantizedBase(W0).weight()
