"""LoRA — Low-Rank Adaptation, implemented from scratch in NumPy.

A linear layer ``y = x W0ᵀ + b`` is adapted by adding a **low-rank** update
instead of touching the (frozen) base weight ``W0``:

    y = x W0ᵀ + b + s · (x Aᵀ) Bᵀ           s = alpha / r

``A`` is ``(r, in)`` and ``B`` is ``(out, r)`` — together only ``r·(in+out)``
trainable parameters instead of ``out·in``. ``B`` is initialised to zeros so the
adapter is a no-op at start (training only ever *adds* behaviour). The update can
be **merged** into a single weight ``W0 + s·B A`` for zero-overhead serving, or
kept separate so many adapters share one base model and can be hot-swapped.

This is the same technique used to fine-tune large self-hosted LLMs cheaply; here
it runs on small matrices so it is fully testable offline with no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 8.0

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class LoRALinear:
    """A frozen base linear layer plus a trainable low-rank adapter."""

    def __init__(self, W0: np.ndarray, b: np.ndarray | None = None,
                 config: LoRAConfig | None = None, seed: int = 0) -> None:
        self.W0 = np.asarray(W0, dtype=float)        # (out, in) — frozen
        self.out_dim, self.in_dim = self.W0.shape
        self.b = np.zeros(self.out_dim) if b is None else np.asarray(b, dtype=float)
        self.cfg = config or LoRAConfig()
        rng = np.random.default_rng(seed)
        # standard init: A ~ small random, B = 0  → adapter starts as a no-op
        self.A = rng.normal(0, 0.01, size=(self.cfg.r, self.in_dim))
        self.B = np.zeros((self.out_dim, self.cfg.r))
        self.enabled = True

    # ---- forward -------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = x @ self.W0.T + self.b
        if self.enabled:
            y = y + self.cfg.scaling * (x @ self.A.T) @ self.B.T
        return y

    __call__ = forward

    # ---- the merged weight (base + adapter) ----------------------------
    def merged_weight(self) -> np.ndarray:
        """W0 + s·B A — a single equivalent weight for zero-overhead serving."""
        return self.W0 + self.cfg.scaling * (self.B @ self.A)

    def merge(self) -> None:
        """Fold the adapter into the base weight and reset the adapter to a no-op."""
        self.W0 = self.merged_weight()
        self.B[:] = 0.0

    # ---- adapter (de)serialization — base weight never travels ---------
    def adapter_state(self) -> dict:
        return {"r": self.cfg.r, "alpha": self.cfg.alpha,
                "A": self.A.tolist(), "B": self.B.tolist()}

    def load_adapter(self, state: dict) -> None:
        if state["A"] and len(state["A"][0]) != self.in_dim:
            raise ValueError("adapter in_dim does not match this layer")
        if state["B"] and len(state["B"]) != self.out_dim:
            raise ValueError("adapter out_dim does not match this layer")
        self.cfg = LoRAConfig(r=int(state["r"]), alpha=float(state["alpha"]))
        self.A = np.asarray(state["A"], dtype=float)
        self.B = np.asarray(state["B"], dtype=float)

    def clear_adapter(self) -> None:
        """Detach: back to the pure base model."""
        self.A = np.zeros((self.cfg.r, self.in_dim))
        self.B = np.zeros((self.out_dim, self.cfg.r))

    # ---- gradients w.r.t. the adapter only (base is frozen) ------------
    def grads_mse(self, x: np.ndarray, target: np.ndarray):
        """Return (loss, dA, dB) for mean-squared-error against ``target``.

        Only ``A`` and ``B`` receive gradients — ``W0`` and ``b`` are frozen,
        which is the whole point of LoRA.
        """
        x = np.asarray(x, dtype=float)
        target = np.asarray(target, dtype=float)
        n = x.shape[0]
        z = x @ self.A.T                       # (n, r)
        y = x @ self.W0.T + self.b + self.cfg.scaling * (z @ self.B.T)
        diff = y - target
        loss = float(np.mean(diff ** 2))
        dY = (2.0 / n) * diff                   # (n, out)
        dB = self.cfg.scaling * (dY.T @ z)      # (out, r)
        dZ = self.cfg.scaling * (dY @ self.B)   # (n, r)
        dA = dZ.T @ x                           # (r, in)
        return loss, dA, dB

    def n_trainable(self) -> int:
        return self.A.size + self.B.size

    def n_base(self) -> int:
        return self.W0.size + self.b.size
