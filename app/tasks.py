"""Synthetic 'domains' to fine-tune adapters on — deterministic, no downloads.

Each task is a hidden linear mapping ``x -> x Wᵀ`` with its own weight, standing
in for a different downstream domain (e.g. 'billing' vs 'travel' phrasing). A base
model can't serve all of them well; a small LoRA adapter per domain can. Swap the
generator for a real dataset without touching the trainer or the registry.
"""
from __future__ import annotations

import numpy as np


def make_domain(name: str, in_dim: int = 6, out_dim: int = 3,
                n: int = 256, seed: int = 0):
    """Return (X, Y, W) for one domain's linear mapping."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, size=(out_dim, in_dim))
    X = rng.normal(0, 1.0, size=(n, in_dim))
    Y = X @ W.T
    return {"name": name, "X": X, "Y": Y, "W": W}


def split(data: dict, frac: float = 0.8):
    """Deterministic train/test split."""
    n = data["X"].shape[0]
    k = int(n * frac)
    tr = {"X": data["X"][:k], "Y": data["Y"][:k]}
    te = {"X": data["X"][k:], "Y": data["Y"][k:]}
    return tr, te
