#!/usr/bin/env python3
"""One-command, offline demo of LoRA fine-tuning + hot-swappable adapters.

    python scripts/demo.py

Trains one small LoRA adapter per 'domain' on a frozen base model, saves them to
a registry, then serves one base model and hot-swaps adapters per request —
the exact pattern used to serve a self-hosted LLM with many LoRA adapters.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.lora import LoRAConfig  # noqa: E402
from app.model import BaseModel  # noqa: E402
from app.registry import AdapterRegistry  # noqa: E402
from app.serve import AdapterServer  # noqa: E402
from app.tasks import make_domain, split  # noqa: E402
from app.train import evaluate, train_adapter  # noqa: E402


def rule(t): print(f"\n\033[1m--- {t} ---\033[0m")


def main() -> int:
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    counts = base.param_counts()
    rule("Base model (frozen) + LoRA")
    print(f"base params: {counts['base']}  ·  adapter params (r=4): {counts['adapter']}")
    print("On a 512x512 projection an r=4 adapter is <2% of the base — that's the win.")

    reg = AdapterRegistry(root=Path(tempfile.mkdtemp()) / "adapters")
    domains = {"billing": 1, "travel": 2, "support": 3}

    rule("Fine-tune one adapter per domain (base stays frozen)")
    holdout = {}
    for name, seed in domains.items():
        dom = make_domain(name, seed=seed)
        tr, te = split(dom)
        state, hist = train_adapter(base.layer.W0, tr["X"], tr["Y"],
                                    config=LoRAConfig(r=4, alpha=8.0), steps=400)
        reg.save(name, state)
        holdout[name] = te
        print(f"  {name:8s}: loss {hist[0]:.3f} -> {hist[-1]:.4f}  (saved adapter: {name}.json)")

    rule("Serve ONE base model, hot-swap adapters per request")
    server = AdapterServer(base, reg)
    print(f"  registered adapters: {server.available()}")
    for name in domains:
        te = holdout[name]
        errs = {a: float(np.mean((server.infer(te["X"], a) - te["Y"]) ** 2))
                for a in domains}
        best = min(errs, key=errs.get)
        mark = "OK" if best == name else "??"
        print(f"  input from '{name}': best adapter = '{best}' [{mark}]  "
              f"errors={{" + ", ".join(f'{k}:{v:.3f}' for k, v in errs.items()) + "}}")

    rule("Merge an adapter for zero-overhead serving")
    base.with_adapter(reg.load("billing"))
    x = holdout["billing"]["X"][:1]
    before = base.predict(x).copy()
    base.layer.merge()
    after = base.predict(x)
    print(f"  output identical after merge: {np.allclose(before, after)} "
          f"(adapter folded into the base weight)")

    rule("Done")
    print("All offline, NumPy-only. Swap the synthetic domains for a real dataset,")
    print("or the base for an open LLM's projections, without changing the registry/serving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
