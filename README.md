<div align="center">

# ⊹ loraforge

**A from-scratch toolkit for self-hosted LLM apps: LoRA fine-tuning + a grounded RAG retrieval stack.**

Two halves of a production GenAI system, implemented from first principles and dependency-light so every idea is inspectable and unit-testable offline on CPU:
**(1) LoRA** — the real low-rank math, an adapter registry, per-request hot-swapping, and merging.
**(2) RAG** — a wiki-style chunk index, HyDE, BM25 reranking, RRF fusion, a scope-gate decision layer, an eval harness, and an RSI-style self-improvement loop.

[![CI](https://github.com/Naveenkumaar/loraforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Naveenkumaar/loraforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/core-NumPy_only-013243?logo=numpy&logoColor=white)](app/lora.py)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)](tests/)
[![Runs offline](https://img.shields.io/badge/default-offline_CPU-238636)](scripts/demo.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> A portfolio demonstration of how large self-hosted LLMs are cheaply fine-tuned
> and served: freeze the base, train a tiny low-rank adapter per task, and swap
> adapters onto one shared base at serve time. Implemented on small matrices so
> the whole thing is unit-testable with no GPU and no downloads.

---

## Why LoRA

Fully fine-tuning a large model means training and storing a full copy of its
weights per task — expensive and slow to serve. **LoRA** freezes the base weight
`W₀` and learns a small **low-rank** update instead:

```
y = x W₀ᵀ + b + (alpha/r) · (x Aᵀ) Bᵀ
```

`A` is `r×in`, `B` is `out×r`. On a 512×512 projection with `r=4` that's **4,096
trainable parameters vs 262,144** — under 2% — and each saved adapter is
kilobytes, so one base model can host many behaviours and switch between them
with no reload.

---

## Quick start

Runs **offline on CPU** — NumPy only.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# ⭐ LoRA demo (train adapters → hot-swap → merge)
.venv/bin/python scripts/demo.py

# ⭐ RAG demo (scope-gate → HyDE → recall → rerank → eval → self-improve)
.venv/bin/python scripts/rag_demo.py

# run the tests
.venv/bin/python -m pytest -q
```

---

## What it does

```
                 ┌──────────────┐   train adapter (base frozen)
  domain data ──►│  train_adapter│──► adapter.json  (only A,B — kilobytes)
                 └──────────────┘          │
                                           ▼
   one BaseModel  ◄── attach / hot-swap ── AdapterRegistry
        │                                     ▲
        ▼                                     │
   AdapterServer.infer(x, adapter="billing")  swap("travel")  ...
```

- **`app/lora.py`** — `LoRALinear`: the low-rank layer, forward pass, exact MSE
  gradients for `A`/`B` only (base frozen), `merged_weight()` / `merge()`, and
  adapter (de)serialization (the base weight never travels).
- **`app/train.py`** — fit an adapter by gradient descent on the adapter alone.
- **`app/registry.py`** — save/list adapters (JSON + optional SQLite), and
  `attach` / `swap` onto a live base layer.
- **`app/serve.py`** — one base model, adapters routed per request with a swap cache.
- **`app/model.py`, `app/tasks.py`** — a tiny frozen base + synthetic domains.

Everything mirrors production LoRA serving; only the matrices are small.

---

## Verified by tests

- Adapter is a **no-op at init** (B=0) and the **base weight never moves** during training.
- The **merged weight** reproduces the separate-path output exactly.
- Training **reduces loss** and a domain's adapter **beats the bare base** on held-out data.
- **Hot-swapping** adapters on one base changes the output with no state leak, and the
  right adapter wins on its own domain.
- The adapter is **<2% of the base** at realistic layer sizes.

---

## Swapping in a real model

The base is behind a small interface, so replace the toy projection with an open
LLM's attention/MLP projections (Qwen/Llama-class) and the synthetic domains with
a real dataset — the registry, serving, and merge logic are unchanged. A PyTorch
+ PEFT backend is the natural next step; the NumPy core keeps the concepts (and
the tests) honest and dependency-free.

---

## The RAG stack (`app/rag/`)

The second half — a grounded retrieval pipeline, each stage a standard, public
technique implemented from scratch and swappable:

| Module | What it is |
|--------|-----------|
| `corpus.py` | wiki-style articles → overlapping chunks (the retrieval unit) |
| `index.py` | **TF-IDF cosine** first-stage recall (with stopword filtering) |
| `rerank.py` | **BM25 (Okapi)** second-stage reranker — term saturation + length norm |
| `hyde.py` | **HyDE** — write a hypothetical answer, retrieve with *that*, blended with the query |
| `fusion.py` | **Reciprocal Rank Fusion** — combine lexical + dense rankings, score-free |
| `scope.py` | **scope gate** — nearest-centroid routing that refuses out-of-domain queries before any LLM call |
| `search.py` | the pipeline: (HyDE) → recall → rerank, every hit traceable to its chunk |
| `evalharness.py` | **recall@k / MRR / nDCG** — cheap, no-LLM metrics to gate quality in CI |
| `improve.py` | **RSI-style self-improvement** — hill-climb the retrieval policy on an eval set (MRR), deterministic |

```bash
.venv/bin/python scripts/rag_demo.py   # scope-gate → HyDE → recall → rerank → RRF → eval → self-improve
```

Same discipline as the LoRA half: pure-Python, offline, fully tested. Swap the
built-in corpus for a real one, or the TF-IDF index for a dense embedding index,
without touching the reranker, fusion, scope gate, eval, or self-improvement loop.

---

<div align="center">

Built from scratch as a portfolio demonstration of LoRA fine-tuning, adapter serving &amp; grounded RAG · [MIT License](LICENSE)

</div>
