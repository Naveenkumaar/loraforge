# loraforge — design write-up

A from-scratch LoRA implementation and adapter-serving layer, kept dependency-light
(NumPy) so the ideas are inspectable and the whole thing is unit-testable on CPU.

## 1. The problem

Fully fine-tuning a large self-hosted model per task is expensive to train and to
serve — you end up with a full weight copy per behaviour. I wanted to demonstrate
the production pattern instead: **freeze the base, learn a tiny low-rank adapter
per task, and serve one base model with many adapters swapped in per request.**

## 2. The math (why low-rank works)

A dense update `ΔW` to a projection is `out×in` parameters. LoRA factorises it as
`ΔW = (alpha/r)·B A` with `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}`, `r ≪ min(in,out)`:

```
y = x W₀ᵀ + b + (alpha/r) · (x Aᵀ) Bᵀ
```

- `W₀`, `b` are **frozen**; only `A`, `B` train — `r·(in+out)` params.
- `B` is initialised to **zero**, so the adapter is a **no-op at start** — training
  only ever *adds* behaviour to a working base.
- The two paths are algebraically identical to a single **merged** weight
  `W₀ + (alpha/r)·B A`, so once trained an adapter can be folded in for
  **zero inference overhead**, or kept separate for swapping.

I derive and implement the exact gradients for `A` and `B` under MSE (no autograd),
which is what keeps the base provably frozen.

## 3. Code-level flow

```
train_adapter(W0, X, Y)                  # base frozen; SGD on A,B only
   └─ LoRALinear.grads_mse → (loss, dA, dB)
   └─ returns adapter_state {r, alpha, A, B}   # kilobytes, no base weight

AdapterRegistry.save(name, state)        # JSON (+ optional SQLite metadata)
AdapterServer.infer(x, adapter="billing")
   └─ if adapter != current: registry.swap(layer, adapter)   # hot-swap
   └─ BaseModel.predict(x)               # one shared frozen base
```

## 4. Decisions & why

| Decision | Why | Rejected alternative |
|----------|-----|----------------------|
| NumPy core, no framework | The math stays visible and every step is unit-testable on CPU with no downloads | Jump straight to PyTorch+PEFT — faster to a real model, but the concepts hide behind the framework and CI needs a heavy env |
| `B=0` init | Adapter is a no-op at start — safe to attach to a live base | Random `B` — the model degrades the moment an untrained adapter is attached |
| Adapter = only `A`,`B` in JSON | Shipping a behaviour is kilobytes; the base never travels | Save full merged weights — defeats the point |
| Separate `merge()` path | Zero-overhead serving when you don't need to swap | Always keep adapters separate — small but real per-call cost |
| Registry + per-request routing | One expensive base, many cheap behaviours, swapped live | One process per fine-tune — wastes memory, no hot-swap |

## 5. Problems faced & fixes

| Problem | Cause | Fix | Lesson |
|---------|-------|-----|--------|
| "Adapter smaller than base" test failed | On a 3×6 toy, `r=4` adapter (36) > base (21) | Assert the saving on a realistically-sized 512×512 layer | LoRA's benefit is asymptotic — demonstrate it at scale, not on a toy |
| State leak risk across swaps | Reusing one layer between adapters | `swap()` clears the adapter before attaching the next; a serve-time cache keys on the current adapter name | Make hot-swap explicitly stateless-per-request |

## 6. What it's capable of

- Train a domain adapter on a frozen base; the base provably never moves.
- Save/list adapters and hot-swap them onto one shared base at serve time.
- Merge an adapter for zero-overhead inference; output is identical to the separate path.
- Show the parameter/size win (<2% at realistic sizes).
- **Limits:** a linear base and MSE task (deliberately, for clarity); a real LLM base
  + a task loss + a PyTorch/PEFT backend are the next step, behind the same interfaces.

## 7. If I rebuilt it today

Keep the NumPy core as the reference and the test oracle; add an optional
PyTorch+PEFT backend that plugs into the same registry/serving interfaces, and a
small FastAPI server so adapters can be swapped over HTTP.
