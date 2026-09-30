"""Quantization (QLoRA base) and DoRA / QDoRA adapters."""
import numpy as np

from app.dora import DoRALinear, make_quantized_base, train_dora
from app.lora import LoRAConfig, LoRALinear
from app.quant import QuantizedBase, dequantize_int8, quantization_error, quantize_int8
from app.model import BaseModel
from app.tasks import make_domain, split
from app.train import evaluate, train_adapter


# ---- quantization ----
def test_int8_roundtrip_is_close():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 12))
    codes, scales = quantize_int8(W)
    assert codes.dtype == np.int8
    err = np.mean((W - dequantize_int8(codes, scales)) ** 2)
    assert err < 1e-3                              # int8 is near-lossless here


def test_quantized_base_saves_memory():
    W = np.random.default_rng(1).normal(size=(64, 64))
    qb = QuantizedBase(W)
    assert qb.bytes_stored() < QuantizedBase.bytes_full(W) * 0.35   # ~4x smaller


# ---- QLoRA: LoRA on a quantized (lossy) base still learns ----
def test_qlora_fits_despite_quantization():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    Wq = make_quantized_base(base.layer.W0)        # frozen quantized base
    assert quantization_error(base.layer.W0) >= 0.0
    dom = make_domain("billing", seed=1)
    tr, te = split(dom)
    state, hist = train_adapter(Wq, tr["X"], tr["Y"], config=LoRAConfig(r=4, alpha=8.0), steps=300)
    assert hist[-1] < hist[0] * 0.25               # adapter recovers on the lossy base
    adapted = LoRALinear(Wq, seed=0); adapted.load_adapter(state)
    assert evaluate(adapted, te["X"], te["Y"]) < 0.1


# ---- DoRA ----
def test_dora_is_noop_at_init():
    W = np.random.default_rng(2).normal(size=(3, 6))
    lyr = DoRALinear(W, config=LoRAConfig(r=4, alpha=8.0))
    x = np.random.default_rng(3).normal(size=(5, 6))
    assert np.allclose(lyr(x), x @ W.T)            # magnitude=colnorm, B=0 → exact base


def test_dora_trains_and_freezes_base():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    W0 = base.layer.W0.copy()
    dom = make_domain("travel", seed=2)
    tr, _ = split(dom)
    state, hist = train_dora(W0, tr["X"], tr["Y"], config=LoRAConfig(r=4, alpha=8.0),
                             steps=120, lr=0.1)
    assert hist[-1] < hist[0] * 0.5               # numerical DoRA reduces loss
    assert np.allclose(base.layer.W0, W0)         # base untouched
    assert state["kind"] == "dora" and "m" in state


def test_qdora_composes_quant_and_dora():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    Wq = make_quantized_base(base.layer.W0)        # quantized frozen base
    dom = make_domain("support", seed=3)
    tr, _ = split(dom)
    state, hist = train_dora(Wq, tr["X"], tr["Y"], steps=100, lr=0.1)
    assert hist[-1] < hist[0]                      # QDoRA = quantized base + DoRA, still learns
    assert state["kind"] == "dora"


def test_dora_has_magnitude_params_beyond_lora():
    W = np.zeros((5, 7))
    d = DoRALinear(W, config=LoRAConfig(r=2, alpha=4.0))
    lora_only = 2 * (7 + 5)
    assert d.n_trainable() == lora_only + 7        # + one magnitude per column
