"""Training an adapter reduces loss and specializes a frozen base to a domain."""
import numpy as np

from app.lora import LoRAConfig, LoRALinear
from app.model import BaseModel
from app.tasks import make_domain, split
from app.train import evaluate, train_adapter


def test_training_reduces_loss():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    dom = make_domain("billing", seed=1)
    tr, _ = split(dom)
    state, hist = train_adapter(base.layer.W0, tr["X"], tr["Y"],
                                config=LoRAConfig(r=4, alpha=8.0), steps=300)
    assert hist[-1] < hist[0] * 0.2          # loss dropped substantially
    assert hist[-1] < 0.05                    # and is genuinely small


def test_base_weight_is_frozen_during_training():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    W0_before = base.layer.W0.copy()
    dom = make_domain("travel", seed=2)
    train_adapter(base.layer.W0, dom["X"], dom["Y"], steps=100)
    assert np.allclose(base.layer.W0, W0_before)   # base never moved


def test_adapter_beats_bare_base_on_its_domain():
    base = BaseModel(in_dim=6, out_dim=3, seed=42)
    dom = make_domain("billing", seed=3)
    tr, te = split(dom)
    bare = evaluate(base.layer, te["X"], te["Y"])   # base only
    state, _ = train_adapter(base.layer.W0, tr["X"], tr["Y"], steps=300)
    adapted = LoRALinear(base.layer.W0, seed=0)
    adapted.load_adapter(state)
    assert evaluate(adapted, te["X"], te["Y"]) < bare * 0.5   # generalizes
