"""Registry round-trip + hot-swapping adapters on one shared base at serve time."""
import numpy as np

from app.model import BaseModel
from app.registry import AdapterRegistry
from app.serve import AdapterServer
from app.tasks import make_domain, split
from app.train import train_adapter


def _trained_state(base, seed):
    dom = make_domain(f"d{seed}", seed=seed)
    tr, _ = split(dom)
    state, _ = train_adapter(base.layer.W0, tr["X"], tr["Y"], steps=200)
    return state, dom


def test_registry_save_load_roundtrip(tmp_path):
    reg = AdapterRegistry(root=tmp_path, db_path=str(tmp_path / "reg.db"))
    base = BaseModel(seed=42)
    state, _ = _trained_state(base, 1)
    reg.save("billing", state)
    assert reg.names() == ["billing"]
    assert np.allclose(reg.load("billing")["A"], state["A"])


def test_hot_swap_changes_output_on_one_base(tmp_path):
    reg = AdapterRegistry(root=tmp_path)
    base = BaseModel(seed=42)
    s1, d1 = _trained_state(base, 1)
    s2, d2 = _trained_state(base, 2)
    reg.save("billing", s1)
    reg.save("travel", s2)

    server = AdapterServer(base, reg)
    x = d1["X"][:1]
    y_billing = server.infer(x, adapter="billing").copy()
    y_travel = server.infer(x, adapter="travel").copy()
    y_base = server.infer(x, adapter=None).copy()

    # each adapter yields a different mapping on the SAME base model
    assert not np.allclose(y_billing, y_travel)
    assert not np.allclose(y_billing, y_base)
    # swapping back reproduces the first result exactly (no state leak)
    assert np.allclose(server.infer(x, adapter="billing"), y_billing)


def test_billing_adapter_fits_billing_domain_best(tmp_path):
    reg = AdapterRegistry(root=tmp_path)
    base = BaseModel(seed=42)
    s1, d1 = _trained_state(base, 1)
    s2, d2 = _trained_state(base, 2)
    reg.save("billing", s1); reg.save("travel", s2)
    server = AdapterServer(base, reg)
    # on domain-1 data, the billing adapter should beat the travel adapter
    x, y = d1["X"][:20], d1["Y"][:20]
    err_billing = np.mean((server.infer(x, "billing") - y) ** 2)
    err_travel = np.mean((server.infer(x, "travel") - y) ** 2)
    assert err_billing < err_travel


def test_unknown_adapter_raises(tmp_path):
    reg = AdapterRegistry(root=tmp_path)
    import pytest
    with pytest.raises(KeyError):
        reg.load("does-not-exist")
