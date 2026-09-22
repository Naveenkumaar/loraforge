"""LoRA layer math: no-op init, merge equivalence, frozen base, param count."""
import numpy as np

from app.lora import LoRAConfig, LoRALinear


def _layer(seed=0):
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(3, 6))
    return LoRALinear(W0, config=LoRAConfig(r=4, alpha=8.0), seed=seed)


def test_adapter_starts_as_a_noop():
    # B initialised to zero → adapter output equals the pure base output
    lyr = _layer()
    x = np.random.default_rng(1).normal(size=(5, 6))
    base = x @ lyr.W0.T + lyr.b
    assert np.allclose(lyr(x), base)


def test_merged_weight_matches_separate_path():
    lyr = _layer()
    # give the adapter some non-zero values
    lyr.A = np.random.default_rng(2).normal(size=lyr.A.shape)
    lyr.B = np.random.default_rng(3).normal(size=lyr.B.shape)
    x = np.random.default_rng(4).normal(size=(7, 6))
    separate = lyr(x)
    merged = x @ lyr.merged_weight().T + lyr.b
    assert np.allclose(separate, merged)


def test_merge_then_disable_is_identical():
    lyr = _layer()
    lyr.A = np.random.default_rng(5).normal(size=lyr.A.shape)
    lyr.B = np.random.default_rng(6).normal(size=lyr.B.shape)
    x = np.random.default_rng(7).normal(size=(4, 6))
    before = lyr(x).copy()
    lyr.merge()                     # fold adapter into base, reset B to 0
    assert np.allclose(lyr(x), before)      # output unchanged
    assert np.allclose(lyr.B, 0.0)          # adapter is now a no-op


def test_disable_falls_back_to_base():
    lyr = _layer()
    lyr.B = np.random.default_rng(8).normal(size=lyr.B.shape)
    x = np.random.default_rng(9).normal(size=(3, 6))
    lyr.enabled = False
    assert np.allclose(lyr(x), x @ lyr.W0.T + lyr.b)


def test_adapter_is_far_smaller_than_base_at_realistic_size():
    # LoRA's whole point: on a large projection, r*(in+out) << out*in.
    big = LoRALinear(np.zeros((512, 512)), config=LoRAConfig(r=4, alpha=8.0))
    assert big.n_trainable() == 4 * (512 + 512)          # 4,096
    assert big.n_base() >= 512 * 512                     # 262,144+
    assert big.n_trainable() < big.n_base() * 0.02       # <2% of the base
    # and the formula holds on the small layer too
    assert _layer().n_trainable() == 4 * (6 + 3)


def test_dimension_mismatch_rejected():
    lyr = _layer()
    import pytest
    with pytest.raises(ValueError):
        lyr.load_adapter({"r": 2, "alpha": 4.0, "A": [[1, 2]], "B": [[0, 0]]})
