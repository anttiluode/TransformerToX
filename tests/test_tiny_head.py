import importlib

import numpy as np
import pytest


def _tiny_head_class():
    try:
        module = importlib.import_module("transformer_to_x.tiny_head")
    except ModuleNotFoundError:
        pytest.fail("transformer_to_x.tiny_head is not implemented yet")
    assert hasattr(module, "TinyCausalHead"), "TinyCausalHead is not implemented yet"
    return module.TinyCausalHead


def test_reference_head_is_deterministic_and_causal():
    TinyCausalHead = _tiny_head_class()
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    x = np.random.default_rng(7).normal(size=(8, 12))
    a = head.reference(x)
    b = head.reference(x)
    np.testing.assert_array_equal(a.output, b.output)
    assert np.allclose(np.triu(a.attention, 1), 0.0)

    changed = x.copy()
    changed[-1] += 100.0
    before = head.reference(x).output[:-1]
    after = head.reference(changed).output[:-1]
    np.testing.assert_allclose(before, after, atol=0.0, rtol=0.0)


def test_reference_head_rejects_bad_shape_and_nonfinite_input():
    TinyCausalHead = _tiny_head_class()
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    with pytest.raises(ValueError):
        head.reference(np.zeros((4, 11)))
    bad = np.zeros((4, 12))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        head.reference(bad)
