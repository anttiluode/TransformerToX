import importlib

import numpy as np
import pytest

from transformer_to_x.tiny_head import TinyCausalHead


def _compiled_head_class():
    try:
        module = importlib.import_module("transformer_to_x.compiler")
    except ModuleNotFoundError:
        pytest.fail("transformer_to_x.compiler is not implemented yet")
    assert hasattr(module, "CompiledHead"), "CompiledHead is not implemented yet"
    return module.CompiledHead


def test_compiled_head_matches_reference_logits_attention_and_output():
    CompiledHead = _compiled_head_class()
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    compiled = CompiledHead.from_weights(head.weights)
    x = np.random.default_rng(99).normal(size=(16, 12))
    ref = head.reference(x)
    got = compiled.run(x)
    finite = np.isfinite(ref.logits)
    np.testing.assert_allclose(got.logits[finite], ref.logits[finite], atol=2e-14, rtol=2e-14)
    assert np.array_equal(np.isneginf(got.logits), np.isneginf(ref.logits))
    np.testing.assert_allclose(got.attention, ref.attention, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(got.output, ref.output, atol=2e-14, rtol=2e-14)


def test_compiler_operators_have_expected_shapes():
    CompiledHead = _compiled_head_class()
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    compiled = CompiledHead.from_weights(head.weights)
    assert compiled.address_operator.shape == (12, 12)
    assert compiled.write_operator.shape == (12, 12)
