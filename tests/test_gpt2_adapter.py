import numpy as np
import pytest

from transformer_to_x.gpt2_adapter import (
    GPT2Capture,
    slice_gpt2_head,
    verify_gpt2_capture,
)
from transformer_to_x.pretrained_shadow import reconstruct_head


def packed_fixture():
    d_model = 6
    n_head = 3
    packed_w = np.arange(d_model * 3 * d_model, dtype=float).reshape(d_model, 3 * d_model)
    packed_b = np.arange(3 * d_model, dtype=float) + 1000.0
    proj_w = np.arange(d_model * d_model, dtype=float).reshape(d_model, d_model) + 2000.0
    return packed_w, packed_b, proj_w, n_head


def test_slice_gpt2_head_respects_conv1d_packed_orientation():
    packed_w, packed_b, proj_w, n_head = packed_fixture()
    g = slice_gpt2_head(packed_w, packed_b, proj_w, n_head=n_head, head_index=1)
    np.testing.assert_array_equal(g.w_q, packed_w[:, 2:4])
    np.testing.assert_array_equal(g.w_k, packed_w[:, 8:10])
    np.testing.assert_array_equal(g.w_v, packed_w[:, 14:16])
    np.testing.assert_array_equal(g.b_q, packed_b[2:4])
    np.testing.assert_array_equal(g.b_k, packed_b[8:10])
    np.testing.assert_array_equal(g.b_v, packed_b[14:16])
    np.testing.assert_array_equal(g.w_o, proj_w[2:4, :])
    assert g.score_scale == pytest.approx(1 / np.sqrt(2.0))


def test_slice_rejects_invalid_head_and_unsupported_scaling():
    packed_w, packed_b, proj_w, n_head = packed_fixture()
    with pytest.raises(ValueError):
        slice_gpt2_head(packed_w, packed_b, proj_w, n_head=n_head, head_index=3)
    with pytest.raises(ValueError):
        slice_gpt2_head(
            packed_w,
            packed_b,
            proj_w,
            n_head=n_head,
            head_index=0,
            scale_attn_by_inverse_layer_idx=True,
        )
    with pytest.raises(ValueError):
        slice_gpt2_head(
            packed_w,
            packed_b,
            proj_w,
            n_head=n_head,
            head_index=0,
            reorder_and_upcast_attn=True,
        )


def test_single_head_contribution_excludes_shared_cproj_bias():
    d_model = 4
    d_head = 2
    m0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    m1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    pre = np.concatenate([m0, m1], axis=-1)
    proj_w = np.arange(16, dtype=float).reshape(4, 4) / 10
    bias = np.array([0.4, -0.3, 0.2, 0.1])
    full = pre @ proj_w + bias
    h0 = m0 @ proj_w[:d_head, :]
    h1 = m1 @ proj_w[d_head:, :]
    np.testing.assert_allclose(full, h0 + h1 + bias)
    changed_bias = bias + 10
    np.testing.assert_allclose(h0, m0 @ proj_w[:d_head, :])
    assert not np.allclose(full, pre @ proj_w + changed_bias)


def test_verify_gpt2_capture_accepts_exact_synthetic_module_decomposition():
    rng = np.random.default_rng(42)
    d_model = 4
    n_head = 2
    packed_w = rng.normal(size=(d_model, 3 * d_model))
    packed_b = rng.normal(size=(3 * d_model,))
    proj_w = rng.normal(size=(d_model, d_model))
    proj_b = rng.normal(size=(d_model,))
    head_index = 1
    g = slice_gpt2_head(packed_w, packed_b, proj_w, n_head=n_head, head_index=head_index)
    x = rng.normal(size=(3, d_model))
    mixtures = []
    attentions = []
    for h in range(n_head):
        gh = slice_gpt2_head(packed_w, packed_b, proj_w, n_head=n_head, head_index=h)
        refh = reconstruct_head(gh, x)
        mixtures.append(refh.mixture)
        attentions.append(refh.attention)
    pre = np.concatenate(mixtures, axis=-1)
    out = pre @ proj_w + proj_b
    capture = GPT2Capture(
        x=x,
        attention=attentions[head_index],
        pre_cproj=pre,
        cproj_output=out,
    )
    parity = verify_gpt2_capture(
        g,
        capture,
        c_proj_weight=proj_w,
        c_proj_bias=proj_b,
        head_index=head_index,
        tolerance=1e-12,
    )
    assert parity.passed
    assert parity.max_attention_abs_error < 1e-12
    assert parity.max_mixture_abs_error < 1e-12
    assert parity.max_full_module_abs_error < 1e-12


def test_verify_gpt2_capture_rejects_bad_shapes_and_nonfinite_arrays():
    rng = np.random.default_rng(0)
    d_model = 4
    n_head = 2
    packed_w = rng.normal(size=(d_model, 3 * d_model))
    packed_b = rng.normal(size=(3 * d_model,))
    proj_w = rng.normal(size=(d_model, d_model))
    g = slice_gpt2_head(packed_w, packed_b, proj_w, n_head=n_head, head_index=0)
    x = rng.normal(size=(2, d_model))
    ref = reconstruct_head(g, x)
    with pytest.raises(ValueError):
        verify_gpt2_capture(
            g,
            GPT2Capture(x=x, attention=np.ones((2, 3)), pre_cproj=np.ones((2, 4)), cproj_output=np.ones((2, 4))),
            c_proj_weight=proj_w,
            c_proj_bias=np.zeros(4),
            head_index=0,
        )
    bad = x.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        GPT2Capture(x=bad, attention=ref.attention, pre_cproj=np.ones((2, 4)), cproj_output=np.ones((2, 4)))
