import numpy as np

from transformer_to_x.gpt2_numpy import MemorySpec
from transformer_to_x.cache_policies import PolicySpec
from transformer_to_x.neox_numpy import NeoXNumpy, apply_partial_rope, rope_cos_sin


def test_rope_scores_depend_only_on_relative_offset():
    rng = np.random.default_rng(0)
    d = 8
    cos, sin = rope_cos_sin(40, d, 10000.0)
    a, b = rng.normal(size=d), rng.normal(size=d)
    rot = lambda x, t: apply_partial_rope(np.tile(x, (1, 40, 1)), cos, sin, d)[0, t]
    s1 = rot(a, 30) @ rot(b, 25)
    s2 = rot(a, 12) @ rot(b, 7)
    np.testing.assert_allclose(s1, s2, atol=1e-10)


def test_partial_rope_leaves_tail_untouched():
    x = np.random.default_rng(1).normal(size=(2, 10, 8))
    cos, sin = rope_cos_sin(10, 4, 10000.0)
    np.testing.assert_array_equal(apply_partial_rope(x, cos, sin, 4)[..., 4:], x[..., 4:])


def _tiny(seed=0, D=8, H=2, V=13, layers=2):
    rng = np.random.default_rng(seed)
    n = lambda *s: rng.normal(size=s) * 0.3
    blocks = [{
        "ln1_w": 1 + n(D), "ln1_b": n(D), "ln2_w": 1 + n(D), "ln2_b": n(D),
        "qkv_w": n(3 * D, D), "qkv_b": n(3 * D), "o_w": n(D, D), "o_b": n(D),
        "up_w": n(4 * D, D), "up_b": n(4 * D), "down_w": n(D, 4 * D), "down_b": n(D),
    } for _ in range(layers)]
    return NeoXNumpy(embed=n(V, D), blocks=blocks, ln_f_w=1 + n(D), ln_f_b=n(D), unembed=n(V, D),
                     n_head=H, rot_dim=2, rope_base=10000.0, eps=1e-5)


def test_neox_causal_and_memory_consistency():
    m = _tiny()
    ids = np.random.default_rng(2).integers(0, 13, size=25)
    full = m.logits(ids)
    np.testing.assert_allclose(m.logits(ids[:10]), full[:10], atol=1e-12)
    np.testing.assert_allclose(m.logits(ids, MemorySpec(window=25)), full, atol=1e-12)
    np.testing.assert_allclose(m.logits(ids, PolicySpec("h2o", 25)), full, atol=1e-12)
    assert np.abs(m.logits(ids, MemorySpec(window=4, sink=1))[10:] - full[10:]).max() > 1e-6
