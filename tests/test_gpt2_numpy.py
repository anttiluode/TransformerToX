import numpy as np
import pytest

from transformer_to_x.gpt2_numpy import GPT2Numpy, MemorySpec, memory_attention, token_nll
from transformer_to_x.window_residue import WindowResidueMemory, rhos_from_taus


def _loop(q, k, v, scale, spec):
    H, T, d = q.shape
    out = np.zeros_like(q)
    for h in range(H):
        mem = WindowResidueMemory(d_head=d, window=spec.window, sink=spec.sink,
                                  rhos=spec.rhos, read_mode=spec.read_mode)
        for t in range(T):
            mem.update(k[h, t], v[h, t])
            out[h, t] = mem.read(q[h, t], scale)
    return out


@pytest.mark.parametrize("mode", ["lowpass", "band", "window_only"])
@pytest.mark.parametrize("sink", [0, 1, 3])
def test_vectorised_attention_equals_stepwise_reference(mode, sink):
    rng = np.random.default_rng(sink + len(mode))
    q, k, v = (rng.normal(size=(3, 40, 5)) for _ in range(3))
    spec = MemorySpec(window=6, sink=sink, rhos=rhos_from_taus((2, 4, 8)), read_mode=mode)
    np.testing.assert_allclose(memory_attention(q, k, v, 0.7, spec), _loop(q, k, v, 0.7, spec), atol=1e-12)


def test_plain_window_and_full_cache():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(2, 20, 4)) for _ in range(3))
    full = memory_attention(q, k, v, 0.5, MemorySpec())
    big = memory_attention(q, k, v, 0.5, MemorySpec(window=20, sink=1, rhos=(0.5,)))
    np.testing.assert_allclose(full, big, atol=1e-12)
    np.testing.assert_allclose(memory_attention(q, k, v, 0.5, MemorySpec(window=5)),
                               _loop(q, k, v, 0.5, MemorySpec(window=5)), atol=1e-12)


def test_budgets():
    r = rhos_from_taus((8, 16, 32, 64, 128, 256, 512))
    assert MemorySpec(window=47, sink=1).scalar_budget(64) == 6144
    assert MemorySpec(window=39, sink=1, rhos=r).scalar_budget(64) == 6023
    assert MemorySpec(window=127, sink=1).scalar_budget(64) == 16384
    assert MemorySpec(window=119, sink=1, rhos=r).scalar_budget(64) == 16263
    assert MemorySpec().scalar_budget(64) is None


def _tiny_model(seed=0, D=8, H=2, V=11, P=64, layers=2):
    rng = np.random.default_rng(seed)
    n = lambda *s: rng.normal(size=s) * 0.3
    blocks = [{
        "ln1_w": 1 + n(D), "ln1_b": n(D), "attn_w": n(D, 3 * D), "attn_b": n(3 * D),
        "proj_w": n(D, D), "proj_b": n(D), "ln2_w": 1 + n(D), "ln2_b": n(D),
        "fc_w": n(D, 4 * D), "fc_b": n(4 * D), "mproj_w": n(4 * D, D), "mproj_b": n(D),
    } for _ in range(layers)]
    return GPT2Numpy(wte=n(V, D), wpe=n(P, D), blocks=blocks, ln_f_w=1 + n(D), ln_f_b=n(D), n_head=H)


def test_forward_is_causal_and_memory_specs_agree_when_nothing_is_evicted():
    m = _tiny_model()
    ids = np.random.default_rng(2).integers(0, 11, size=30)
    full = m.logits(ids)
    np.testing.assert_allclose(m.logits(ids[:12]), full[:12], atol=1e-12)  # causal
    np.testing.assert_allclose(m.logits(ids, MemorySpec(window=30)), full, atol=1e-12)
    short = m.logits(ids, MemorySpec(window=4, sink=1))
    np.testing.assert_allclose(short[:5], full[:5], atol=1e-12)
    assert np.abs(short[10:] - full[10:]).max() > 1e-6


def test_token_nll_matches_direct_computation():
    rng = np.random.default_rng(3)
    lg = rng.normal(size=(6, 9))
    ids = rng.integers(0, 9, size=6)
    p = np.exp(lg) / np.exp(lg).sum(-1, keepdims=True)
    np.testing.assert_allclose(token_nll(lg, ids), -np.log(p[np.arange(5), ids[1:]]))
