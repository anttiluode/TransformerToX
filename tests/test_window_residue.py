import numpy as np
import pytest

from transformer_to_x.pretrained_shadow import PretrainedHeadGeometry, reconstruct_head
from transformer_to_x.window_residue import (
    WindowResidueMemory,
    ZeroMemory,
    rhos_from_taus,
    run_head_space_context,
)


def random_geometry(seed=0, d_model=16, d_head=4):
    rng = np.random.default_rng(seed)
    return PretrainedHeadGeometry(
        d_model=d_model,
        d_head=d_head,
        w_q=rng.normal(size=(d_model, d_head)) / 2,
        w_k=rng.normal(size=(d_model, d_head)) / 2,
        w_v=rng.normal(size=(d_model, d_head)),
        b_q=rng.normal(size=d_head) / 4,
        b_k=rng.normal(size=d_head) / 4,
        b_v=rng.normal(size=d_head) / 4,
        w_o=rng.normal(size=(d_head, d_model)),
        score_scale=1.0 / np.sqrt(d_head),
    )


def test_window_at_least_sequence_length_is_exact_attention():
    g = random_geometry()
    x = np.random.default_rng(1).normal(size=(12, g.d_model))
    ref = reconstruct_head(g, x).contribution
    for mode in WindowResidueMemory.READ_MODES:
        mem = WindowResidueMemory(d_head=g.d_head, window=12, rhos=rhos_from_taus((2, 4)), read_mode=mode)
        np.testing.assert_allclose(run_head_space_context(g, x, mem), ref, atol=1e-12)


def test_short_window_is_not_exact_on_longer_sequence():
    g = random_geometry()
    x = np.random.default_rng(2).normal(size=(30, g.d_model))
    ref = reconstruct_head(g, x).contribution
    pred = run_head_space_context(g, x, WindowResidueMemory(d_head=g.d_head, window=5))
    np.testing.assert_allclose(pred[:5], ref[:5], atol=1e-12)
    assert np.max(np.abs(pred[5:] - ref[5:])) > 1e-6


def test_budget_counts_window_traces_and_masses():
    mem = WindowResidueMemory(d_head=64, window=40, rhos=rhos_from_taus((2, 4, 8, 16, 32, 64, 128)))
    assert mem.scalar_state_budget == 40 * 128 + 7 * 129 == 6023
    assert WindowResidueMemory(d_head=64, window=48).scalar_state_budget == 6144
    assert ZeroMemory(d_head=64).scalar_state_budget == 0


def test_bands_are_nonnegative_time_windows_that_telescope_to_slowest_level():
    rng = np.random.default_rng(3)
    mem = WindowResidueMemory(d_head=3, window=2, rhos=rhos_from_taus((2, 8, 32)))
    for _ in range(40):
        mem.update(rng.normal(size=3), rng.normal(size=3))
    bands = mem.band_masses()
    assert np.all(bands >= 0.0)
    np.testing.assert_allclose(np.sum(bands), mem._mass[-1])
    band_k_sum = np.sum(np.diff(mem._sk, axis=0, prepend=np.zeros((1, 3))), axis=0)
    np.testing.assert_allclose(band_k_sum, mem._sk[-1])


def test_levels_only_receive_evicted_tokens():
    mem = WindowResidueMemory(d_head=2, window=3, rhos=(0.5,))
    for i in range(3):
        mem.update(np.ones(2) * i, np.ones(2))
    assert mem._mass[0] == 0.0
    mem.update(np.zeros(2), np.zeros(2))
    assert mem._mass[0] == 1.0
    np.testing.assert_allclose(mem._sk[0], np.zeros(2))  # first evicted key was 0


def test_identical_keys_make_band_read_exact():
    # Jensen summary is exact when every token in a band shares one key.
    d = 3
    key = np.array([0.3, -0.2, 0.5])
    rng = np.random.default_rng(4)
    vals = rng.normal(size=(25, d))
    q = rng.normal(size=d)
    mem = WindowResidueMemory(d_head=d, window=4, rhos=rhos_from_taus((2, 4, 8)))
    for v in vals:
        mem.update(key, v)
    got = mem.read(q, 1.0)
    # exact weights: window tokens weight 1, evicted tokens weight = slowest-level decay
    rho = mem.rhos[-1]
    n_ev = 25 - 4
    w = np.concatenate([rho ** np.arange(n_ev)[::-1], np.ones(4)])
    exact = (w[:, None] * vals).sum(0) / w.sum()
    np.testing.assert_allclose(got, exact, atol=1e-12)


def test_lowpass_and_band_share_state_but_differ_in_read():
    rng = np.random.default_rng(5)
    rh = rhos_from_taus((2, 8))
    a = WindowResidueMemory(d_head=3, window=2, rhos=rh, read_mode="band")
    b = WindowResidueMemory(d_head=3, window=2, rhos=rh, read_mode="lowpass")
    for _ in range(20):
        k, v = rng.normal(size=3), rng.normal(size=3)
        a.update(k, v)
        b.update(k, v)
    np.testing.assert_array_equal(a._sk, b._sk)
    q = rng.normal(size=3)
    assert not np.allclose(a.read(q, 1.0), b.read(q, 1.0))


@pytest.mark.parametrize("bad", [(0.5, 0.5), (0.9, 0.5), (1.0,), (-0.1,)])
def test_rejects_bad_rhos(bad):
    with pytest.raises(ValueError):
        WindowResidueMemory(d_head=2, window=1, rhos=bad)


def test_zero_memory_has_unit_relative_rmse():
    from transformer_to_x.metrics import relative_rmse

    g = random_geometry()
    x = np.random.default_rng(6).normal(size=(10, g.d_model))
    ref = reconstruct_head(g, x).contribution
    pred = run_head_space_context(g, x, ZeroMemory(d_head=g.d_head))
    assert relative_rmse(ref, pred) == pytest.approx(1.0)


def test_sink_tokens_are_never_evicted_and_are_counted():
    mem = WindowResidueMemory(d_head=64, window=47, sink=1)
    assert mem.scalar_state_budget == 6144
    hyb = WindowResidueMemory(d_head=64, window=39, sink=1, rhos=rhos_from_taus((8, 16, 32, 64, 128, 256, 512)))
    assert hyb.scalar_state_budget == 40 * 128 + 7 * 129 == 6023
    small = WindowResidueMemory(d_head=2, window=2, sink=1, rhos=(0.5,))
    for i in range(6):
        small.update(np.full(2, float(i)), np.full(2, float(i)))
    assert [k[0] for k in small._sink_k] == [0.0]
    assert [k[0] for k in small._win_k] == [4.0, 5.0]
    assert small._mass[0] == pytest.approx(1 + 0.5 + 0.25)  # tokens 1,2,3 evicted


def test_sink_plus_window_covering_sequence_is_exact():
    g = random_geometry()
    x = np.random.default_rng(9).normal(size=(10, g.d_model))
    ref = reconstruct_head(g, x).contribution
    pred = run_head_space_context(g, x, WindowResidueMemory(d_head=g.d_head, window=9, sink=1))
    np.testing.assert_allclose(pred, ref, atol=1e-12)


def test_sink_restores_mass_when_first_token_dominates():
    # Key of token 0 aligns with every query; later keys are weak. Dropping token 0
    # renormalises attention onto content tokens; keeping it as a sink does not.
    rng = np.random.default_rng(10)
    d = 4
    g = random_geometry(11, d_model=8, d_head=d)
    x = rng.normal(size=(30, 8)) * 0.1
    q_mean = (x @ g.w_q + g.b_q).mean(0)
    # solve for a first-token residual whose key is a large multiple of the mean query
    target_k = 40.0 * q_mean / np.linalg.norm(q_mean)
    x[0] = np.linalg.lstsq(g.w_k.T, target_k - g.b_k, rcond=None)[0]
    ref = reconstruct_head(g, x).contribution
    with_sink = run_head_space_context(g, x, WindowResidueMemory(d_head=d, window=4, sink=1))
    without = run_head_space_context(g, x, WindowResidueMemory(d_head=d, window=5))
    err = lambda p: np.sqrt(np.mean((p[6:] - ref[6:]) ** 2))
    assert err(with_sink) < err(without)
