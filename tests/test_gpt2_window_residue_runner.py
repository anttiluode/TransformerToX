import numpy as np

from experiments.run_gpt2_window_residue import (
    FAR_START,
    FULL_WINDOW,
    HYBRID_WINDOW,
    TAU_LADDERS,
    build_window_residue_receipt,
    load_long_contexts,
)
from transformer_to_x.gpt2_adapter import GPT2Parity
from transformer_to_x.pretrained_shadow import PretrainedHeadGeometry, reconstruct_head
from transformer_to_x.window_residue import WindowResidueMemory, rhos_from_taus


def _geometry(d_model=8, d_head=4):
    rng = np.random.default_rng(7)
    return PretrainedHeadGeometry(
        d_model=d_model, d_head=d_head,
        w_q=rng.normal(size=(d_model, d_head)), w_k=rng.normal(size=(d_model, d_head)),
        w_v=rng.normal(size=(d_model, d_head)),
        b_q=np.zeros(d_head), b_k=np.zeros(d_head), b_v=np.zeros(d_head),
        w_o=rng.normal(size=(d_head, d_model)), score_scale=0.5,
    )


def _fake_capture():
    g = _geometry()

    def capture(text):
        rng = np.random.default_rng(sum(map(ord, text)) % 1000)
        x = rng.normal(size=(min(len(text.split()) + 3, 40), g.d_model))
        return g, x, reconstruct_head(g, x).contribution, GPT2Parity(0.0, 0.0, 0.0, True), x.shape[0]

    return capture


def test_frozen_constants_and_budgets():
    assert FULL_WINDOW == 48 and HYBRID_WINDOW == 40 and FAR_START == 48
    assert WindowResidueMemory(d_head=64, window=FULL_WINDOW).scalar_state_budget == 6144
    for taus in TAU_LADDERS:
        assert len(taus) == 7
        b = WindowResidueMemory(d_head=64, window=HYBRID_WINDOW, rhos=rhos_from_taus(taus)).scalar_state_budget
        assert b <= 6144


def test_long_fixture_is_disjoint_and_long():
    fx = load_long_contexts()
    assert len(fx["selection"]) == 3 and len(fx["held_out"]) == 5
    assert all(len(t.split()) >= 150 for t in fx["selection"] + fx["held_out"])


def test_receipt_runs_end_to_end_on_fake_head():
    long_ctx = {
        "selection": [" ".join(["alpha"] * 30), " ".join(["beta"] * 31)],
        "held_out": [" ".join(["gamma"] * 32), " ".join(["delta"] * 33)],
    }
    short_ctx = {"selection": ["a b c"], "held_out": ["d e f", "g h i j"]}
    r = build_window_residue_receipt(
        long_ctx, short_ctx, _fake_capture(), {"fake": True},
        full_window=12, hybrid_window=8, far_start=12, tau_ladders=((2, 4), (4, 8)),
        require_canonical_shape=False,
    )
    assert r["classification"].startswith(("PASS_", "FAIL_WINDOW"))
    assert r["short_context_check"]["kv_window_full_relative_rmse"] < 1e-12
    m = r["held_out"]["methods"]
    assert abs(m["zero"]["far"]["relative_rmse"] - 1.0) < 1e-12
    assert set(r["held_out"]["questions"]) == {
        "primary_band_beats_equal_budget_window",
        "residue_adds_over_its_own_window",
        "band_beats_lowpass_same_state",
    }
    assert r["boundary"]["removes_kv_cache"] is False
