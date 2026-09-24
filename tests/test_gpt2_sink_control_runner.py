import numpy as np

from experiments.run_gpt2_sink_control import (
    FROZEN_TAUS,
    PRIMARY_READ,
    SINK,
    build_sink_receipt,
    load_fresh_contexts,
    method_factories,
)
from experiments.run_gpt2_window_residue import load_long_contexts
from tests.test_gpt2_window_residue_runner import _fake_capture


def test_frozen_choices_and_canonical_budgets():
    assert FROZEN_TAUS == (8, 16, 32, 64, 128, 256, 512)
    assert PRIMARY_READ == "lowpass" and SINK == 1
    b = {n: f().scalar_state_budget for n, f in method_factories(64).items()}
    assert b["kv_window_full"] == b["sink_window_full"] == 6144
    assert b["sink_hybrid_lowpass"] == b["sink_hybrid_band"] == b["nosink_hybrid_band"] == 6023
    assert b["sink_window_hybrid_only"] == 6023  # stores the levels, reads only 40 exact pairs (5,120)


def test_fresh_fixture_is_disjoint_from_every_earlier_fixture():
    fresh = set(load_fresh_contexts())
    g2 = load_long_contexts()
    assert len(fresh) == 5
    assert fresh.isdisjoint(g2["selection"] + g2["held_out"])
    assert all(len(t.split()) >= 150 for t in fresh)


def test_receipt_end_to_end_on_fake_head():
    fresh = [" ".join([w] * 30) for w in ("aa", "bb", "cc")]
    g2 = {"selection": ["x " * 30], "held_out": [" ".join(["dd"] * 31), " ".join(["ee"] * 32)]}
    r = build_sink_receipt(fresh, g2, _fake_capture(), {"fake": True}, far_start=12,
                           taus=(2, 4), require_canonical_shape=False)
    assert r["classification"].startswith(("PASS_", "FAIL_TEMPORAL"))
    for key in ("fresh", "gate2_fixture"):
        m = r[key]["methods"]
        assert abs(m["zero"]["relative_rmse"] - 1.0) < 1e-12
        assert m["zero"]["rms_norm_ratio_pred_over_target"] == 0.0
        d = r[key]["head_diagnostics"]
        assert 0.0 <= d["mean_attention_token0"] <= d["mean_attention_first4"] <= 1.0
