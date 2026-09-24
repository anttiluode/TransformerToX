import numpy as np

from experiments.run_gpt2_perplexity import (
    FAR_START,
    TAUS,
    load_text,
    method_specs,
    questions,
    score_methods,
)
from tests.test_gpt2_numpy import _tiny_model


def test_frozen_text_and_budgets():
    assert len(load_text()) > 40000
    specs = method_specs()
    b = {k: s.scalar_budget(64) for k, s in specs.items()}
    assert b["full_cache"] is None
    for k in ("b48_window48", "b48_sink1_window47", "b48_sink4_window44"):
        assert b[k] == 6144
    for k in ("b48_sink1_window39_only", "b48_sink1_window39_lowpass", "b48_sink1_window39_band"):
        assert b[k] == 6023
    assert b["b128_sink1_window127"] == 16384 and b["b128_sink1_window119_lowpass"] == 16263
    assert TAUS == (8, 16, 32, 64, 128, 256, 512) and FAR_START == 48


def test_scoring_and_questions_on_tiny_model():
    m = _tiny_model(D=8, H=2, V=13, P=128)
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 13, size=100) for _ in range(2)]
    res = score_methods(m, chunks, method_specs(), far_start=FAR_START, log=lambda *_: None)
    q = questions(res)
    assert res["full_cache"]["far_perplexity"] > 1.0
    assert q["primary_b48_lowpass_beats_sink_window"]["chunks"] == 2
    # nothing is evicted by a 127-token window on 100 tokens: equals the full cache
    assert abs(res["b128_sink1_window127"]["far_mean_nll"] - res["full_cache"]["far_mean_nll"]) < 1e-12
