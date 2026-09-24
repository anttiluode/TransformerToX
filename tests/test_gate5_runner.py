import numpy as np

from experiments.run_gate5 import (
    MAX_CHUNKS_PER_TEXT,
    MODELS,
    chunk_ids,
    classify,
    load_texts,
    questions,
    row_specs,
    score,
)
from tests.test_gpt2_numpy import _tiny_model
from tests.test_neox_numpy import _tiny


def test_frozen_setup():
    assert MODELS["gpt2"][1] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert MODELS["pythia-160m"][1] == "step143000"
    assert set(load_texts()) == {"shakespeare", "python313_docs", "unseen_stories"}
    b = {k: s.scalar_budget(64) for k, s in row_specs().items()}
    assert b["b48_sink_window"] == b["b48_h2o"] == 6144
    assert b["b48_lowpass"] == 6023 and b["b48_merge"] == 6063
    assert b["b128_sink_window"] == b["b128_h2o"] == 16384
    assert b["b128_lowpass"] == 16263 and b["b128_merge"] == 16383
    assert MAX_CHUNKS_PER_TEXT == 6
    assert len(chunk_ids(np.arange(10_000))) == 6 and len(chunk_ids(np.arange(3_500))) == 3


def test_end_to_end_on_tiny_models():
    rng = np.random.default_rng(0)
    for m in (_tiny_model(D=8, H=2, V=13, P=200), _tiny(D=8, H=2, V=13)):
        chunks = {"a": [rng.integers(0, 13, size=150)], "b": [rng.integers(0, 13, size=150)] * 2}
        res = score(m, chunks, row_specs(), far_start=48, log=lambda *_: None)
        q = questions(res)
        assert q["b48_lowpass_vs_h2o"]["chunks"] == 3
        assert classify(q).startswith(("PASS_", "FAIL_"))
        # at 150 tokens nothing is evicted by the 128-budget rows except via the window
        assert res["full_cache"]["pooled_far_perplexity"] > 1
