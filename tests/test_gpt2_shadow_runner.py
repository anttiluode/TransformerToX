import json
from pathlib import Path

import numpy as np
import pytest

from experiments.run_gpt2_shadow import (
    HEAD_INDEX,
    LAYER_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    PARITY_TOLERANCE,
    RYTMI_WINDOW_CONFIGS,
    STATE_BUDGET,
    build_shadow_receipt,
    load_context_fixture,
)
from transformer_to_x.gpt2_adapter import GPT2Parity
from transformer_to_x.pretrained_shadow import PretrainedHeadGeometry, reconstruct_head


ROOT = Path(__file__).resolve().parents[1]


def identity_geometry(d_model=4):
    eye = np.eye(d_model)
    return PretrainedHeadGeometry(
        d_model=d_model,
        d_head=d_model,
        w_q=eye,
        w_k=eye,
        w_v=eye,
        b_q=np.zeros(d_model),
        b_k=np.zeros(d_model),
        b_v=np.zeros(d_model),
        w_o=eye,
        score_scale=1.0,
    )


def fake_capture_factory(*, parity_pass=True, d_model=4):
    geometry = identity_geometry(d_model)

    def capture(text):
        seed = sum(ord(ch) for ch in text) % 997
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(3 + seed % 3, d_model))
        reference_y = reconstruct_head(geometry, x).contribution
        value = 0.0 if parity_pass else 1e-2
        parity = GPT2Parity(value, value, value, parity_pass)
        return geometry, x, reference_y, parity, x.shape[0]

    return capture


def test_frozen_context_fixture_is_valid_and_disjoint():
    fixture = load_context_fixture(ROOT / "experiments" / "gpt2_shadow_contexts.json")
    assert fixture["selection"]
    assert fixture["held_out"]
    assert all(isinstance(x, str) and x.strip() for x in fixture["selection"] + fixture["held_out"])
    assert len(set(fixture["selection"] + fixture["held_out"])) == len(fixture["selection"] + fixture["held_out"])
    assert set(fixture["selection"]).isdisjoint(fixture["held_out"])


def test_canonical_constants_are_frozen_before_scoring():
    assert MODEL_ID == "openai-community/gpt2"
    assert MODEL_REVISION == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert LAYER_INDEX == 5
    assert HEAD_INDEX == 0
    assert STATE_BUDGET == 6144
    assert PARITY_TOLERANCE == 1e-5


def test_rytmi_window_grid_is_tiny_fixed_and_uses_four_fast_slow_pairs():
    assert len(RYTMI_WINDOW_CONFIGS) == 3
    for config in RYTMI_WINDOW_CONFIGS:
        assert len(config) == 4
        assert all(0.0 <= fast < slow < 1.0 for fast, slow in config)


def test_build_shadow_receipt_reports_parity_methods_rytmi_ablation_and_non_claims():
    contexts = {"selection": ["alpha", "beta"], "held_out": ["gamma", "delta"]}
    receipt = build_shadow_receipt(
        contexts,
        fake_capture_factory(),
        {"torch_version": "fake", "transformers_version": "fake"},
        require_canonical_shape=False,
    )
    assert receipt["gate"] == "gpt2_real_head_shadow"
    assert receipt["classification"] == "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
    assert receipt["model_id"] == MODEL_ID
    assert receipt["model_revision"] == MODEL_REVISION
    assert receipt["layer_index"] == LAYER_INDEX
    assert receipt["head_index"] == HEAD_INDEX
    assert receipt["state_budget"] == 8 * 4
    assert receipt["adapter_parity"]["classification"] == "PASS"
    assert set(receipt["held_out"]["methods"]) == {
        "static",
        "decay",
        "resonant",
        "rytmi_window",
        "rytmi_raw_fast_slow",
        "rytmi_slow_only",
    }
    assert "selected_decay_config" in receipt["selection"]
    assert "selected_resonant_config" in receipt["selection"]
    selected_rytmi = receipt["selection"]["selected_rytmi_window_config"]
    assert len(selected_rytmi) == 4
    assert receipt["held_out"]["relative_rmse_delta_vs_resonant"]["resonant"] == 0.0
    assert receipt["held_out"]["cosine_delta_vs_resonant"]["resonant"] == 0.0
    assert receipt["boundary"]["removes_kv_cache"] is False
    assert receipt["boundary"]["changes_generation"] is False
    assert receipt["boundary"]["measures_perplexity"] is False


def test_parity_failure_short_circuits_resident_scoring():
    contexts = {"selection": ["alpha"], "held_out": ["gamma"]}
    receipt = build_shadow_receipt(
        contexts,
        fake_capture_factory(parity_pass=False),
        {},
        require_canonical_shape=False,
    )
    assert receipt["classification"] == "FAIL_ADAPTER_PARITY"
    assert receipt["adapter_parity"]["classification"] == "FAIL"
    assert "held_out" not in receipt
    assert "selection" not in receipt


def test_canonical_mode_rejects_non_gpt2_width_before_scoring():
    contexts = {"selection": ["alpha"], "held_out": ["gamma"]}
    with pytest.raises(ValueError, match="768"):
        build_shadow_receipt(
            contexts,
            fake_capture_factory(d_model=4),
            {},
            require_canonical_shape=True,
        )


def test_context_loader_rejects_empty_duplicate_and_overlapping_splits(tmp_path):
    cases = [
        {"selection": [], "held_out": ["x"]},
        {"selection": ["x", "x"], "held_out": ["y"]},
        {"selection": ["x"], "held_out": ["x"]},
        {"selection": [""], "held_out": ["y"]},
    ]
    for i, value in enumerate(cases):
        path = tmp_path / f"bad{i}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError):
            load_context_fixture(path)
