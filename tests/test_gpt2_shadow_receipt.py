import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_gpt2_shadow_receipt_preserves_measured_result_and_boundary():
    receipt = json.loads((ROOT / "results" / "gpt2_shadow.json").read_text())
    assert receipt["classification"] == "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
    assert receipt["model_id"] == "openai-community/gpt2"
    assert receipt["model_revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert receipt["layer_index"] == 5
    assert receipt["head_index"] == 0
    assert receipt["state_budget"] == 6144
    assert receipt["adapter_parity"]["classification"] == "PASS"
    assert receipt["held_out"]["best_held_out_method"] == "resonant"
    assert receipt["boundary"]["removes_kv_cache"] is False
    assert receipt["boundary"]["changes_generation"] is False
    assert receipt["boundary"]["measures_perplexity"] is False
    assert receipt["boundary"]["replaces_live_attention"] is False
    assert receipt["source_artifact"]["artifact_sha256"] == (
        "aaeb790d6a5a7a5f37cccddabf208177f55a86f4c72e27edd699e480409c3e95"
    )


def test_frozen_rytmi_shadow_receipt_preserves_direction_gain_and_ablation():
    receipt = json.loads((ROOT / "results" / "gpt2_shadow_rytmi.json").read_text())
    methods = receipt["held_out"]["methods"]

    assert receipt["classification"] == "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
    assert receipt["model_id"] == "openai-community/gpt2"
    assert receipt["model_revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert receipt["layer_index"] == 5
    assert receipt["head_index"] == 0
    assert receipt["state_budget"] == 6144
    assert receipt["adapter_parity"]["classification"] == "PASS"
    assert receipt["held_out"]["best_held_out_method"] == "rytmi_window"
    assert receipt["selection"]["selected_rytmi_window_config"] == [
        [0.0, 0.5],
        [0.25, 0.75],
        [0.5, 0.875],
        [0.75, 0.96875],
    ]

    assert methods["resonant"]["relative_rmse"] == 2.4242373438756686
    assert methods["resonant"]["mean_cosine_similarity"] == 0.18877317503706997
    assert methods["rytmi_window"]["relative_rmse"] == 1.8595309831300948
    assert methods["rytmi_window"]["mean_cosine_similarity"] == 0.23771841453292217
    assert methods["rytmi_slow_only"]["relative_rmse"] == 9.443948750365779
    assert methods["rytmi_slow_only"]["mean_cosine_similarity"] == 0.16156587411201595

    assert receipt["held_out"]["relative_rmse_delta_vs_resonant"]["rytmi_window"] == (
        -0.5647063607455738
    )
    assert receipt["held_out"]["cosine_delta_vs_resonant"]["rytmi_window"] == (
        0.048945239495852194
    )
    assert receipt["boundary"]["removes_kv_cache"] is False
    assert receipt["boundary"]["changes_generation"] is False
    assert receipt["boundary"]["measures_perplexity"] is False
    assert receipt["boundary"]["replaces_live_attention"] is False
    assert receipt["source_artifact"]["artifact_sha256"] == (
        "37fd94feaf1615b08b6460c97c3c61513d168453948544efe5f5376543951fb9"
    )


def test_frozen_raw_fast_slow_receipt_preserves_coordinate_control():
    receipt = json.loads((ROOT / "results" / "gpt2_shadow_raw_fast_slow.json").read_text())
    methods = receipt["held_out"]["methods"]

    assert receipt["classification"] == "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
    assert receipt["model_id"] == "openai-community/gpt2"
    assert receipt["model_revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert receipt["layer_index"] == 5
    assert receipt["head_index"] == 0
    assert receipt["state_budget"] == 6144
    assert receipt["adapter_parity"]["classification"] == "PASS"
    assert receipt["held_out"]["best_held_out_method"] == "rytmi_window"

    direction = methods["rytmi_window"]
    raw = methods["rytmi_raw_fast_slow"]
    assert direction["relative_rmse"] == 1.8595309831300948
    assert direction["mean_cosine_similarity"] == 0.23771841453292217
    assert raw["relative_rmse"] == 6.5545560883860015
    assert raw["mean_cosine_similarity"] == 0.16863343800093566
    assert raw["max_abs_error"] == 20.300800742857636

    for direction_context, raw_context in zip(
        direction["per_context"], raw["per_context"], strict=True
    ):
        assert direction_context["text"] == raw_context["text"]
        assert direction_context["relative_rmse"] < raw_context["relative_rmse"]
        assert (
            direction_context["mean_cosine_similarity"]
            > raw_context["mean_cosine_similarity"]
        )

    assert receipt["boundary"]["removes_kv_cache"] is False
    assert receipt["boundary"]["changes_generation"] is False
    assert receipt["boundary"]["measures_perplexity"] is False
    assert receipt["boundary"]["replaces_live_attention"] is False
    assert receipt["source_artifact"]["artifact_sha256"] == (
        "0bbff02514f0dbb3ecd5c7ad89bede3384c49f1b25c1555ee0835ba9420e4c32"
    )
