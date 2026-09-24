from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from transformer_to_x.gpt2_adapter import GPT2Parity, capture_gpt2_context
from transformer_to_x.pretrained_shadow import (
    DECAY_CONFIGS,
    RESONANT_CONFIGS,
    PretrainedHeadGeometry,
    evaluate_contexts,
    reconstruct_head,
    select_temporal_config,
)
from transformer_to_x.trace_bank import (
    DecayMemory,
    ResonantMemory,
    RytmiWindowMemory,
    StaticMemory,
    assert_equal_state_budget,
)

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
LAYER_INDEX = 5
HEAD_INDEX = 0
STATE_BUDGET = 6144
PARITY_TOLERANCE = 1e-5
COMPONENT_VECTORS = 8
CONTEXT_FIXTURE = Path(__file__).with_name("gpt2_shadow_contexts.json")

# Frozen before the first Rytmi-window real-head score. Each candidate stores
# four fast/slow pairs = eight d_model vectors, exactly matching the 6,144-scalar
# GPT-2 budget. Selection uses only the existing selection contexts and primary
# relative RMSE, exactly like the earlier temporal candidates.
RYTMI_WINDOW_CONFIGS = (
    ((0.00, 0.50), (0.25, 0.75), (0.50, 0.875), (0.75, 0.96875)),
    ((0.25, 0.75), (0.50, 0.875), (0.75, 0.96875), (0.875, 0.9921875)),
    ((0.50, 0.85), (0.75, 0.94), (0.85, 0.975), (0.90, 0.995)),
)


def _validate_context_dict(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict) or set(value) != {"selection", "held_out"}:
        raise ValueError("context fixture must contain exactly selection and held_out")
    result: dict[str, list[str]] = {}
    for split in ("selection", "held_out"):
        items = value[split]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{split} must be a non-empty list")
        if not all(isinstance(item, str) and item.strip() for item in items):
            raise ValueError(f"{split} must contain non-empty strings")
        if len(set(items)) != len(items):
            raise ValueError(f"{split} contains duplicate contexts")
        result[split] = list(items)
    if set(result["selection"]) & set(result["held_out"]):
        raise ValueError("selection and held_out contexts must be disjoint")
    return result


def load_context_fixture(path: Path = CONTEXT_FIXTURE) -> dict[str, list[str]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load context fixture: {path}") from exc
    return _validate_context_dict(raw)


def _geometry_matches(a: PretrainedHeadGeometry, b: PretrainedHeadGeometry) -> bool:
    if (a.d_model, a.d_head, a.score_scale) != (b.d_model, b.d_head, b.score_scale):
        return False
    for name in ("w_q", "w_k", "w_v", "b_q", "b_k", "b_v", "w_o"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            return False
    return True


def _parity_summary(parities: list[GPT2Parity]) -> dict[str, object]:
    if not parities:
        raise ValueError("at least one parity result is required")
    max_attention = max(p.max_attention_abs_error for p in parities)
    max_mixture = max(p.max_mixture_abs_error for p in parities)
    max_full = max(p.max_full_module_abs_error for p in parities)
    passed = all(p.passed for p in parities) and max(max_attention, max_mixture, max_full) <= PARITY_TOLERANCE
    return {
        "classification": "PASS" if passed else "FAIL",
        "tolerance": PARITY_TOLERANCE,
        "max_attention_abs_error": float(max_attention),
        "max_mixture_abs_error": float(max_mixture),
        "max_full_module_abs_error": float(max_full),
    }


def _metrics_dict(metrics) -> dict[str, object]:
    return {
        "relative_rmse": float(metrics.relative_rmse),
        "mean_cosine_similarity": float(metrics.mean_cosine_similarity),
        "max_abs_error": float(metrics.max_abs_error),
        "token_relative_rmse": [float(x) for x in metrics.token_relative_rmse],
        "token_cosine_similarity": [float(x) for x in metrics.token_cosine_similarity],
    }


def _evaluate_method(
    geometry: PretrainedHeadGeometry,
    held_texts: list[str],
    held_captures: list[tuple[np.ndarray, np.ndarray, int]],
    memory_factory: Callable[[], object],
) -> dict[str, object]:
    contexts = [(x, reference_y) for x, reference_y, _ in held_captures]
    result = evaluate_contexts(geometry, contexts, memory_factory)
    per_context = []
    for text, (_, _, token_count), run in zip(held_texts, held_captures, result["runs"], strict=True):
        per_context.append({
            "text": text,
            "token_count": int(token_count),
            **_metrics_dict(run.metrics),
        })
    return {
        "relative_rmse": float(result["relative_rmse"]),
        "mean_cosine_similarity": float(result["mean_cosine_similarity"]),
        "max_abs_error": float(result["max_abs_error"]),
        "per_context": per_context,
    }


def build_shadow_receipt(
    contexts: dict[str, list[str]],
    capture_context: Callable[[str], tuple[PretrainedHeadGeometry, np.ndarray, np.ndarray, GPT2Parity, int]],
    runtime_metadata: dict[str, object],
    *,
    require_canonical_shape: bool = True,
) -> dict[str, object]:
    contexts = _validate_context_dict(contexts)
    all_texts = contexts["selection"] + contexts["held_out"]
    captured = [capture_context(text) for text in all_texts]
    if not captured:
        raise ValueError("no contexts captured")
    geometry = captured[0][0]
    if any(not _geometry_matches(geometry, item[0]) for item in captured[1:]):
        raise ValueError("captured contexts do not share one frozen head geometry")
    if require_canonical_shape and geometry.d_model != 768:
        raise ValueError("canonical GPT-2 shadow gate requires d_model=768")

    parities = [item[3] for item in captured]
    parity = _parity_summary(parities)
    state_budget = COMPONENT_VECTORS * geometry.d_model
    if require_canonical_shape and state_budget != STATE_BUDGET:
        raise ValueError("canonical GPT-2 shadow gate requires state budget 6144")

    boundary = {
        "removes_kv_cache": False,
        "changes_generation": False,
        "measures_perplexity": False,
        "replaces_live_attention": False,
    }
    receipt: dict[str, object] = {
        "gate": "gpt2_real_head_shadow",
        "classification": (
            "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
            if parity["classification"] == "PASS"
            else "FAIL_ADAPTER_PARITY"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "head_index": HEAD_INDEX,
        "d_model": geometry.d_model,
        "d_head": geometry.d_head,
        "state_budget": state_budget,
        "component_vectors": COMPONENT_VECTORS,
        "runtime": dict(runtime_metadata),
        "adapter_parity": parity,
        "contexts": contexts,
        "boundary": boundary,
    }
    if parity["classification"] != "PASS":
        return receipt

    n_selection = len(contexts["selection"])
    selection_items = captured[:n_selection]
    held_items = captured[n_selection:]
    selection_pairs = [(item[1], item[2]) for item in selection_items]
    held_captures = [(item[1], item[2], item[4]) for item in held_items]

    selected_decay = select_temporal_config(
        DECAY_CONFIGS,
        (geometry, selection_pairs),
        lambda config: DecayMemory(d_model=geometry.d_model, rhos=config),
    )
    selected_resonant = select_temporal_config(
        RESONANT_CONFIGS,
        (geometry, selection_pairs),
        lambda config: ResonantMemory(d_model=geometry.d_model, modes=config),
    )
    selected_rytmi = select_temporal_config(
        RYTMI_WINDOW_CONFIGS,
        (geometry, selection_pairs),
        lambda config: RytmiWindowMemory(d_model=geometry.d_model, pairs=config),
    )

    representative_memories = {
        "static": StaticMemory(d_model=geometry.d_model, state_slots=COMPONENT_VECTORS),
        "decay": DecayMemory(d_model=geometry.d_model, rhos=selected_decay),
        "resonant": ResonantMemory(d_model=geometry.d_model, modes=selected_resonant),
        "rytmi_window": RytmiWindowMemory(d_model=geometry.d_model, pairs=selected_rytmi),
        "rytmi_raw_fast_slow": RytmiWindowMemory(
            d_model=geometry.d_model,
            pairs=selected_rytmi,
            read_mode="raw_fast_slow",
        ),
        "rytmi_slow_only": RytmiWindowMemory(
            d_model=geometry.d_model,
            pairs=selected_rytmi,
            read_mode="slow_only",
        ),
    }
    if assert_equal_state_budget(representative_memories.values()) != state_budget:
        raise AssertionError("resident state budget mismatch")

    methods = {
        "static": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: StaticMemory(d_model=geometry.d_model, state_slots=COMPONENT_VECTORS),
        ),
        "decay": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: DecayMemory(d_model=geometry.d_model, rhos=selected_decay),
        ),
        "resonant": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: ResonantMemory(d_model=geometry.d_model, modes=selected_resonant),
        ),
        "rytmi_window": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: RytmiWindowMemory(d_model=geometry.d_model, pairs=selected_rytmi),
        ),
        "rytmi_raw_fast_slow": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: RytmiWindowMemory(
                d_model=geometry.d_model,
                pairs=selected_rytmi,
                read_mode="raw_fast_slow",
            ),
        ),
        "rytmi_slow_only": _evaluate_method(
            geometry,
            contexts["held_out"],
            held_captures,
            lambda: RytmiWindowMemory(
                d_model=geometry.d_model,
                pairs=selected_rytmi,
                read_mode="slow_only",
            ),
        ),
    }
    static_rrmse = float(methods["static"]["relative_rmse"])
    resonant_rrmse = float(methods["resonant"]["relative_rmse"])
    resonant_cosine = float(methods["resonant"]["mean_cosine_similarity"])
    deltas_static = {
        name: float(value["relative_rmse"] - static_rrmse)
        for name, value in methods.items()
    }
    deltas_resonant = {
        name: float(value["relative_rmse"] - resonant_rrmse)
        for name, value in methods.items()
    }
    cosine_deltas_resonant = {
        name: float(value["mean_cosine_similarity"] - resonant_cosine)
        for name, value in methods.items()
    }
    best_method = min(methods, key=lambda name: methods[name]["relative_rmse"])
    receipt["selection"] = {
        "selected_decay_config": list(selected_decay),
        "selected_resonant_config": [list(mode) for mode in selected_resonant],
        "selected_rytmi_window_config": [list(pair) for pair in selected_rytmi],
        "rytmi_selection_metric": "relative_rmse",
    }
    receipt["held_out"] = {
        "methods": methods,
        "best_held_out_method": best_method,
        "relative_rmse_delta_vs_static": deltas_static,
        "relative_rmse_delta_vs_resonant": deltas_resonant,
        "cosine_delta_vs_resonant": cosine_deltas_resonant,
    }
    return receipt


def load_real_runtime():
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            'Install pretrained extras with: python -m pip install -e ".[pretrained]"'
        ) from exc

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", MODEL_ID),
    }
    if metadata["model_revision"] not in (None, MODEL_REVISION):
        raise ValueError("loaded GPT-2 revision does not match the frozen model revision")
    return model, tokenizer, metadata


def make_capture_context(model, tokenizer):
    def capture(text: str):
        encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        geometry, captured, parity = capture_gpt2_context(
            model,
            encoded["input_ids"],
            layer_index=LAYER_INDEX,
            head_index=HEAD_INDEX,
            tolerance=PARITY_TOLERANCE,
        )
        reference_y = reconstruct_head(geometry, captured.x).contribution
        return geometry, captured.x, reference_y, parity, int(captured.x.shape[0])
    return capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gpt2_shadow.json"))
    args = parser.parse_args()
    contexts = load_context_fixture()
    model, tokenizer, metadata = load_real_runtime()
    receipt = build_shadow_receipt(
        contexts,
        make_capture_context(model, tokenizer),
        metadata,
        require_canonical_shape=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {receipt['classification']}")
    if receipt["classification"] == "FAIL_ADAPTER_PARITY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
