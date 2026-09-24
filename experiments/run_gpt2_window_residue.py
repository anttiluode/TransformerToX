"""Gate: exact KV window + temporal residue stack vs an equal-budget KV window.

Pre-registered in docs/superpowers/specs/2026-09-24-window-residue-gate.md
BEFORE the first real GPT-2 score. Same frozen model, revision, layer and head
as every earlier shadow gate; new long contexts (~250 tokens), because the
original one-sentence contexts are at most 29 tokens and an equal-budget KV
window (48 tokens) reproduces them exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.run_gpt2_shadow import (
    HEAD_INDEX,
    LAYER_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    PARITY_TOLERANCE,
    STATE_BUDGET,
    _geometry_matches,
    _parity_summary,
    _validate_context_dict,
    load_context_fixture,
    load_real_runtime,
    make_capture_context,
)
from transformer_to_x.metrics import mean_cosine_similarity, relative_rmse
from transformer_to_x.pretrained_shadow import run_shadow_context
from transformer_to_x.trace_bank import RytmiWindowMemory
from transformer_to_x.window_residue import (
    WindowResidueMemory,
    ZeroMemory,
    rhos_from_taus,
    run_head_space_context,
)

LONG_CONTEXT_FIXTURE = Path(__file__).with_name("gpt2_long_contexts.json")

# ---- frozen before the first real score -------------------------------------
FULL_WINDOW = 48          # 48 tokens * (64 + 64) = 6144 scalars = STATE_BUDGET
HYBRID_WINDOW = 40        # 5120 scalars
FAR_START = 48            # primary metric: positions where the 48-window is inexact
TAU_LADDERS = (           # seven octave levels each: 7 * 129 = 903 scalars
    (2, 4, 8, 16, 32, 64, 128),
    (4, 8, 16, 32, 64, 128, 256),
    (8, 16, 32, 64, 128, 256, 512),
)
# The Rytmi config selected in PR #3 (results/gpt2_shadow_rytmi.json), reused unchanged.
FROZEN_RYTMI_PAIRS = ((0.0, 0.5), (0.25, 0.75), (0.5, 0.875), (0.75, 0.96875))
# -----------------------------------------------------------------------------


def load_long_contexts(path: Path = LONG_CONTEXT_FIXTURE) -> dict[str, list[str]]:
    return _validate_context_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _stats(refs: list[np.ndarray], preds: list[np.ndarray], start: int) -> dict[str, float]:
    ref = np.concatenate([r[start:] for r in refs])
    pred = np.concatenate([p[start:] for p in preds])
    return {
        "relative_rmse": relative_rmse(ref, pred),
        "mean_cosine_similarity": mean_cosine_similarity(ref, pred),
        "tokens": int(ref.shape[0]),
    }


def _score(geometry, items, predict: Callable[[np.ndarray], np.ndarray], far_start: int) -> dict[str, object]:
    refs = [ref for _, ref in items]
    preds = [predict(x) for x, _ in items]
    return {
        "far": _stats(refs, preds, far_start),
        "all": _stats(refs, preds, 0),
        "per_context_far_relative_rmse": [
            relative_rmse(r[far_start:], p[far_start:]) for r, p in zip(refs, preds, strict=True)
        ],
    }


def _head_space(geometry, factory):
    return lambda x: run_head_space_context(geometry, x, factory())


def build_window_residue_receipt(
    long_contexts: dict[str, list[str]],
    short_contexts: dict[str, list[str]],
    capture_context,
    runtime_metadata: dict[str, object],
    *,
    full_window: int = FULL_WINDOW,
    hybrid_window: int = HYBRID_WINDOW,
    far_start: int = FAR_START,
    tau_ladders=TAU_LADDERS,
    require_canonical_shape: bool = True,
) -> dict[str, object]:
    long_contexts = _validate_context_dict(long_contexts)
    short_contexts = _validate_context_dict(short_contexts)
    long_texts = long_contexts["selection"] + long_contexts["held_out"]
    captured = [capture_context(text) for text in long_texts]
    short_captured = [capture_context(text) for text in short_contexts["held_out"]]
    geometry = captured[0][0]
    if any(not _geometry_matches(geometry, item[0]) for item in captured[1:] + short_captured):
        raise ValueError("captured contexts do not share one frozen head geometry")

    d_head = geometry.d_head
    budgets = {
        "kv_window_full": WindowResidueMemory(d_head=d_head, window=full_window).scalar_state_budget,
        "hybrid": WindowResidueMemory(
            d_head=d_head, window=hybrid_window, rhos=rhos_from_taus(tau_ladders[0])
        ).scalar_state_budget,
    }
    if require_canonical_shape:
        if geometry.d_model != 768 or d_head != 64:
            raise ValueError("canonical gate requires GPT-2 small head shapes")
        if budgets["kv_window_full"] != STATE_BUDGET or budgets["hybrid"] > STATE_BUDGET:
            raise AssertionError("budget mismatch")
    for text, item in zip(long_texts, captured, strict=True):
        if item[4] <= far_start:
            raise ValueError(f"long context too short for far metric: {item[4]} tokens")

    parity = _parity_summary([item[3] for item in captured + short_captured])
    receipt: dict[str, object] = {
        "gate": "gpt2_window_residue",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "head_index": HEAD_INDEX,
        "d_model": geometry.d_model,
        "d_head": d_head,
        "state_budget": STATE_BUDGET,
        "budgets": budgets,
        "frozen": {
            "full_window": full_window,
            "hybrid_window": hybrid_window,
            "far_start": far_start,
            "tau_ladders": [list(t) for t in tau_ladders],
            "rytmi_pairs": [list(p) for p in FROZEN_RYTMI_PAIRS],
        },
        "runtime": dict(runtime_metadata),
        "adapter_parity": parity,
        "contexts": long_contexts,
        "long_token_counts": [int(item[4]) for item in captured],
        "boundary": {
            "removes_kv_cache": False,
            "changes_generation": False,
            "measures_perplexity": False,
            "replaces_live_attention": False,
        },
    }
    if parity["classification"] != "PASS":
        receipt["classification"] = "FAIL_ADAPTER_PARITY"
        return receipt

    # 1. Sanity row for every earlier table: the equal-budget KV window on the
    #    ORIGINAL short held-out contexts.
    short_items = [(item[1], item[2]) for item in short_captured]
    short_scores = _score(
        geometry, short_items,
        _head_space(geometry, lambda: WindowResidueMemory(d_head=d_head, window=full_window)), 0,
    )
    receipt["short_context_check"] = {
        "max_token_count": int(max(item[4] for item in short_captured)),
        "kv_window_full_relative_rmse": short_scores["all"]["relative_rmse"],
        "note": "equal-budget KV window on the original one-sentence held-out contexts",
    }

    n_sel = len(long_contexts["selection"])
    selection = [(item[1], item[2]) for item in captured[:n_sel]]
    held = [(item[1], item[2]) for item in captured[n_sel:]]

    # 2. Select the tau ladder for the BAND read on selection contexts only.
    ladder_scores = []
    for index, taus in enumerate(tau_ladders):
        rhos = rhos_from_taus(taus)
        s = _score(geometry, selection, _head_space(
            geometry, lambda rhos=rhos: WindowResidueMemory(
                d_head=d_head, window=hybrid_window, rhos=rhos, read_mode="band")), far_start)
        ladder_scores.append((s["far"]["relative_rmse"], index))
    selected = tau_ladders[min(ladder_scores)[1]]
    rhos = rhos_from_taus(selected)
    receipt["selection"] = {
        "selected_taus": list(selected),
        "selection_far_relative_rmse": [float(v) for v, _ in sorted(ladder_scores, key=lambda t: t[1])],
    }

    # 3. Held-out scoring.
    methods = {
        "zero": _head_space(geometry, lambda: ZeroMemory(d_head=d_head)),
        "kv_window_full": _head_space(geometry, lambda: WindowResidueMemory(d_head=d_head, window=full_window)),
        "kv_window_hybrid_only": _head_space(geometry, lambda: WindowResidueMemory(
            d_head=d_head, window=hybrid_window, rhos=rhos, read_mode="window_only")),
        "window_residue_band": _head_space(geometry, lambda: WindowResidueMemory(
            d_head=d_head, window=hybrid_window, rhos=rhos, read_mode="band")),
        "window_residue_lowpass": _head_space(geometry, lambda: WindowResidueMemory(
            d_head=d_head, window=hybrid_window, rhos=rhos, read_mode="lowpass")),
    }
    results = {name: _score(geometry, held, fn, far_start) for name, fn in methods.items()}

    def rytmi_predict(x):
        # residual-space resident memory from PR #3, same 6144-scalar budget
        dummy_ref = np.zeros((x.shape[0], geometry.d_model))
        return run_shadow_context(
            geometry, x, dummy_ref, RytmiWindowMemory(d_model=geometry.d_model, pairs=FROZEN_RYTMI_PAIRS)
        ).prediction
    results["rytmi_resident_pr3"] = _score(geometry, held, rytmi_predict, far_start)

    far = {name: r["far"]["relative_rmse"] for name, r in results.items()}
    per = {name: r["per_context_far_relative_rmse"] for name, r in results.items()}

    def wins(a, b):
        return int(sum(x < y for x, y in zip(per[a], per[b], strict=True)))

    n_held = len(held)
    primary = far["window_residue_band"] < far["kv_window_full"]
    receipt["held_out"] = {
        "methods": results,
        "questions": {
            "primary_band_beats_equal_budget_window": {
                "pass": bool(primary),
                "band_far": far["window_residue_band"],
                "window_full_far": far["kv_window_full"],
                "context_wins": wins("window_residue_band", "kv_window_full"),
                "contexts": n_held,
            },
            "residue_adds_over_its_own_window": {
                "pass": bool(far["window_residue_band"] < far["kv_window_hybrid_only"]),
                "context_wins": wins("window_residue_band", "kv_window_hybrid_only"),
            },
            "band_beats_lowpass_same_state": {
                "pass": bool(far["window_residue_band"] < far["window_residue_lowpass"]),
                "context_wins": wins("window_residue_band", "window_residue_lowpass"),
            },
        },
    }
    receipt["classification"] = (
        "PASS_RESIDUE_BEATS_EQUAL_BUDGET_WINDOW" if primary else "FAIL_WINDOW_WINS_AT_EQUAL_BUDGET"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gpt2_window_residue.json"))
    args = parser.parse_args()
    model, tokenizer, metadata = load_real_runtime()
    receipt = build_window_residue_receipt(
        load_long_contexts(),
        load_context_fixture(),
        make_capture_context(model, tokenizer),
        metadata,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {receipt['classification']}")
    if receipt["classification"] == "FAIL_ADAPTER_PARITY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
