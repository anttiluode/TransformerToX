"""Gate 3: sink control for the window + temporal-summary memory.

Pre-registered in docs/superpowers/specs/2026-09-24-sink-control-gate.md BEFORE
the first real score. Nothing is selected in this gate: the tau ladder and the
primary read (lowpass) were fixed from Gate 2, whose five contexts now serve as
the selection set. Scoring is on a FRESH five-context fixture that no gate has
seen, with the Gate 2 fixture reported alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.run_gpt2_shadow import (
    HEAD_INDEX,
    LAYER_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    STATE_BUDGET,
    _geometry_matches,
    _parity_summary,
    load_real_runtime,
    make_capture_context,
)
from experiments.run_gpt2_window_residue import load_long_contexts
from transformer_to_x.metrics import mean_cosine_similarity, relative_rmse
from transformer_to_x.pretrained_shadow import reconstruct_head
from transformer_to_x.window_residue import (
    WindowResidueMemory,
    ZeroMemory,
    rhos_from_taus,
    run_head_space_context,
)

FRESH_FIXTURE = Path(__file__).with_name("gpt2_long_contexts_fresh.json")

# ---- frozen before the first real score -------------------------------------
FAR_START = 48
SINK = 1
FROZEN_TAUS = (8, 16, 32, 64, 128, 256, 512)   # selected in Gate 2
PRIMARY_READ = "lowpass"                        # the better read in Gate 2
FIRST_K = 4                                     # "first few tokens" diagnostic
# -----------------------------------------------------------------------------


def load_fresh_contexts(path: Path = FRESH_FIXTURE) -> list[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("held_out") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items or len(set(items)) != len(items):
        raise ValueError("fresh fixture must hold a non-empty, duplicate-free held_out list")
    return list(items)


def method_factories(d_head: int, taus=FROZEN_TAUS, sink: int = SINK):
    rhos = rhos_from_taus(taus)
    full = STATE_BUDGET // (2 * d_head) if d_head == 64 else 12
    hyb = full - 8 if d_head == 64 else 8
    return {
        "zero": lambda: ZeroMemory(d_head=d_head),
        "kv_window_full": lambda: WindowResidueMemory(d_head=d_head, window=full),
        "sink_window_full": lambda: WindowResidueMemory(d_head=d_head, window=full - sink, sink=sink),
        "sink_window_hybrid_only": lambda: WindowResidueMemory(
            d_head=d_head, window=hyb - sink, sink=sink, rhos=rhos, read_mode="window_only"),
        "sink_hybrid_lowpass": lambda: WindowResidueMemory(
            d_head=d_head, window=hyb - sink, sink=sink, rhos=rhos, read_mode="lowpass"),
        "sink_hybrid_band": lambda: WindowResidueMemory(
            d_head=d_head, window=hyb - sink, sink=sink, rhos=rhos, read_mode="band"),
        "nosink_hybrid_lowpass": lambda: WindowResidueMemory(
            d_head=d_head, window=hyb, rhos=rhos, read_mode="lowpass"),
        "nosink_hybrid_band": lambda: WindowResidueMemory(
            d_head=d_head, window=hyb, rhos=rhos, read_mode="band"),
    }


def head_diagnostics(geometry, items, far_start: int, full_window: int, first_k: int = FIRST_K) -> dict:
    """What does the real head spend its attention on at far positions?"""
    a0, afirst, dropped_first, dropped_other = [], [], [], []
    tok0_share, without0_norm, v0_ratio = [], [], []
    for x, _ in items:
        ref = reconstruct_head(geometry, x)
        v = x @ geometry.w_v + geometry.b_v
        vo = v @ geometry.w_o
        vo_norm = np.linalg.norm(vo, axis=1)
        v0_ratio.append(vo_norm[0] / np.median(vo_norm[1:]))
        for t in range(far_start, x.shape[0]):
            att = ref.attention[t, : t + 1]
            y = ref.contribution[t]
            a0.append(att[0])
            afirst.append(att[:first_k].sum())
            lo = t + 1 - full_window                      # first index kept by the 48-window
            dropped_first.append(att[: min(first_k, lo)].sum())
            dropped_other.append(att[first_k:lo].sum() if lo > first_k else 0.0)
            term0 = att[0] * vo[0]
            yn = np.linalg.norm(y)
            tok0_share.append(np.linalg.norm(term0) / yn)
            without0_norm.append(np.linalg.norm(y - term0) / yn)
    m = lambda xs: float(np.mean(xs))
    return {
        "mean_attention_token0": m(a0),
        f"mean_attention_first{first_k}": m(afirst),
        f"mean_mass_dropped_by_window_on_first{first_k}": m(dropped_first),
        f"mean_mass_dropped_by_window_elsewhere": m(dropped_other),
        "token0_value_output_norm_over_median_other": m(v0_ratio),
        "mean_token0_term_norm_over_output_norm": m(tok0_share),
        "mean_output_norm_without_token0_over_output_norm": m(without0_norm),
    }


def score(geometry, items, factory, far_start: int) -> dict:
    refs, preds = [], []
    for x, ref in items:
        refs.append(ref[far_start:])
        preds.append(run_head_space_context(geometry, x, factory())[far_start:])
    R, P = np.concatenate(refs), np.concatenate(preds)
    rn, pn = np.linalg.norm(R, axis=1), np.linalg.norm(P, axis=1)
    return {
        "relative_rmse": relative_rmse(R, P),
        "mean_cosine_similarity": mean_cosine_similarity(R, P),
        "rms_norm_ratio_pred_over_target": float(np.sqrt(np.mean(P**2)) / np.sqrt(np.mean(R**2))),
        "mean_token_norm_ratio_pred_over_target": float(np.mean(pn / rn)),
        "per_context_relative_rmse": [relative_rmse(r, p) for r, p in zip(refs, preds, strict=True)],
        "tokens": int(R.shape[0]),
    }


def evaluate_fixture(geometry, items, factories, far_start: int, full_window: int) -> dict:
    methods = {name: score(geometry, items, f, far_start) for name, f in factories.items()}
    rm = {n: r["relative_rmse"] for n, r in methods.items()}
    per = {n: r["per_context_relative_rmse"] for n, r in methods.items()}
    wins = lambda a, b: int(sum(x < y for x, y in zip(per[a], per[b], strict=True)))

    def beats_both(name):
        return {
            "pass": bool(rm[name] < rm["sink_window_full"] and rm[name] < rm["zero"]),
            "relative_rmse": rm[name],
            "vs_sink_window_full": rm["sink_window_full"],
            "context_wins_vs_sink_window_full": wins(name, "sink_window_full"),
            "context_wins_vs_zero": wins(name, "zero"),
            "contexts": len(items),
        }

    return {
        "head_diagnostics": head_diagnostics(geometry, items, far_start, full_window),
        "methods": methods,
        "questions": {
            "sink_explains_gate2": {
                "pass": bool(rm["sink_window_full"] < rm["kv_window_full"] and rm["sink_window_full"] < 1.0),
                "sink_window_full": rm["sink_window_full"],
                "kv_window_full": rm["kv_window_full"],
                "context_wins": wins("sink_window_full", "kv_window_full"),
            },
            "primary_lowpass_beats_sink_window_and_zero": beats_both("sink_hybrid_lowpass"),
            "secondary_band_beats_sink_window_and_zero": beats_both("sink_hybrid_band"),
            "band_vs_lowpass_with_sink": {
                "band_better": bool(rm["sink_hybrid_band"] < rm["sink_hybrid_lowpass"]),
                "context_wins_band": wins("sink_hybrid_band", "sink_hybrid_lowpass"),
            },
        },
    }


def build_sink_receipt(fresh_texts, gate2_contexts, capture_context, runtime_metadata, *,
                       far_start: int = FAR_START, taus=FROZEN_TAUS, require_canonical_shape=True):
    gate2_texts = gate2_contexts["held_out"]
    fresh = [capture_context(t) for t in fresh_texts]
    old = [capture_context(t) for t in gate2_texts]
    geometry = fresh[0][0]
    if any(not _geometry_matches(geometry, c[0]) for c in fresh[1:] + old):
        raise ValueError("captured contexts do not share one frozen head geometry")
    d_head = geometry.d_head
    factories = method_factories(d_head, taus)
    budgets = {n: int(f().scalar_state_budget) for n, f in factories.items()}
    full_window = budgets["kv_window_full"] // (2 * d_head)
    if require_canonical_shape:
        if (geometry.d_model, d_head) != (768, 64):
            raise ValueError("canonical gate requires GPT-2 small head shapes")
        assert budgets["kv_window_full"] == budgets["sink_window_full"] == STATE_BUDGET
        assert budgets["sink_hybrid_lowpass"] == budgets["sink_hybrid_band"] == 6023
    for c in fresh + old:
        if c[4] <= far_start:
            raise ValueError("context too short for far metric")

    parity = _parity_summary([c[3] for c in fresh + old])
    receipt = {
        "gate": "gpt2_sink_control",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX, "head_index": HEAD_INDEX,
        "d_model": geometry.d_model, "d_head": d_head,
        "budgets": budgets,
        "frozen": {"far_start": far_start, "sink": SINK, "taus": list(taus),
                   "primary_read": PRIMARY_READ, "first_k": FIRST_K},
        "runtime": dict(runtime_metadata),
        "adapter_parity": parity,
        "fresh_contexts": list(fresh_texts),
        "fresh_token_counts": [int(c[4]) for c in fresh],
        "gate2_token_counts": [int(c[4]) for c in old],
        "boundary": {"removes_kv_cache": False, "changes_generation": False,
                     "measures_perplexity": False, "replaces_live_attention": False},
    }
    if parity["classification"] != "PASS":
        receipt["classification"] = "FAIL_ADAPTER_PARITY"
        return receipt
    receipt["fresh"] = evaluate_fixture(geometry, [(c[1], c[2]) for c in fresh], factories, far_start, full_window)
    receipt["gate2_fixture"] = evaluate_fixture(geometry, [(c[1], c[2]) for c in old], factories, far_start, full_window)
    primary = receipt["fresh"]["questions"]["primary_lowpass_beats_sink_window_and_zero"]["pass"]
    receipt["classification"] = (
        "PASS_TEMPORAL_SUMMARY_BEATS_SINK_WINDOW_AND_ZERO" if primary
        else "FAIL_TEMPORAL_SUMMARY_NOT_USEFUL_OVER_SINK_WINDOW"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gpt2_sink_control.json"))
    args = parser.parse_args()
    model, tokenizer, metadata = load_real_runtime()
    receipt = build_sink_receipt(load_fresh_contexts(), load_long_contexts(),
                                 make_capture_context(model, tokenizer), metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {receipt['classification']}")
    if receipt["classification"] == "FAIL_ADAPTER_PARITY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
