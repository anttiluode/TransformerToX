from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from transformer_to_x.compiler import CompiledHead
from transformer_to_x.metrics import (
    max_abs_error,
    mean_cosine_similarity,
    relative_rmse,
)
from transformer_to_x.tiny_head import TinyCausalHead

WEIGHT_SEED = 20260924
D_MODEL = 12
D_HEAD = 6
SEQUENCE_LENGTHS = (4, 8, 16, 24)
EVALUATION_SEEDS = (11, 23, 47, 89, 173, 347, 691, 1381)
ATOL = 1e-12


def build_gate0_receipt() -> dict[str, object]:
    head = TinyCausalHead.deterministic(
        seed=WEIGHT_SEED,
        d_model=D_MODEL,
        d_head=D_HEAD,
    )
    compiled = CompiledHead.from_weights(head.weights)

    by_length: list[dict[str, object]] = []
    all_reference_outputs: list[np.ndarray] = []
    all_compiled_outputs: list[np.ndarray] = []
    max_logit_error = 0.0
    max_attention_error = 0.0
    max_output_error = 0.0

    for length in SEQUENCE_LENGTHS:
        length_logit_error = 0.0
        length_attention_error = 0.0
        length_output_error = 0.0
        length_reference_outputs: list[np.ndarray] = []
        length_compiled_outputs: list[np.ndarray] = []
        for seed in EVALUATION_SEEDS:
            tokens = np.random.default_rng(seed + 10000 * length).normal(
                size=(length, D_MODEL)
            )
            reference = head.reference(tokens)
            got = compiled.run(tokens)
            finite_logits = np.isfinite(reference.logits)
            if not np.array_equal(
                np.isneginf(reference.logits),
                np.isneginf(got.logits),
            ):
                raise AssertionError("compiled and reference causal masks differ")

            logit_error = max_abs_error(
                reference.logits[finite_logits],
                got.logits[finite_logits],
            )
            attention_error = max_abs_error(reference.attention, got.attention)
            output_error = max_abs_error(reference.output, got.output)
            length_logit_error = max(length_logit_error, logit_error)
            length_attention_error = max(length_attention_error, attention_error)
            length_output_error = max(length_output_error, output_error)
            length_reference_outputs.append(reference.output)
            length_compiled_outputs.append(got.output)

        ref_length = np.concatenate(length_reference_outputs, axis=0)
        got_length = np.concatenate(length_compiled_outputs, axis=0)
        by_length.append(
            {
                "sequence_length": int(length),
                "case_count": len(EVALUATION_SEEDS),
                "max_logit_abs_error": length_logit_error,
                "max_attention_abs_error": length_attention_error,
                "max_output_abs_error": length_output_error,
                "output_relative_rmse": relative_rmse(ref_length, got_length),
                "mean_output_cosine_similarity": mean_cosine_similarity(
                    ref_length,
                    got_length,
                ),
            }
        )
        max_logit_error = max(max_logit_error, length_logit_error)
        max_attention_error = max(max_attention_error, length_attention_error)
        max_output_error = max(max_output_error, length_output_error)
        all_reference_outputs.append(ref_length)
        all_compiled_outputs.append(got_length)

    reference_outputs = np.concatenate(all_reference_outputs, axis=0)
    compiled_outputs = np.concatenate(all_compiled_outputs, axis=0)
    aggregate_rel_rmse = relative_rmse(reference_outputs, compiled_outputs)
    aggregate_cosine = mean_cosine_similarity(reference_outputs, compiled_outputs)
    passed = (
        max_logit_error < ATOL
        and max_attention_error < ATOL
        and max_output_error < ATOL
        and aggregate_rel_rmse < ATOL
    )

    return {
        "gate": "gate0_exact_address_write_compiler",
        "classification": (
            "PASS_EXACT_COMPILER" if passed else "FAIL_EXACT_COMPILER"
        ),
        "d_model": D_MODEL,
        "d_head": D_HEAD,
        "weight_seed": WEIGHT_SEED,
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "case_count": len(SEQUENCE_LENGTHS) * len(EVALUATION_SEEDS),
        "tolerance": ATOL,
        "learned_parameter_count": 0,
        "address_operator_shape": list(compiled.address_operator.shape),
        "write_operator_shape": list(compiled.write_operator.shape),
        "max_logit_abs_error": max_logit_error,
        "max_attention_abs_error": max_attention_error,
        "max_output_abs_error": max_output_error,
        "output_relative_rmse": aggregate_rel_rmse,
        "mean_output_cosine_similarity": aggregate_cosine,
        "by_sequence_length": by_length,
        "claim": (
            "For this frozen deterministic head, M=W_Q W_K^T and "
            "N=W_V W_O re-express ordinary causal attention with zero "
            "fitted parameters to floating-point tolerance."
        ),
        "boundary": (
            "Gate 0 is algebraic validation only. It does not compress KV "
            "history, replace softmax attention, establish an LLM "
            "architecture advantage, or make a biological claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    receipt = build_gate0_receipt()
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
