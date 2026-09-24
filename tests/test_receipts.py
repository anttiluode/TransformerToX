import importlib
import json
from pathlib import Path

import pytest

from experiments.verify_receipts import assert_receipts_close


def _gate0_builder():
    try:
        module = importlib.import_module("experiments.run_gate0")
    except ModuleNotFoundError:
        pytest.fail("Gate 0 experiment is not implemented yet")
    assert hasattr(module, "build_gate0_receipt")
    return module.build_gate0_receipt


def test_gate0_is_exact_zero_fit_compilation():
    receipt = _gate0_builder()()
    assert receipt["classification"] == "PASS_EXACT_COMPILER"
    assert receipt["learned_parameter_count"] == 0
    assert receipt["max_logit_abs_error"] < 1e-12
    assert receipt["max_attention_abs_error"] < 1e-12
    assert receipt["max_output_abs_error"] < 1e-12
    assert receipt["output_relative_rmse"] < 1e-12


def test_gate0_canonical_receipt_matches_frozen_file_with_declared_fp_tolerance():
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "results" / "gate0.json").read_text())
    generated = _gate0_builder()()
    assert_receipts_close(generated, frozen, abs_tolerance=1e-12)
