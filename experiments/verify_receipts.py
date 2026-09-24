from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ABS_TOLERANCE = 1e-12


def assert_receipts_close(
    actual: Any,
    expected: Any,
    *,
    abs_tolerance: float = DEFAULT_ABS_TOLERANCE,
    path: str = "root",
) -> None:
    """Compare receipt structure exactly and floating values within tolerance.

    Frozen receipts are scientific records, but final-bit floating-point values can
    differ across BLAS kernels/runner CPUs. Strings, keys, integers, sequence order,
    and classifications remain exact; only JSON floating values get a declared
    absolute tolerance.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected dict, got {type(actual).__name__}")
        if set(actual) != set(expected):
            raise AssertionError(
                f"{path}: keys differ: actual={sorted(actual)} expected={sorted(expected)}"
            )
        for key in expected:
            assert_receipts_close(
                actual[key],
                expected[key],
                abs_tolerance=abs_tolerance,
                path=f"{path}.{key}",
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list shape differs")
        for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
            assert_receipts_close(
                got,
                want,
                abs_tolerance=abs_tolerance,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, float):
        try:
            value = float(actual)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{path}: expected floating value") from exc
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=abs_tolerance):
            raise AssertionError(
                f"{path}: {value!r} != {expected!r} within abs_tol={abs_tolerance}"
            )
        return

    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def verify_pair(
    generated_path: Path,
    frozen_path: Path,
    *,
    abs_tolerance: float = DEFAULT_ABS_TOLERANCE,
) -> None:
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert_receipts_close(generated, frozen, abs_tolerance=abs_tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("--abs-tolerance", type=float, default=DEFAULT_ABS_TOLERANCE)
    args = parser.parse_args()
    verify_pair(args.generated, args.frozen, abs_tolerance=args.abs_tolerance)
    print(f"receipt matches within abs_tol={args.abs_tolerance:g}: {args.frozen}")


if __name__ == "__main__":
    main()
