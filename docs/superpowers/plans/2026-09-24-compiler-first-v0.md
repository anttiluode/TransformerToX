# TransformerToX Compiler-First v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze a zero-fit compiler that re-expresses one deterministic causal attention head as address geometry plus write geometry, then add a cheap matched-budget resident-memory experiment.

**Architecture:** A NumPy reference head computes ordinary causal attention. A compiler constructs `M = W_Q W_K^T` and `N = W_V W_O`, then reproduces logits, attention probabilities, and outputs directly from token states. A separate fixed-state memory API supports static, decaying, and resonant memories so Gate 1 can test whether explicit history can be compressed without changing Gate 0 semantics.

**Tech Stack:** Python 3.11/3.12, NumPy >=1.26,<3, pytest >=8, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-24-transformer-to-x-design.md`

## Global Constraints

- `d_model = 12`, `d_head = 6` for the canonical deterministic head.
- Canonical sequence lengths: 4, 8, 16, 24.
- Frozen weight seed: `20260924`.
- Gate 0 introduces zero fitted or learned parameters.
- Gate 1 compares methods at equal scalar state budget.
- Gate 1 candidate selection may use only a training split; all reported metrics use a held-out split.
- Gate 1 fixed-state implementations must not retain raw token history.
- Gate 2 and real pretrained adapters are out of scope until Gate 0 is exact; Gate 2 requires a separate follow-up if Gate 1 leaves a useful mechanism.
- Cheap canonical runs must fit ordinary GitHub Actions CPU CI.

## Review Focus

- Non-finite input or weight values must fail before softmax or recurrence arithmetic; Task 1 tests this explicitly.
- Shape-incompatible matrices must raise clear `ValueError`s instead of broadcasting; Tasks 1 and 2 test this explicitly.
- Causal masking must prevent future-token influence exactly; Task 1 tests a future-token perturbation.
- Equal-budget Gate 1 comparison must reject mismatched scalar budgets; Task 4 tests this explicitly.
- Fixed-state Gate 1 memories must expose no raw-history field or sequence cache; Task 4 tests bounded state shape after long input sequences.

---

### Task 1: Scaffold the package and deterministic reference head

**Files:**
- Create: `pyproject.toml`
- Create: `src/transformer_to_x/__init__.py`
- Create: `src/transformer_to_x/tiny_head.py`
- Create: `tests/test_tiny_head.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `HeadWeights`, `TinyCausalHead`, `stable_softmax`, `causal_mask`.
- `TinyCausalHead.reference(x: np.ndarray) -> HeadRun`, where `x` has shape `[seq, d_model]`.
- `HeadRun` contains `logits`, `attention`, and `output` arrays.

- [ ] **Step 1: Write failing reference-head tests**

```python
import numpy as np
import pytest
from transformer_to_x.tiny_head import TinyCausalHead


def test_reference_head_is_deterministic_and_causal():
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    x = np.random.default_rng(7).normal(size=(8, 12))
    a = head.reference(x)
    b = head.reference(x)
    np.testing.assert_array_equal(a.output, b.output)
    assert np.allclose(np.triu(a.attention, 1), 0.0)

    changed = x.copy()
    changed[-1] += 100.0
    before = head.reference(x).output[:-1]
    after = head.reference(changed).output[:-1]
    np.testing.assert_allclose(before, after, atol=0.0, rtol=0.0)


def test_reference_head_rejects_bad_shape_and_nonfinite_input():
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    with pytest.raises(ValueError):
        head.reference(np.zeros((4, 11)))
    bad = np.zeros((4, 12))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        head.reference(bad)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_tiny_head.py -q`
Expected: import/module failure because the package does not yet exist.

- [ ] **Step 3: Implement the minimal reference head**

`HeadWeights` stores `w_q: [d_model,d_head]`, `w_k: [d_model,d_head]`, `w_v: [d_model,d_head]`, `w_o: [d_head,d_model]`. `TinyCausalHead.deterministic(...)` samples all four matrices from `N(0, 1/sqrt(d_model))` using one NumPy RNG. `reference` validates shape/finiteness, computes `Q,K,V`, masks future logits with `-inf`, applies row-wise stable softmax, and returns `attention @ V @ W_O`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest tests/test_tiny_head.py -q`
Expected: all tests pass.

- [ ] **Step 5: Add package metadata and CI**

`pyproject.toml` must declare NumPy and pytest extras. CI runs `python -m pip install -e ".[test]"` and `pytest -q` on Python 3.11 and 3.12.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/transformer_to_x tests/test_tiny_head.py .github/workflows/ci.yml
git commit -m "feat: add deterministic causal attention head"
```

### Task 2: Implement the exact address/write compiler

**Files:**
- Create: `src/transformer_to_x/compiler.py`
- Create: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `HeadWeights`, `HeadRun` from Task 1.
- Produces: `CompiledHead.from_weights(weights)`, properties `address_operator`, `write_operator`, and `run(x) -> HeadRun`.

- [ ] **Step 1: Write failing exact-equivalence tests**

```python
import numpy as np
import pytest
from transformer_to_x.compiler import CompiledHead
from transformer_to_x.tiny_head import TinyCausalHead


def test_compiled_head_matches_reference_logits_attention_and_output():
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    compiled = CompiledHead.from_weights(head.weights)
    x = np.random.default_rng(99).normal(size=(16, 12))
    ref = head.reference(x)
    got = compiled.run(x)
    np.testing.assert_allclose(got.logits, ref.logits, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(got.attention, ref.attention, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(got.output, ref.output, atol=2e-14, rtol=2e-14)


def test_compiler_operators_have_expected_shapes():
    head = TinyCausalHead.deterministic(seed=20260924, d_model=12, d_head=6)
    compiled = CompiledHead.from_weights(head.weights)
    assert compiled.address_operator.shape == (12, 12)
    assert compiled.write_operator.shape == (12, 12)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_compiler.py -q`
Expected: module/import failure.

- [ ] **Step 3: Implement exact compilation**

Construct `M = W_Q @ W_K.T` and `N = W_V @ W_O`. For token matrix `X`, compute unmasked logits as `(X @ M @ X.T) / sqrt(d_head)`, apply the same causal mask/softmax helper as the reference path, and compute output as `attention @ (X @ N)`.

- [ ] **Step 4: Run focused tests and full tests**

Run: `pytest tests/test_compiler.py -q && pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/transformer_to_x/compiler.py tests/test_compiler.py
git commit -m "feat: compile attention into address and write operators"
```

### Task 3: Freeze Gate 0 as a reproducible scientific receipt

**Files:**
- Create: `src/transformer_to_x/metrics.py`
- Create: `experiments/__init__.py`
- Create: `experiments/run_gate0.py`
- Create: `results/gate0.json`
- Create: `tests/test_receipts.py`
- Create: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `build_gate0_receipt() -> dict[str, object]`.
- Metrics functions: `relative_rmse(reference, estimate)`, `max_abs_error(reference, estimate)`, `mean_cosine_similarity(reference, estimate)`.

- [ ] **Step 1: Write failing receipt regression test**

```python
from experiments.run_gate0 import build_gate0_receipt


def test_gate0_is_exact_zero_fit_compilation():
    receipt = build_gate0_receipt()
    assert receipt["classification"] == "PASS_EXACT_COMPILER"
    assert receipt["learned_parameter_count"] == 0
    assert receipt["max_logit_abs_error"] < 1e-12
    assert receipt["max_output_abs_error"] < 1e-12
    assert receipt["output_relative_rmse"] < 1e-12
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_receipts.py::test_gate0_is_exact_zero_fit_compilation -q`
Expected: import failure because `experiments.run_gate0` does not exist.

- [ ] **Step 3: Implement metrics and Gate 0 experiment**

Use sequence lengths `[4,8,16,24]` and frozen evaluation seeds `[11,23,47,89,173,347,691,1381]`. Record dimensions, all seeds, tolerances, per-sequence maxima, aggregate maxima, zero learned parameters, claim text, and boundary text stating that Gate 0 is algebraic validation only.

- [ ] **Step 4: Generate `results/gate0.json`**

Run: `python -m experiments.run_gate0 --out results/gate0.json`
Expected: JSON classification `PASS_EXACT_COMPILER`.

- [ ] **Step 5: Add exact frozen-receipt regression**

Extend `tests/test_receipts.py` to load `results/gate0.json` and assert `build_gate0_receipt() == frozen`.

- [ ] **Step 6: Document only the frozen claim**

README must explain the address/write translation, report Gate 0 numbers from the receipt, and explicitly say the scientific question starts at Gate 1.

- [ ] **Step 7: Add receipt reproduction to CI**

CI command:

```bash
python -m experiments.run_gate0 --out /tmp/gate0.json
cmp /tmp/gate0.json results/gate0.json
```

- [ ] **Step 8: Run full verification**

Run: `pytest -q && python -m experiments.run_gate0 --out /tmp/gate0.json && cmp /tmp/gate0.json results/gate0.json`
Expected: success.

- [ ] **Step 9: Commit**

```bash
git add src/transformer_to_x/metrics.py experiments results/gate0.json tests/test_receipts.py README.md .github/workflows/ci.yml
git commit -m "test: freeze exact compiler Gate 0"
```

### Task 4: Add fixed-state resident memory candidates and budget accounting

**Files:**
- Create: `src/transformer_to_x/trace_bank.py`
- Create: `tests/test_trace_bank.py`

**Interfaces:**
- `StaticMemory(d_model: int, state_slots: int)`.
- `DecayMemory(d_model: int, rhos: tuple[float,...])`.
- `ResonantMemory(d_model: int, modes: tuple[tuple[float,float],...])` where each mode is `(rho, omega)` and stores cosine/sine components.
- Every memory exposes `reset()`, `update(x_t)`, `read(query)`, `scalar_state_budget`, and `state_shape`.

- [ ] **Step 1: Write failing recurrence/budget/no-history tests**

```python
import numpy as np
import pytest
from transformer_to_x.trace_bank import DecayMemory, ResonantMemory


def test_decay_memory_has_fixed_state_independent_of_sequence_length():
    mem = DecayMemory(d_model=12, rhos=(0.25, 0.75))
    initial_shape = mem.state_shape
    for _ in range(100):
        mem.update(np.ones(12))
    assert mem.state_shape == initial_shape
    assert mem.scalar_state_budget == 24
    assert not hasattr(mem, "history")


def test_resonant_memory_counts_real_and_imaginary_state():
    mem = ResonantMemory(d_model=12, modes=((0.9, 0.4),))
    assert mem.scalar_state_budget == 24


def test_unstable_resonance_is_rejected():
    with pytest.raises(ValueError):
        ResonantMemory(d_model=12, modes=((1.01, 0.4),))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_trace_bank.py -q`
Expected: module/import failure.

- [ ] **Step 3: Implement minimal fixed-state recurrences**

Decay state uses `z_r <- rho_r z_r + x_t`. Resonant state keeps real/imaginary vectors and applies the damped 2D rotation for each mode before adding `x_t` to the real component. Static memory uses the same scalar budget and a non-temporal running projection/update rule declared in code; it must not retain token history.

- [ ] **Step 4: Add equal-budget guard**

Provide `assert_equal_state_budget(memories)` that raises `ValueError` if budgets differ. Add a test passing one 24-scalar decay bank and one 12-scalar static bank and assert rejection.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest tests/test_trace_bank.py -q && pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/transformer_to_x/trace_bank.py tests/test_trace_bank.py
git commit -m "feat: add fixed-state resident memory candidates"
```

### Task 5: Run cheap Gate 1 matched-budget compression experiment

**Files:**
- Create: `experiments/run_gate1.py`
- Create: `results/gate1.json`
- Modify: `tests/test_receipts.py`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `build_gate1_receipt() -> dict[str, object]`.
- Uses the exact frozen head from Gate 0 and compares static, decay, and resonant memories at equal scalar budgets.

- [ ] **Step 1: Write the failing Gate 1 receipt-shape test**

```python
from experiments.run_gate1 import build_gate1_receipt


def test_gate1_receipt_uses_heldout_matched_budget_comparison():
    receipt = build_gate1_receipt()
    budgets = {m["scalar_state_budget"] for m in receipt["methods"].values()}
    assert len(budgets) == 1
    assert receipt["selection_split_seed"] != receipt["heldout_split_seed"]
    assert receipt["classification"] in {
        "TEMPORAL_MEMORY_WINS",
        "STATIC_MEMORY_MATCHES_OR_WINS",
        "MIXED",
    }
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_receipts.py::test_gate1_receipt_uses_heldout_matched_budget_comparison -q`
Expected: import failure.

- [ ] **Step 3: Implement the canonical Gate 1 evaluator**

Use deterministic sequence generators with training/selection seed `3101` and held-out seed `9173`. Use a small declared candidate grid: decay `rho` values from `{0.25,0.5,0.75,0.9}` and resonant modes from `rho in {0.75,0.9}` × `omega in {0.25,0.5,1.0,1.5}` radians. Select the best candidate only on the selection split. Freeze it and evaluate held-out sequence lengths `[8,16,24]` using relative RMSE, max absolute error, and cosine similarity. Use one common scalar state budget chosen so all three methods have exact parity.

- [ ] **Step 4: Preserve the result even if negative**

Classification rule:

```python
if temporal_best_rel_rmse < 0.95 * static_rel_rmse:
    classification = "TEMPORAL_MEMORY_WINS"
elif static_rel_rmse <= temporal_best_rel_rmse:
    classification = "STATIC_MEMORY_MATCHES_OR_WINS"
else:
    classification = "MIXED"
```

Receipt boundary must say this toy head/sequence family does not establish an LLM replacement.

- [ ] **Step 5: Generate and freeze `results/gate1.json`**

Run: `python -m experiments.run_gate1 --out results/gate1.json`
Expected: one of the three declared classifications; do not tune after seeing held-out results.

- [ ] **Step 6: Extend exact receipt regression and CI**

Add equality against frozen `results/gate1.json`. CI reproduces both gate receipts and compares them byte-for-byte.

- [ ] **Step 7: Update README from the actual result**

If temporal memory wins, describe only the held-out matched-budget advantage. If not, explicitly record the negative result and stop short of Gate 2.

- [ ] **Step 8: Run full verification**

Run: `pytest -q && python -m experiments.run_gate0 --out /tmp/gate0.json && cmp /tmp/gate0.json results/gate0.json && python -m experiments.run_gate1 --out /tmp/gate1.json && cmp /tmp/gate1.json results/gate1.json`
Expected: success.

- [ ] **Step 9: Commit**

```bash
git add experiments/run_gate1.py results/gate1.json tests/test_receipts.py README.md .github/workflows/ci.yml
git commit -m "test: freeze matched-budget resident-memory Gate 1"
```

### Task 6: Integration verification and handoff

**Files:**
- Review all files changed by Tasks 1–5.

**Interfaces:**
- No new interfaces. This task verifies that Gate 0 remains exact and Gate 1 remains bounded by its declared frozen experiment.

- [ ] **Step 1: Run the complete project verification from a clean checkout**

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_gate0 --out /tmp/gate0.json
cmp /tmp/gate0.json results/gate0.json
python -m experiments.run_gate1 --out /tmp/gate1.json
cmp /tmp/gate1.json results/gate1.json
```

Expected: exit code 0 throughout.

- [ ] **Step 2: Inspect the final diff for claim drift**

Check that README claims correspond exactly to `results/gate0.json` and `results/gate1.json`; verify no Gate 2 or biological claim was added.

- [ ] **Step 3: Open a pull request**

PR summary must state the exact Gate 0 numerical error, Gate 1 classification and matched budget, frozen seeds, and that pretrained extraction is the next milestone rather than part of v0.

- [ ] **Step 4: Merge only after the final branch CI matrix is green**

Use squash merge and preserve the frozen results on `main`.
