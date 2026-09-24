# GPT-2 Real-Head Shadow Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a fixed-size resident memory can predict the bias-free output contribution of one frozen real GPT-2 attention head while GPT-2 itself runs unchanged with explicit history.

**Architecture:** Normalize one selected GPT-2 head into a NumPy `PretrainedHeadGeometry`, keep Hugging Face-specific extraction in `gpt2_adapter.py`, and keep resident-memory evaluation in model-independent `pretrained_shadow.py`. The canonical runner freezes layer 5 / head 0 and the context set before scoring, proves adapter parity first, selects decay/resonant settings on selection contexts only, and reports held-out metrics under a matched 6,144-scalar state budget.

**Tech Stack:** Python 3.11+, NumPy 2.4.6 core, pytest; optional PyTorch and Hugging Face Transformers for the real-model run.

**Spec:** `docs/superpowers/specs/2026-09-24-gpt2-shadow-real-head-design.md`

## Global Constraints

- Core dependency remains exactly `numpy==2.4.6`; PyTorch/Transformers stay optional.
- CI never downloads GPT-2 and must pass without PyTorch/Transformers installed.
- Canonical model: `openai-community/gpt2`, eval mode, zero-based layer 5 / head 0.
- GPT-2 causal attention includes the current token, so resident evaluation is always `update(x_t)` then `read(address_t)`.
- Canonical state budget: 6,144 scalars = 8 resident vectors × 768 model dimensions.
- A resident memory may not retain raw token, hidden-state, K/V, attention, or model-cache history.
- Single-head target excludes shared `c_proj.bias`.
- Adapter parity is a prerequisite; resident scores are invalid when parity fails.
- No generation modification, KV-cache removal, perplexity claim, or Rytmi mechanism in this gate.

## Review Focus

- GPT-2 Conv1D packed-weight orientation must be pinned by a non-symmetric fixture.
- Unsupported inverse-layer attention scaling must fail loudly.
- `c_proj.bias` must appear exactly once in full-module parity and never in a single-head contribution.
- Token 0 must prove inclusive update-then-read ordering.
- Empty/non-finite/mismatched captured tensors and invalid head indices must fail before metrics.

---

### Task 1: Build model-independent real-head geometry and resident shadow evaluation

**Files:**
- Create: `src/transformer_to_x/pretrained_shadow.py`
- Create: `tests/test_pretrained_shadow.py`

**Interfaces:**
- `PretrainedHeadGeometry(d_model, d_head, w_q, w_k, w_v, b_q, b_k, b_v, w_o, score_scale)`
- `HeadReference(attention, mixture, contribution)`
- `ShadowMetrics(relative_rmse, mean_cosine_similarity, max_abs_error, token_relative_rmse, token_cosine_similarity)`
- `ShadowRun(prediction, metrics, state_budget)`
- `reconstruct_head(geometry, x) -> HeadReference`
- `resident_address_query(geometry, x_t) -> np.ndarray`
- `resident_write_operator(geometry) -> np.ndarray`
- `resident_value_offset(geometry) -> np.ndarray`
- `run_shadow_context(geometry, x, reference_y, memory) -> ShadowRun`
- `evaluate_contexts(geometry, contexts, memory_factory) -> dict`
- `select_temporal_config(geometry, configs, selection_contexts, memory_factory_for_config) -> tuple`

- [ ] **Step 1: Write RED tests for exact algebra, K-bias cancellation, V-bias offset, inclusive ordering, and validation**

```python
import dataclasses
import numpy as np
import pytest

from transformer_to_x.pretrained_shadow import (
    PretrainedHeadGeometry,
    reconstruct_head,
    resident_address_query,
    resident_value_offset,
    run_shadow_context,
)


def tiny_geometry():
    return PretrainedHeadGeometry(
        d_model=4,
        d_head=2,
        w_q=np.array([[1,2],[3,5],[7,11],[13,17]], float) / 17,
        w_k=np.array([[2,1],[5,3],[11,7],[17,13]], float) / 19,
        w_v=np.array([[1,0],[0,2],[3,1],[1,4]], float) / 5,
        b_q=np.array([0.2, -0.1]),
        b_k=np.array([0.7, -0.4]),
        b_v=np.array([0.3, -0.2]),
        w_o=np.array([[1,2,0,-1],[0.5,-1,2,1]], float),
        score_scale=1 / np.sqrt(2.0),
    )


def test_key_bias_cancels_from_attention():
    x = np.arange(12, dtype=float).reshape(3, 4) / 10
    g = tiny_geometry()
    zero_bk = dataclasses.replace(g, b_k=np.zeros(2))
    assert np.allclose(reconstruct_head(g, x).attention,
                       reconstruct_head(zero_bk, x).attention)


def test_value_bias_is_fixed_output_offset():
    g = tiny_geometry()
    assert np.allclose(resident_value_offset(g), g.b_v @ g.w_o)


class SpyMemory:
    scalar_state_budget = 4
    def __init__(self):
        self.calls = []
        self.state = np.zeros(4)
    def reset(self):
        self.calls.append("reset")
        self.state[:] = 0
    def update(self, x_t):
        self.calls.append("update")
        self.state = x_t.copy()
    def read(self, query):
        self.calls.append("read")
        return self.state.copy()


def test_shadow_is_inclusive_update_then_read():
    eye = np.eye(4)
    g = PretrainedHeadGeometry(
        4, 4, eye, eye, eye,
        np.zeros(4), np.zeros(4), np.zeros(4), eye, 1.0,
    )
    memory = SpyMemory()
    run_shadow_context(g, eye[:2], eye[:2], memory)
    assert memory.calls == ["reset", "update", "read", "update", "read"]
```

Also test wrong matrix shapes, non-finite values, non-positive `score_scale`, empty sequence, wrong hidden width, wrong `reference_y` shape, non-finite reference values, invalid/missing `scalar_state_budget`, and wrong/non-finite memory-read vectors.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_pretrained_shadow.py -q`

Expected: collection fails because `transformer_to_x.pretrained_shadow` is absent.

- [ ] **Step 3: Implement geometry and exact reference algebra**

```python
@dataclass(frozen=True)
class PretrainedHeadGeometry:
    d_model: int
    d_head: int
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    b_q: np.ndarray
    b_k: np.ndarray
    b_v: np.ndarray
    w_o: np.ndarray
    score_scale: float

    def __post_init__(self):
        d_model = int(self.d_model)
        d_head = int(self.d_head)
        if d_model <= 0 or d_head <= 0:
            raise ValueError("dimensions must be positive")
        object.__setattr__(self, "d_model", d_model)
        object.__setattr__(self, "d_head", d_head)
        shapes = {
            "w_q": (d_model, d_head), "w_k": (d_model, d_head),
            "w_v": (d_model, d_head), "w_o": (d_head, d_model),
            "b_q": (d_head,), "b_k": (d_head,), "b_v": (d_head,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value.copy())
        scale = float(self.score_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("score_scale must be positive and finite")
        object.__setattr__(self, "score_scale", scale)


def reconstruct_head(geometry, x):
    tokens = _validate_tokens(x, geometry.d_model)
    q = tokens @ geometry.w_q + geometry.b_q
    k = tokens @ geometry.w_k + geometry.b_k
    v = tokens @ geometry.w_v + geometry.b_v
    logits = (q @ k.T) * geometry.score_scale
    logits = logits.copy()
    logits[causal_mask(tokens.shape[0])] = -np.inf
    attention = stable_softmax(logits)
    mixture = attention @ v
    return HeadReference(attention, mixture, mixture @ geometry.w_o)


def resident_address_query(geometry, x_t):
    x_t = np.asarray(x_t, dtype=float)
    if x_t.shape != (geometry.d_model,) or not np.all(np.isfinite(x_t)):
        raise ValueError("x_t must be a finite d_model vector")
    return (x_t @ geometry.w_q + geometry.b_q) @ geometry.w_k.T


def resident_write_operator(geometry):
    return geometry.w_v @ geometry.w_o


def resident_value_offset(geometry):
    return geometry.b_v @ geometry.w_o
```

- [ ] **Step 4: Implement fixed-state shadow metrics and update-then-read loop**

```python
def run_shadow_context(geometry, x, reference_y, memory):
    tokens = _validate_tokens(x, geometry.d_model)
    reference = np.asarray(reference_y, dtype=float)
    if reference.shape != (tokens.shape[0], geometry.d_model) or not np.all(np.isfinite(reference)):
        raise ValueError("invalid reference_y")
    budget = getattr(memory, "scalar_state_budget", None)
    if not isinstance(budget, (int, np.integer)) or int(budget) <= 0:
        raise ValueError("memory must expose a positive integer scalar_state_budget")
    memory.reset()
    write = resident_write_operator(geometry)
    offset = resident_value_offset(geometry)
    predictions = []
    for x_t in tokens:
        address = resident_address_query(geometry, x_t)
        memory.update(x_t)
        resident = np.asarray(memory.read(address), dtype=float)
        if resident.shape != (geometry.d_model,) or not np.all(np.isfinite(resident)):
            raise ValueError("memory.read must return a finite d_model vector")
        predictions.append(resident @ write + offset)
    prediction = np.asarray(predictions)
    metrics = ShadowMetrics(
        relative_rmse(reference, prediction),
        mean_cosine_similarity(reference, prediction),
        max_abs_error(reference, prediction),
        tuple(relative_rmse(r, p) for r, p in zip(reference, prediction)),
        tuple(mean_cosine_similarity(r, p) for r, p in zip(reference, prediction)),
    )
    return ShadowRun(prediction, metrics, int(budget))
```

- [ ] **Step 5: Freeze candidate grids and selection logic**

```python
DECAY_CONFIGS = (
    (0.0, 0.25, 0.50, 0.65, 0.75, 0.82, 0.88, 0.92),
    (0.0, 0.50, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875),
    (0.75, 0.85, 0.90, 0.94, 0.96, 0.975, 0.985, 0.995),
)
RESONANT_CONFIGS = (
    ((0.90,0.20),(0.94,0.40),(0.97,0.70),(0.985,1.00)),
    ((0.90,0.35),(0.94,0.70),(0.97,1.05),(0.985,1.40)),
    ((0.90,0.50),(0.94,1.00),(0.97,1.50),(0.985,2.00)),
)


def evaluate_contexts(geometry, contexts, memory_factory):
    if not contexts:
        raise ValueError("at least one context is required")
    runs = [run_shadow_context(geometry, x, y, memory_factory()) for x, y in contexts]
    reference = np.concatenate([y for _, y in contexts], axis=0)
    prediction = np.concatenate([run.prediction for run in runs], axis=0)
    return {
        "relative_rmse": relative_rmse(reference, prediction),
        "mean_cosine_similarity": mean_cosine_similarity(reference, prediction),
        "max_abs_error": max_abs_error(reference, prediction),
        "runs": runs,
    }


def select_temporal_config(geometry, configs, selection_contexts, memory_factory_for_config):
    scored = []
    for index, config in enumerate(configs):
        metrics = evaluate_contexts(
            geometry,
            selection_contexts,
            lambda config=config: memory_factory_for_config(config),
        )
        scored.append((metrics["relative_rmse"], index, config))
    return min(scored, key=lambda item: (item[0], item[1]))[2]
```

Test that static 8 slots, every 8-decay configuration, and every 4-complex-mode resonant configuration have equal budgets; at `d_model=768` each must equal 6,144. Add a selection-leakage regression where one config wins selection but another would win held-out; the selected config must remain unchanged.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_pretrained_shadow.py tests/test_trace_bank.py tests/test_compiler.py -q`

Expected: PASS.

```bash
git add src/transformer_to_x/pretrained_shadow.py tests/test_pretrained_shadow.py
git commit -m "feat: add pretrained shadow evaluator"
```

---

### Task 2: Add GPT-2 packed projection extraction and adapter parity

**Files:**
- Create: `src/transformer_to_x/gpt2_adapter.py`
- Create: `tests/test_gpt2_adapter.py`

**Interfaces:**
- `slice_gpt2_head(c_attn_weight, c_attn_bias, c_proj_weight, *, n_head, head_index, scale_attn_weights, scale_attn_by_inverse_layer_idx, layer_index) -> PretrainedHeadGeometry`
- `GPT2Capture(x, attention, pre_cproj, cproj_output)`
- `GPT2Parity(max_attention_abs_error, max_mixture_abs_error, max_full_module_abs_error, passed)`
- `capture_gpt2_context(model, input_ids, *, layer_index, head_index, tolerance=1e-5)`

- [ ] **Step 1: Write RED packed-slice and shared-bias tests**

For a non-symmetric fixture with `d_model=6`, `n_head=3`, `d_head=2`, head 1 must use Q columns `2:4`, K columns `8:10`, V columns `14:16`, and `c_proj` rows `2:4`. Also synthesize two head mixtures and prove:

```python
pre = np.concatenate([m0, m1], axis=-1)
full = pre @ c_proj_weight + c_proj_bias
head0 = m0 @ c_proj_weight[:d_head, :]
head1 = m1 @ c_proj_weight[d_head:, :]
assert np.allclose(full, head0 + head1 + c_proj_bias)
```

Changing `c_proj_bias` must not alter `head0` or `head1`.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gpt2_adapter.py -q`

Expected: collection fails because `gpt2_adapter` is absent.

- [ ] **Step 3: Implement pure NumPy GPT-2 head slicing**

```python
def slice_gpt2_head(c_attn_weight, c_attn_bias, c_proj_weight, *, n_head,
                    head_index, scale_attn_weights=True,
                    scale_attn_by_inverse_layer_idx=False, layer_index=0):
    packed_w = np.asarray(c_attn_weight, dtype=float)
    packed_b = np.asarray(c_attn_bias, dtype=float)
    proj_w = np.asarray(c_proj_weight, dtype=float)
    if packed_w.ndim != 2:
        raise ValueError("c_attn_weight must be 2D")
    d_model = packed_w.shape[0]
    if packed_w.shape != (d_model, 3*d_model):
        raise ValueError("invalid c_attn_weight shape")
    if packed_b.shape != (3*d_model,) or proj_w.shape != (d_model, d_model):
        raise ValueError("invalid GPT-2 projection shape")
    if d_model % int(n_head) != 0:
        raise ValueError("d_model must be divisible by n_head")
    if not 0 <= int(head_index) < int(n_head):
        raise ValueError("head_index out of range")
    if scale_attn_by_inverse_layer_idx:
        raise ValueError("inverse-layer attention scaling is outside this gate")
    if not all(np.all(np.isfinite(v)) for v in (packed_w, packed_b, proj_w)):
        raise ValueError("projection arrays must be finite")
    d_head = d_model // int(n_head)
    start = int(head_index) * d_head
    stop = start + d_head
    q = slice(start, stop)
    k = slice(d_model + start, d_model + stop)
    v = slice(2*d_model + start, 2*d_model + stop)
    return PretrainedHeadGeometry(
        d_model, d_head,
        packed_w[:, q], packed_w[:, k], packed_w[:, v],
        packed_b[q], packed_b[k], packed_b[v],
        proj_w[start:stop, :],
        1 / np.sqrt(float(d_head)) if scale_attn_weights else 1.0,
    )
```

- [ ] **Step 4: Implement hook-based real capture and parity**

`capture_gpt2_context` must pre-hook `block.attn` for the real layer-normalized `x`, pre-hook `block.attn.c_proj` for concatenated head mixtures, and forward-hook `c_proj` for projected output. It must run with `output_attentions=True`, `use_cache=False`, `return_dict=True`, remove hooks in `finally`, then compute:

```python
reference = reconstruct_head(geometry, capture.x)
start = head_index * geometry.d_head
stop = start + geometry.d_head
attention_error = max_abs_error(capture.attention, reference.attention)
mixture_error = max_abs_error(capture.pre_cproj[:, start:stop], reference.mixture)
full_error = max_abs_error(
    capture.cproj_output,
    capture.pre_cproj @ c_proj_weight + c_proj_bias,
)
parity = GPT2Parity(
    attention_error,
    mixture_error,
    full_error,
    max(attention_error, mixture_error, full_error) <= tolerance,
)
```

Keep `torch` imported only inside the capture function. Validate batch size 1, returned attentions, model width, head index, capture shapes, and finite arrays before parity.

- [ ] **Step 5: Add failure tests**

Pin invalid head index, non-divisible width, malformed capture shape, missing attentions, and inverse-layer scaling. Use fake objects/pure helpers so CI needs no optional dependency.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_gpt2_adapter.py tests/test_pretrained_shadow.py -q`

Expected: PASS without model download.

```bash
git add src/transformer_to_x/gpt2_adapter.py tests/test_gpt2_adapter.py
git commit -m "feat: add GPT-2 head adapter"
```

---

### Task 3: Freeze contexts and add the canonical optional real-model runner

**Files:**
- Create: `experiments/gpt2_shadow_contexts.json`
- Create: `experiments/run_gpt2_shadow.py`
- Create: `tests/test_gpt2_shadow_runner.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Constants: `MODEL_ID="openai-community/gpt2"`, `LAYER_INDEX=5`, `HEAD_INDEX=0`, `STATE_BUDGET=6144`, `PARITY_TOLERANCE=1e-5`.
- `load_context_fixture(path) -> dict[str, list[str]]`
- `build_shadow_receipt(contexts, capture_context, runtime_metadata, *, canonical=False) -> dict`
- CLI: `python -m experiments.run_gpt2_shadow --out <path>`

- [ ] **Step 1: Freeze the context fixture before scoring**

Use exactly:

```json
{
  "selection": [
    "The small brass key was hidden beneath the blue notebook, and Mara checked the desk twice before leaving.",
    "After the rain stopped, the children returned to the courtyard and found their chalk drawings still visible.",
    "Jonas told the mechanic that the engine clicked only after the car had been running for several minutes."
  ],
  "held_out": [
    "The museum closed at six, but the guide stayed behind to answer questions about the old navigation instruments.",
    "On Tuesday, Lina mailed the red envelope to her brother and kept the green one for the meeting on Friday.",
    "The dog ignored the empty bowl until someone opened the cupboard where its food was stored.",
    "At the station, the northbound train arrived first, even though the display had predicted the southbound train.",
    "The programmer changed one line, reran the test, and discovered that the earlier failure had moved to a different module.",
    "When Elena returned to the kitchen, the kettle was quiet, the window was open, and the cup she had left on the table was gone."
  ]
}
```

Test non-empty unique strings and disjoint splits.

- [ ] **Step 2: Add optional runtime dependencies**

```toml
[project.optional-dependencies]
test = ["pytest>=8"]
pretrained = ["torch>=2.3", "transformers>=4.45,<5"]
```

- [ ] **Step 3: Write RED runner tests with injected fake captures**

Use `d_model=4` fake geometry so CI stays cheap. `build_shadow_receipt(..., canonical=False)` must produce method metrics and boundaries. Separately assert canonical constants, including `STATE_BUDGET == 6144`. Test that parity failure omits resident comparison fields, and that held-out evaluation never reselects configs.

- [ ] **Step 4: Implement real runtime loading**

```python
def load_real_runtime():
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            'Install pretrained extras with: python -m pip install -e ".[pretrained]"'
        ) from exc
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", MODEL_ID),
    }
    return model, tokenizer, metadata
```

Tokenize each context independently, with no padding or truncation, and wrap `capture_gpt2_context` in a closure returning `(geometry, x, reference_y, parity, token_count)`.

- [ ] **Step 5: Implement receipt algorithm explicitly**

`build_shadow_receipt` must:

1. validate selection/held-out fixture;
2. capture each context once;
3. require one consistent geometry across captures;
4. aggregate adapter maxima and stop resident scoring on any parity failure;
5. select one decay and one resonant configuration using selection contexts only;
6. evaluate static, selected decay, and selected resonant memories on held-out contexts;
7. report aggregate and per-context metrics, exact budget, selected configs, and relative-RMSE deltas vs static;
8. when `canonical=True`, require `d_model==768`, layer/head constants, and budget 6,144.

Classification is exactly `FAIL_ADAPTER_PARITY` on failed parity and `PASS_ADAPTER_PARITY_SHADOW_MEASURED` after a successful held-out measurement. Boundary booleans are always `removes_kv_cache=false`, `changes_generation=false`, `measures_perplexity=false`.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest -q`

Expected: all offline tests PASS without downloading GPT-2.

```bash
git add experiments/gpt2_shadow_contexts.json experiments/run_gpt2_shadow.py tests/test_gpt2_shadow_runner.py pyproject.toml
git commit -m "feat: add canonical GPT-2 shadow runner"
```

---

### Task 4: Run the real GPT-2 experiment and freeze evidence

**Files:**
- Create: `results/gpt2_shadow.json`
- Create: `tests/test_gpt2_shadow_receipt.py`

- [ ] **Step 1: Install optional runtime on a machine with model-download access**

```bash
python -m pip install -e ".[test,pretrained]"
```

- [ ] **Step 2: Run the frozen experiment once**

```bash
python -m experiments.run_gpt2_shadow --out /tmp/gpt2_shadow.json
```

Parity must pass before any resident result is accepted. If parity fails, fix adapter correctness only; do not change head, contexts, budget, or candidate grids.

- [ ] **Step 3: Accept only a correct adapter run**

```python
assert receipt["classification"] == "PASS_ADAPTER_PARITY_SHADOW_MEASURED"
assert receipt["adapter_parity"]["max_attention_abs_error"] <= 1e-5
assert receipt["adapter_parity"]["max_mixture_abs_error"] <= 1e-5
assert receipt["adapter_parity"]["max_full_module_abs_error"] <= 1e-5
assert receipt["state_budget"] == 6144
```

- [ ] **Step 4: Freeze the exact receipt and add a no-download regression test**

Copy `/tmp/gpt2_shadow.json` unchanged to `results/gpt2_shadow.json`. The regression test must assert model/head/budget, parity PASS, classification, and all three false boundary booleans; it must not rerun GPT-2.

- [ ] **Step 5: Run verification and commit**

Run: `pytest tests/test_gpt2_shadow_receipt.py -q && pytest -q`

Expected: PASS.

```bash
git add results/gpt2_shadow.json tests/test_gpt2_shadow_receipt.py
git commit -m "data: freeze GPT-2 real-head shadow receipt"
```

---

### Task 5: Report the result, preserve Gate 0, and publish

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-gpt2-shadow-real-head-design.md`

- [ ] **Step 1: Update README from frozen evidence only**

Report model/head, adapter parity maxima, static/decay/resonant held-out relative RMSE and cosine, selected configs, 6,144-scalar budget, and descriptive deltas versus static. State explicitly that GPT-2 was unchanged and KV cache/generation/perplexity were not tested.

Include:

```bash
python -m pip install -e ".[test,pretrained]"
python -m experiments.run_gpt2_shadow --out /tmp/gpt2_shadow.json
```

- [ ] **Step 2: Mark spec implemented**

Change `Status: approved` to `Status: implemented` and add `Frozen receipt: results/gpt2_shadow.json` immediately below it. Do not rewrite the design around whichever memory scored best.

- [ ] **Step 3: Verify all old and new evidence**

```bash
pytest -q
python -m experiments.run_gate0 --out /tmp/gate0.json
python -m experiments.verify_receipts /tmp/gate0.json results/gate0.json
```

Expected: PASS; Gate 0 frozen evidence remains unchanged.

- [ ] **Step 4: Scope review**

Only the new adapter/shadow/runner/tests/context/receipt files, `pyproject.toml`, README, approved spec, and this plan may differ from `main`. No live-generation patch or Rytmi implementation enters this branch.

- [ ] **Step 5: Commit, open PR, and merge only on green CI**

```bash
git add README.md docs/superpowers/specs/2026-09-24-gpt2-shadow-real-head-design.md
git commit -m "docs: report GPT-2 shadow gate"
```

PR summary must separate adapter parity, resident shadow metrics, and the explicit non-claim about KV-cache replacement. Merge only after Python 3.11/3.12 CI, Gate 0 reproduction, and frozen shadow receipt tests pass.
