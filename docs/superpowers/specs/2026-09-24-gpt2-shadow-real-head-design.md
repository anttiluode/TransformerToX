# GPT-2 shadow real-head design

Date: 2026-09-24
Status: implemented
Frozen receipt: `results/gpt2_shadow.json`
Raw run evidence: workflow `36015751802`, artifact `10814183911`, SHA-256 `aaeb790d6a5a7a5f37cccddabf208177f55a86f4c72e27edd699e480409c3e95`

## Purpose

TransformerToX has proved the exact address/write factorization on a deterministic synthetic head and has fixed-size static, decay, and resonant resident-memory primitives. The next question is whether those resident memories can predict what a real pretrained transformer attention head would have retrieved from its explicit token history.

This gate does **not** replace GPT-2 attention, does not alter generation, and does not claim KV-cache replacement. It runs a frozen pretrained model normally, observes one real attention head, and asks whether a fixed-size resident state can shadow that head's explicit-history output.

The experiment is intentionally narrow because a negative result is useful. If fixed resident state cannot approximate one real head even in shadow mode, there is no reason yet to patch live decoding or remove KV cache.

## Model and scope

The first supported model is `openai-community/gpt2` in evaluation mode.

The canonical experiment uses:

- one frozen GPT-2 model;
- **layer 5, head 0 (zero-based)**, fixed before any resident-memory scores are inspected;
- a small frozen set of English contexts committed with the experiment;
- no gradient updates;
- no generated continuation;
- one full forward pass per context to capture the reference trajectory;
- resident memories updated causally, one token event at a time;
- static, decay, and resonant memories compared under the same scalar state budget.

CI must not download GPT-2. Real-model execution is an explicit optional command that writes a machine-readable receipt. Unit tests cover all model-independent algebra and adapter contracts using tiny local fixtures.

## Reference quantity

For the selected GPT-2 head, let `x_t` be the actual layer-normalized residual entering the attention projection at token `t`.

The pretrained attention module computes packed Q/K/V projections, causal attention, one value mixture per head, concatenates the head mixtures, then applies the shared output projection `c_proj` and one shared output bias.

A single head does not uniquely own any fraction of that shared `c_proj` bias. Therefore the target for this gate is the selected head's **bias-free contribution through its slice of the shared output-projection weight**. Call this model-dimensional vector `y_t`.

Adapter parity is established in two steps before any resident score is trusted:

1. reconstructed attention probabilities for the selected head must match GPT-2's own returned attention probabilities within an explicit tolerance;
2. the adapter must reproduce the selected head's pre-`c_proj` value-mixture slice captured at the real module boundary, and the reconstruction of all head slices through `c_proj` plus the shared bias must reproduce the real attention-module output within tolerance.

This makes `y_t` an exact algebraic decomposition of a verified real module output rather than a quantity inferred from an unverified extraction.

## Pretrained head algebra

For one head, write the frozen affine projections as

- `q_t = x_t W_Q + b_Q`
- `k_i = x_i W_K + b_K`
- `v_i = x_i W_V + b_V`

and let `W_O` be the selected head's row slice of GPT-2's shared `c_proj` weight under the adapter's normalized orientation.

The selected head's bias-free model-dimensional contribution is

`y_t = sum_i alpha(t,i) (x_i W_V + b_V) W_O`

with causal softmax weights `alpha(t,i)` from the usual scaled query/key logits.

Two simplifications are part of the adapter contract:

1. `b_K` contributes the same scalar offset to every allowed logit for a fixed query, so it cancels inside softmax and need not appear in resident addressing.
2. Attention weights sum to one, so `b_V W_O` is a fixed selected-head output offset and can be added after the resident read.

The shared `c_proj` output bias is excluded from `y_t` because it belongs to the whole multi-head module, not to an individual head.

The address query used by resident memory is therefore

`a_t = (x_t W_Q + b_Q) W_K^T`

and the write operator remains

`N = W_V W_O`.

The resident prediction is

`x_hat_t = memory.read(a_t)`

`y_hat_t = x_hat_t N + b_V W_O`.

This does **not** assert equivalence to softmax attention. It defines the smallest shadow prediction consistent with the existing TransformerToX address/write semantics.

## Causal resident-state protocol

GPT-2 causal self-attention includes the current token position. The resident path must therefore use the same inclusive convention.

For each context, process tokens in order. At token `t`:

1. obtain the frozen model's real layer-normalized residual `x_t` and verified reference contribution `y_t`;
2. form the resident address query `a_t`;
3. update the resident memory **exactly once with `x_t`**;
4. read the now-inclusive resident state with `a_t`;
5. produce `y_hat_t`;
6. discard direct access to earlier raw events on the resident path.

The update-then-read ordering is part of the scientific contract and must have a regression test. No resident candidate may retain the raw token sequence, raw hidden-state history, K history, V history, attention matrices, or model cache.

For this first gate, the event written to resident memory is the same `x_t` representation used by the compiler semantics. More elaborate Rytmi-style phase/direction state is out of scope for this gate.

## Compared memories

The first real-head shadow comparison reuses the existing memory classes:

1. `StaticMemory` — deterministic order-invariant matched-budget attacker;
2. `DecayMemory` — fixed bank of exponentially decaying traces;
3. `ResonantMemory` — fixed bank of damped oscillatory traces.

All candidates use the same normalized pseudo-event readout already implemented in `trace_bank.py` and must expose exactly matched scalar state budgets.

The canonical first budget is **8 resident component-vectors = 8 × 768 = 6,144 scalar state values** for GPT-2 small. This means:

- static: 8 slots;
- decay: 8 decay traces;
- resonant: 4 complex modes represented by 8 real/imaginary component-vectors.

No new learned memory parameters are allowed in this first real-head gate. If candidate decay constants or resonant modes are selected from a tiny grid, selection must use only the declared selection contexts and freeze before held-out reporting. No gradient optimization is allowed.

## Metrics

Primary metrics are measured against the exact verified selected-head contribution `y_t`:

- aggregate relative RMSE;
- mean cosine similarity;
- maximum absolute error;
- error by token position;
- error by context length;
- exact scalar resident-state budget.

The receipt must separately report adapter-parity diagnostics:

- maximum selected-head attention-probability error;
- maximum pre-`c_proj` selected-head mixture error;
- maximum full attention-module reconstruction error after `c_proj` and shared bias.

These are prerequisites, not resident-memory scores.

No positive resident threshold is predeclared for the first shadow run beyond correctness of the adapter. The scientific result is comparative: does either temporal fixed-state memory retain more of the real head output than the matched static attacker at the same state budget? A failure is recorded as a failure.

## Files and boundaries

### `src/transformer_to_x/gpt2_adapter.py`

Model-specific extraction only:

- identify GPT-2 layer 5 / head 0;
- extract normalized Q/K/V affine slices and output-projection weight slice;
- capture the layer-normalized residual entering attention;
- capture the real pre-`c_proj` concatenated head-mixture tensor and attention-module output;
- reconstruct selected-head attention and verify full module parity;
- expose a model-independent `PretrainedHeadGeometry` data object.

It must not implement resident-memory logic.

### `src/transformer_to_x/pretrained_shadow.py`

Model-independent shadow evaluation:

- validate extracted geometry shapes and finite values;
- build address query and write operator using TransformerToX semantics;
- run inclusive causal resident update/read steps;
- compute per-token and aggregate metrics;
- enforce no-history resident interfaces;
- return structured results independent of Hugging Face.

### `experiments/run_gpt2_shadow.py`

Optional real-model runner:

- imports optional `torch` and `transformers` dependencies;
- loads frozen GPT-2;
- tokenizes the frozen context fixture;
- uses canonical layer 5 / head 0;
- runs adapter parity checks;
- evaluates static/decay/resonant residents;
- writes `results/gpt2_shadow.json` or a user-specified output path.

### `experiments/gpt2_shadow_contexts.json`

Small committed fixture containing the exact selection and held-out English contexts. Contexts are fixed before the first scored resident run and are not modified in response to results.

### `tests/`

Add tests for:

- affine head-slice extraction from tiny synthetic packed projections;
- K-bias cancellation;
- V-bias fixed-offset handling;
- exclusion of the shared output bias from a single-head contribution;
- exact selected-head reconstruction on a tiny local attention fixture;
- full multi-head output reconstruction from verified head slices;
- inclusive update-then-read causal ordering;
- equal state budgets;
- no raw-history retention on resident path;
- deterministic shadow metrics.

CI tests must not require network access or model downloads.

### `pyproject.toml`

Keep NumPy as the core dependency. Add pretrained execution as an optional dependency group rather than making PyTorch/Transformers mandatory for the synthetic gates.

## Frozen receipt inputs

Before the first scored resident run, code/fixtures must freeze and the receipt must record:

- model identifier `openai-community/gpt2`;
- exact model revision when available;
- exact PyTorch and Transformers versions;
- layer index 5 and head index 0;
- exact selection and held-out input strings;
- tokenizer identifier/settings;
- state budget 6,144 scalars;
- exact static/decay/resonant parameters;
- selection/held-out split;
- numerical tolerances for adapter parity.

The canonical head and contexts may not be changed after inspecting resident-memory scores. A broader later survey across heads is a different experiment.

## Receipt classification

The real-model receipt must separate three claims:

1. **Adapter parity:** PASS/FAIL — can TransformerToX exactly reconstruct the relevant frozen GPT-2 attention computation from extracted weights and captured module state?
2. **Resident shadow comparison:** descriptive matched-budget metrics for static, decay, and resonant memories.
3. **Boundary:** this experiment does not alter GPT-2, does not remove its KV cache, does not measure perplexity, and does not establish that a resident memory can substitute for the head in generation.

Only a later intervention gate may replace one live head and measure model-level effects.

## Relationship to Rytmi

Rytmi is deliberately not imported into this first GPT-2 gate.

If static/decay/resonant resident state shows that real-head history is at least partly compressible, the next experiment can add a Rytmi-inspired candidate carrying explicit local temporal direction/phase structure and compare it under the same 6,144-scalar state budget. That later gate should ask whether temporal organization improves compression, not merely whether another larger state works better.

This ordering preserves interpretability:

`real GPT-2 head -> exact adapter parity -> simple fixed-state shadow -> Rytmi-style temporal structure -> only then live head replacement`.

## Success criterion for this stage

This stage is successful if it produces a reproducible answer to the narrow question:

> Can a fixed-size resident machine, seeing each real GPT-2 hidden event once, predict the output contribution that one frozen attention head retrieves from explicit token history?

A negative answer is a valid scientific result. The repo must not call it KV-cache replacement unless a later live intervention actually removes cache for the replaced computation and preserves model behavior.

## Implemented outcome

The frozen real run passed adapter parity at `1e-5`: maximum attention-probability error `4.02e-7`, selected-head mixture error `8.06e-7`, and full `c_proj` reconstruction error `3.12e-6`.

At the matched 6,144-scalar budget, held-out relative RMSE was `4.6506` for static memory, `6.4637` for decay, and `2.4242` for resonant memory. The resonant candidate therefore reduced relative RMSE by about 47.9% versus the static attacker, but mean cosine similarity remained only `0.1888` versus static `0.1902`. The scientific conclusion is consequently bounded: temporal resonance retained more of the head output in squared-error terms, but this first resident readout is still far too inaccurate to substitute for the real head. No KV cache was removed and no generation behavior was tested.
