# TransformerToX compiler-first design

Date: 2026-09-24
Status: approved architecture, awaiting written-spec review

## Purpose

TransformerToX asks whether a frozen transformer attention head can be translated into a brain-inspired resident-operator system without first retraining the transformer.

The first version is intentionally small and falsifiable. It starts from a deterministic single-head causal attention model, proves an exact algebraic decomposition, and only then tests whether explicit KV history can be replaced by a fixed-size resident temporal state.

The project is not trying to claim that transformer weights are biological synapses, that the brain implements attention, or that a recurrent trace bank is automatically a better language-model architecture. The first scientific question begins only after the exact algebraic compiler is established.

## Core translation

For one attention head with model state x,

- q = x W_Q
- k = x W_K
- v = x W_V

The exact compiler exposes two fixed operators:

- Address operator: M = W_Q W_K^T
- Write operator: N = W_V W_O

For current token x_t and prior token x_i, the compiled attention logit is computed from x_t M x_i^T, with the usual head scaling. The value contribution is computed from x_i N.

This separates one frozen attention head into:

1. address geometry: when/where a prior event is compatible with the current state;
2. write geometry: what direction the compatible event contributes to the output.

## Scientific gates

### Gate 0 — exact algebraic compiler

Goal: prove that the address/write decomposition reproduces a conventional frozen causal attention head with zero fitting.

Reference path:

1. X -> Q = X W_Q
2. X -> K = X W_K
3. X -> V = X W_V
4. causal logits = Q K^T / sqrt(d_head)
5. causal softmax
6. output = attention V W_O

Compiled path:

1. M = W_Q W_K^T
2. N = W_V W_O
3. causal logits from pairwise x_t M x_i^T / sqrt(d_head)
4. same causal softmax
5. output from weighted x_i N

Pass conditions:

- compiled logits match reference logits within strict floating-point tolerance;
- compiled attention probabilities match reference probabilities within strict floating-point tolerance;
- final head outputs match within strict floating-point tolerance;
- zero learned or fitted parameters are introduced.

Gate 0 is algebraic validation, not a scientific breakthrough.

### Gate 1 — resident temporal compression

Goal: test whether explicit KV history can be approximated by a fixed-size recurrent state while preserving frozen-head outputs.

The resident-memory API receives each token exactly once and updates a fixed-size state. At query time it must produce the head output without replaying or storing the complete token history.

Candidate memories:

1. Static low-rank memory — matched-capacity non-temporal attacker.
2. Decaying trace bank — several fixed decay modes.
3. Resonant trace bank — damped oscillatory modes with fixed radii/frequencies.

All methods are compared under equal scalar state budget rather than equal channel count.

The first pass uses a fixed candidate grid. Candidate decay constants and resonant frequencies may be selected on a training split, then frozen. Held-out evaluation is the only reported performance.

Primary metrics:

- relative RMSE of head output versus explicit-KV reference;
- maximum absolute error;
- cosine similarity of output vectors;
- scalar state budget;
- error as a function of sequence length and lag.

Gate 1 earns a positive result only if a temporal resident memory beats the matched static-memory attacker at the same state budget on held-out sequences. If it fails, the negative result is preserved.

### Gate 2 — local temporal direction

Gate 2 is attempted only if Gate 1 leaves a useful temporal-memory mechanism.

It imports the narrow lesson from AnotherOddThing v5: two histories can share the same present amplitude while differing in temporal direction.

The test constructs matched histories with the same present resident-memory amplitude but opposite temporal direction. A local fast/slow contrast or resonant phase is allowed to read the direction; a matched level-only attacker is not.

The gate asks whether locally accumulated state can supply a useful entering/leaving or phase-like coordinate without an externally supplied symbolic position variable.

This remains a synthetic mechanism test, not a biological claim.

## v0 model

The first model is a deterministic NumPy implementation of one causal attention head.

Recommended frozen dimensions:

- d_model: 12
- d_head: 6
- sequence lengths: 4, 8, 16, 24
- deterministic weight seed: 20260924
- deterministic evaluation sequence seeds: a fixed published list

Weights are sampled once from a fixed seeded distribution and then frozen. No optimizer is required for Gate 0.

The model should support both batched full-sequence evaluation and step-by-step causal evaluation so the same API can later host pretrained weights.

## Components

### `src/transformer_to_x/tiny_head.py`

Owns the deterministic reference head:

- frozen W_Q, W_K, W_V, W_O;
- causal masking;
- stable softmax;
- reference logits, attention weights, and output;
- token-by-token reference path for later adapter parity.

### `src/transformer_to_x/compiler.py`

Owns the exact translation:

- builds M = W_Q W_K^T;
- builds N = W_V W_O;
- computes compiled logits directly from token states;
- computes compiled output directly from N;
- exposes explicit address and write operators for inspection.

### `src/transformer_to_x/trace_bank.py`

Defines one common resident-memory interface and the Gate 1 candidates:

- static low-rank state;
- decaying trace bank;
- resonant trace bank.

The implementation must report its exact scalar state budget.

No method may secretly retain the raw history in the Gate 1 path.

### `src/transformer_to_x/metrics.py`

Provides deterministic numerical metrics:

- absolute error;
- relative RMSE with explicit zero-denominator handling;
- cosine similarity with explicit zero-vector handling;
- state-budget accounting.

### `experiments/run_gate0.py`

Creates the frozen head and deterministic evaluation sequences, compares reference and compiled paths, and emits a JSON receipt.

### `experiments/run_gate1.py`

Runs the matched-budget memory comparison. Cheap canonical settings stay in CI. Larger sweeps are available as explicit commands but are not required for ordinary validation.

## Data flow

### Exact path

```
X
|-- reference --> Q,K,V --> causal attention --> V W_O
|
`-- compiled ----> M,N --> address scores --> weighted writes
```

Both paths operate on the same frozen tokens and weights.

### Resident-memory path

```
event x_t
   |
   v
resident memory update
   |
   v
fixed-size temporal state
   ^
   |
query/current token --> resident readout --> approximate head output
```

Once an event has updated resident state, the Gate 1 implementation may not consult that raw event again.

## Pretrained adapter boundary

After Gate 0 is proven on the deterministic head, the same compiler API will accept externally supplied matrices with compatible shapes.

The first pretrained adapter should do only three things:

1. extract one attention head's W_Q, W_K, W_V, W_O from a supported small model;
2. convert them into the internal NumPy/tensor representation;
3. run the exact Gate 0 comparison on captured hidden states.

The pretrained adapter must not modify the compiler semantics. Model-specific extraction belongs in a separate adapter module.

No large pretrained model is required for v0 CI. If a model download or GPU-heavy sweep is useful, the repository will provide a frozen command for the user to run and a machine-readable result format to bring back.

## Error handling and invariants

The code should fail loudly on:

- incompatible matrix shapes;
- non-causal sequence use in causal-only routines;
- NaN/Inf values in frozen weights or inputs;
- invalid trace parameters such as non-positive decay times or unstable radii where stability is required;
- state-budget mismatches in matched comparisons;
- accidental history retention by interfaces intended to be fixed-state.

Numerical comparisons should report actual tolerances and measured maxima rather than using vague pass/fail assertions alone.

## Testing strategy

Development follows a red/green cycle.

Unit tests cover:

- exact shape contracts;
- causal masking;
- stable softmax;
- M and N construction;
- logit equality;
- output equality;
- resident-memory state-budget accounting;
- recurrence update equations;
- deterministic candidate selection;
- no-history resident interface.

Scientific regression tests load frozen receipts and reproduce them exactly or within explicitly stored tolerances.

CI targets Python 3.11 and 3.12 and runs only the cheap canonical experiment settings.

## Frozen receipts

Each gate writes a JSON receipt under `results/` containing:

- gate name and classification;
- dimensions and seeds;
- all frozen hyperparameters;
- compared methods and state budgets;
- primary metrics;
- explicit claim;
- explicit boundary / what the gate does not establish.

The README summarizes only claims backed by frozen receipts.

## Initial repository shape

```
TransformerToX/
├── README.md
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/transformer_to_x/
│   ├── __init__.py
│   ├── tiny_head.py
│   ├── compiler.py
│   ├── trace_bank.py
│   └── metrics.py
├── experiments/
│   ├── __init__.py
│   ├── run_gate0.py
│   └── run_gate1.py
├── results/
│   ├── gate0.json
│   └── gate1.json
├── tests/
│   ├── test_tiny_head.py
│   ├── test_compiler.py
│   ├── test_trace_bank.py
│   └── test_receipts.py
└── docs/superpowers/specs/
    └── 2026-09-24-transformer-to-x-design.md
```

Gate 1 files may initially contain only the minimal interfaces/tests needed to stage the experiment; Gate 0 is completed first.

## Success and stopping rules

1. Do not advance to Gate 1 until Gate 0 is exact and frozen.
2. Do not call Gate 0 a scientific result; it validates the compiler.
3. Do not increase Gate 1 capacity after seeing held-out results without creating a new declared experiment.
4. Compare temporal and static memories at equal scalar state budget.
5. Preserve negative results.
6. Do not move to a real pretrained head until the deterministic compiler API is exact.
7. If larger compute is useful, provide a reproducible command and result schema rather than making it a hidden dependency.

## Expected first deliverable

The first merged milestone should contain a complete Gate 0 and the Gate 1 interface/attacker skeleton, with a frozen Gate 0 receipt demonstrating zero-fit equivalence between ordinary causal attention and the compiled address/write representation.
