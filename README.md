# TransformerToX

> Can a frozen transformer head be compiled into **address geometry + write geometry**, and can its explicit token history later be replaced by fixed-size resident temporal state?

This repo starts with the compiler, not the brain analogy. The biological/window language is a hypothesis generator; every claim here has to survive a synthetic mechanism gate or a frozen real-model test.

## Gate 0 — exact address/write compiler

For a single attention head,

```text
q = x W_Q
k = x W_K
v = x W_V
```

we expose two fixed operators:

```text
M = W_Q W_K^T    # address geometry
N = W_V W_O      # write geometry
```

The ordinary reference path computes causal attention through explicit `Q`, `K`, and `V`. The compiled path instead computes

```text
score(t,i) = x_t M x_i^T / sqrt(d_head)
write(i)   = x_i N
```

and applies the same causal softmax.

The canonical deterministic head uses `d_model=12`, `d_head=6`, weight seed `20260924`, sequence lengths 4/8/16/24, and eight frozen sequence seeds per length.

Frozen Gate 0 receipt (`results/gate0.json`):

| metric | result |
|---|---:|
| test cases | 32 |
| learned/fitted parameters | **0** |
| maximum logit absolute error | **1.78e-15** |
| maximum attention absolute error | **3.33e-16** |
| maximum output absolute error | **1.33e-15** |
| aggregate output relative RMSE | **3.43e-16** |
| mean output cosine similarity | **1.000** |

**Classification: `PASS_EXACT_COMPILER`.** For this frozen head, `M` and `N` are an algebraically equivalent representation of the original attention-head weights to floating-point precision.

Gate 0 does **not** remove KV history, replace softmax attention, show an efficiency advantage, or establish a biological mechanism. It only gives us a clean translation target.

## Gate 1 — resident temporal memory primitives

The next question removes the easy part: explicit history.

```text
ordinary head
current query + stored K/V history
             ↓
       attention output

TransformerToX candidate
past events update fixed-size resident state once
             ↓
current query reads that resident state
             ↓
      approximate head output
```

The fixed-state memories all use the same content-addressed pseudo-event readout and expose an exact scalar state budget. None stores raw token history.

The synthetic memory primitives, recurrence equations, bounded-state checks, and equal-budget guard are implemented and regression-tested. The scientific comparisons below use a real frozen pretrained head.

## Real GPT-2 shadow gate — what happens on an established transformer?

The repo runs the resident-state question against **real frozen GPT-2 small** without changing GPT-2's generation path.

Canonical run:

- model: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- zero-based layer **5**, head **0**, fixed before scoring;
- `d_model=768`, `d_head=64`;
- 3 frozen selection contexts and 6 frozen held-out English contexts;
- no gradient updates and no generated continuation;
- exactly **6,144 resident scalars** for every candidate: 8 × 768;
- temporal settings selected only on the selection contexts, then frozen for held-out evaluation.

Before scoring any resident memory, `gpt2_adapter.py` reconstructs the chosen real head from GPT-2's packed Q/K/V weights and captured layer input. It also reconstructs the complete attention-module projection to make sure the head slice and shared `c_proj` geometry have been interpreted correctly.

The first frozen run established adapter parity and compared static, decay, and resonance. Its held-out result was:

| fixed resident memory | relative RMSE ↓ | mean cosine ↑ | max abs error ↓ |
|---|---:|---:|---:|
| static | 4.6506 | **0.1902** | 19.1344 |
| decay | 6.4637 | 0.1710 | 19.3502 |
| resonant | **2.4242** | 0.1888 | **7.2852** |

Resonance cut RMSE strongly but did not improve output-direction alignment: cosine stayed near 0.19. That made `2.4242 RMSE / 0.1888 cosine` the frozen hurdle for the next mechanism.

Frozen baseline evidence is in `results/gpt2_shadow.json`; the full raw baseline receipt is workflow `36015751802`, artifact `10814183911`, SHA-256 `aaeb790d6a5a7a5f37cccddabf208177f55a86f4c72e27edd699e480409c3e95`.

## Rytmi-window follow-up — does local temporal direction help?

This follow-up imports one narrow mechanism from the Rytmi/TATWATASW line, not its biological story: a fast and a slow trace define both a local temporal level and a direction-capable coordinate.

For each of four temporal bands:

```text
fast_t = rho_fast * fast_(t-1) + x_t
slow_t = rho_slow * slow_(t-1) + x_t

read components = (fast - slow, slow)
```

The four fast/slow pairs store exactly eight 768-dimensional vectors, so the state budget remains **6,144 scalars**. No theta rhythm, STDP, replay network, symbolic phase, learned memory parameters, or raw history is added.

A three-choice decay-pair grid was frozen before the first Rytmi real-head run. Selection used only the same three selection contexts and relative RMSE. The selected four pairs were:

```text
(0.00, 0.50), (0.25, 0.75), (0.50, 0.875), (0.75, 0.96875)
```

The first Rytmi follow-up compared the direction-capable coordinate with a slow-only control and found:

| fixed resident memory | relative RMSE ↓ | mean cosine ↑ | max abs error ↓ |
|---|---:|---:|---:|
| static | 4.6506 | 0.1902 | 19.1344 |
| decay | 6.4637 | 0.1710 | 19.3502 |
| resonant | 2.4242 | 0.1888 | 7.2852 |
| **Rytmi `(fast-slow, slow)`** | **1.8595** | **0.2377** | **6.2210** |
| Rytmi slow-only | 9.4439 | 0.1616 | 30.3418 |

Against the frozen resonant hurdle, `(fast-slow, slow)` reduced relative RMSE by **23.3%** and raised cosine by **25.9% relative**. It beat resonance on both RMSE and cosine on all six held-out contexts.

Frozen evidence for that run is in `results/gpt2_shadow_rytmi.json`; the full raw receipt is workflow `36018909673`, artifact `10815697972`, SHA-256 `37fd94feaf1615b08b6460c97c3c61513d168453948544efe5f5376543951fb9`.

## Coordinate control — does subtraction itself matter to the query reader?

The slow-only result did not isolate the direction coordinate because `(fast-slow, slow)` contains the fast trace while slow-only does not. The next control therefore held **everything stored in memory fixed** and changed only the coordinates exposed to the same query-conditioned reader:

```text
direction coordinates: (fast - slow, slow)
raw coordinates:       (fast, slow)
```

These two representations are invertibly related and contain the same information. They use the same selected time constants, the same eight stored vectors, the same 6,144-scalar budget, the same GPT-2-derived instantaneous query, the same normalization/softmax reader, the same frozen head, and the same held-out texts. No new parameter search was performed.

Held-out result:

| coordinate/read control | relative RMSE ↓ | mean cosine ↑ | max abs error ↓ |
|---|---:|---:|---:|
| **`(fast-slow, slow)`** | **1.8595** | **0.2377** | **6.2210** |
| raw `(fast, slow)` | 6.5546 | 0.1686 | 20.3008 |
| slow-only | 9.4439 | 0.1616 | 30.3418 |
| resonant | 2.4242 | 0.1888 | 7.2852 |

The direction coordinate beat raw `(fast, slow)` by **4.6950 absolute RMSE** and **0.0691 absolute cosine**, and it won on both metrics in every one of the six held-out contexts.

This is a coordinate/readout result, not an information-theoretic one. `(fast-slow, slow)` does not contain more information than `(fast, slow)`. The current pseudo-event reader scores each component separately after normalization and softmax, so it is not invariant to an invertible change of basis. On this frozen GPT-2 head, making temporal change explicit creates a basis that the existing GPT-2-derived query can address far more effectively than the raw fast/slow basis.

That narrows the next question: the bottleneck is not merely whether temporal history is present in resident state, but **whether it is presented in coordinates that the current query geometry can separate and retrieve**.

Frozen repository evidence is in `results/gpt2_shadow_raw_fast_slow.json`. The full raw receipt is workflow `36022357602`, artifact `10817671674`, SHA-256 `0bbff02514f0dbb3ecd5c7ad89bede3384c49f1b25c1555ee0835ba9420e4c32`.

**Boundary:** GPT-2 still used ordinary attention internally during all shadow runs. These gates did not remove KV cache, replace a live head, change generation, or measure perplexity. The result is from one frozen head and one fixed reader; it does not establish that this basis is generally optimal across heads, layers, models, or tasks.

## Gate 2 — exact window + temporal summaries vs an equal-budget cache

**A correction to how the tables above should be read.** A GPT-2 small head stores 64 + 64 = 128 scalars per token, so 6,144 scalars is an exact KV cache of the last **48 tokens**. The held-out sentences above are at most **29 tokens** long, and on them the equal-budget cache is exact (measured: relative RMSE 3.4e-16). The earlier tables rank resident memories against each other, not against the cache. Relative RMSE 1.0 is what predicting a zero head output scores, so every resident row above is worse than outputting nothing.

Gate 2 moves to longer contexts (`experiments/gpt2_long_contexts.json`, 236–255 tokens) and tests an exact window of recent K/V plus seven leaky levels fed only by evicted tokens, all counted in K/V scalars. The levels are read two ways from the same state: **band** (level l − level l−1, a partition of the evicted past, the Sihti residue identity in time) and **lowpass** (nested levels read directly). Pre-registration: `docs/superpowers/specs/2026-09-24-window-residue-gate.md`.

Frozen receipt: `results/gpt2_window_residue.json` (local run, torch 2.7.1, transformers 4.57.6, pinned GPT-2 revision; adapter parity PASS, max error 4.5e-6). Selected ladder τ = 8…512. Held-out, positions ≥ 48, 1,014 tokens:

| row | budget | far relative RMSE ↓ | far cosine ↑ |
|---|---:|---:|---:|
| zero output | 0 | **1.000** | 0.000 |
| lowpass (W=40 + 7 levels) | 6,023 | 1.160 | 0.569 |
| band (W=40 + 7 levels) | 6,023 | 1.230 | 0.585 |
| exact 48-token window | 6,144 | 1.338 | **0.588** |
| 40-token window alone | 5,120 read | 1.374 | 0.576 |
| Rytmi resident (PR #3) | 6,144 | 1.573 | 0.323 |

**Classification: `PASS_RESIDUE_BEATS_EQUAL_BUDGET_WINDOW`, but narrow.**

- The pre-registered primary passed: band beat the equal-budget 48-token window in 5/5 contexts. The stack also beat its own 40-token window in 5/5.
- Lowpass beat band in 5/5 contexts. For this reader, partitioned bands did not outperform overlapping lowpass summaries. That says nothing against Sihti's residue identity itself, only that these coordinates did not win this retrieval test.
- **Every nonzero method, including the exact 48-token cache, is worse than predicting zero.** Their cosines of about 0.58 show some directional alignment, but the cosine alone does not show that excess magnitude is the whole problem.
- Five hand-written contexts are too few for a broad claim.

**Hypothesis, not yet established:** cropping old keys renormalizes the remaining softmax weights. If this head parks attention on its first token(s), and those carry small value contributions (an attention sink, as in StreamingLLM), then a window that drops them inflates the output. The summaries would then win by retaining old mass rather than temporal structure. Lowpass counts old tokens several times, and selection chose the slowest ladder, both of which fit this reading, but none of it was measured. Gate 3 measures it.

Gate 2 shows a head-output result only. It does not show better generation or perplexity, and no KV cache was removed.

## Gate 3 — sink control

Pre-registration: `docs/superpowers/specs/2026-09-24-sink-control-gate.md`. Frozen receipt: `results/gpt2_sink_control.json` (local run, adapter parity PASS). Nothing was selected. τ = 8…512 and the lowpass primary were frozen from Gate 2, and scoring used five **fresh** contexts no gate had seen.

**Head diagnostics** (fresh fixture, positions ≥ 48; the Gate 2 fixture agrees to within 0.01):

| measurement | value |
|---|---:|
| mean attention on token 0 | **0.616** |
| mean attention on tokens 0–3 | 0.619 |
| attention mass a 48-token window drops on tokens 0–3 | 0.619 |
| attention mass a 48-token window drops elsewhere | **0.183** |
| token 0's value-output norm / median token's | 0.362 |
| norm of token-0 term / norm of head output | 0.682 |

Layer 5 head 0 is a strong first-token sink. Token 0 gets 62% of the attention at far positions while having about a third of a typical token's value norm. Even with token 0 kept, a 48-token window still loses 18% of attention mass spread over older tokens.

**Result** (fresh fixture, pooled positions ≥ 48; "size" is the RMS norm ratio predicted/target):

| row | scalars | rel. RMSE ↓ | cosine ↑ | size | wins vs sink+47 |
|---|---:|---:|---:|---:|---:|
| zero | 0 | 1.000 | 0.000 | 0.00 | — |
| window 48, no sink | 6,144 | 1.329 | 0.598 | 1.72 | — |
| sink + 39 alone | 6,023 stored | 0.451 | 0.908 | 0.97 | 0/5 |
| sink + 47 | 6,144 | 0.406 | 0.924 | 0.98 | — |
| sink + 39 + band | 6,023 | 0.386 | 0.935 | 0.95 | 4/5 |
| **sink + 39 + lowpass** | 6,023 | **0.330** | **0.957** | 0.92 | **5/5** |

**Classification: `PASS_TEMPORAL_SUMMARY_BEATS_SINK_WINDOW_AND_ZERO`.**

- **The sink explains Gate 2.** Keeping one sink token takes the equal-budget window from 1.33 to 0.41, and the output size from 1.72× to 0.98×. The Gate 2 "pass" was mostly lost sink mass.
- **Summaries still help once the sink is kept.** Seven lowpass levels beat eight extra exact recent tokens, 19% lower error with fewer scalars, 5/5 fresh contexts and 5/5 on the Gate 2 fixture (0.335 vs 0.409).
- **Band loses to lowpass in 10/10 contexts** across both fixtures.
- **Limits:** one sink-heavy head, ten similar hand-written contexts, head output only. Sink + recent window is StreamingLLM (Xiao et al. 2023). Compressing evicted tokens into summaries has prior art (Compressive Transformer and KV-compression work). This is an equal-budget measurement, not a new mechanism.

## Gate 4 — perplexity with every head replaced

Pre-registration: `docs/superpowers/specs/2026-09-24-perplexity-gate.md`. Frozen receipt: `results/gpt2_perplexity.json`.

All 144 heads of GPT-2 small run on the bounded memory inside a float64 NumPy GPT-2 (`src/transformer_to_x/gpt2_numpy.py`). Parity against the real Hugging Face forward (full cache): max logit error **0.0027**, mean-NLL error **4e-7**. Text: eight 1,024-token chunks of tinyshakespeare. Metric: teacher-forced perplexity on positions ≥ 48.

| memory per head | share of cache | far perplexity ↓ | chunk wins | gap to full cache closed |
|---|---:|---:|---:|---:|
| full cache | 100% | **73.2** | — | — |
| window 48, no sink | 4.7% | 5,334.5 | — | — |
| sink4 + 44 (StreamingLLM default) | 4.7% | 137.1 | — | — |
| sink1 + 47 | 4.7% | 133.3 | — | baseline |
| sink1 + 39 alone | 3.9% read | 157.0 | — | — |
| sink1 + 39 + band | 4.7% | 113.5 | 8/8 | 27% |
| **sink1 + 39 + lowpass** | 4.7% | **102.9** | **8/8** | **43%** |
| sink1 + 127 | 12.5% | 83.7 | — | baseline |
| **sink1 + 119 + lowpass** | 12.5% | **74.3** | **8/8** | **89%** |

**Classification: `PASS_LOWPASS_LOWERS_PERPLEXITY_AT_EQUAL_BUDGET`.**

- With every head keeping 1/8 of its cache plus seven leaky summaries, perplexity goes from 73.2 to 74.3. A plain sink + window at the same budget gives 83.7. At the tight 4.7% budget the summaries remove 43% of the damage. Every chunk agrees at both budgets.
- Without a sink GPT-2 collapses (5,334). One sink beat four at this budget.
- Lowpass beat band 8/8 (18/18 across Gates 3–4).
- On 3 of 8 chunks the 12.5% memory scored lower NLL than the full cache. That is a curiosity, not a claim.
- **Limits:** one text from one contiguous passage (chunks are not independent, and Shakespeare is likely in GPT-2's training data); one small model with learned absolute positions; only sink + window as a baseline; teacher-forced only; no measured memory or speed benefit.

## Gate 5 — stronger baselines, new texts, three models

Pre-registration: `docs/superpowers/specs/2026-09-24-robustness-gate.md`. Frozen receipts:

- `results/gate5_gpt2.json`: GPT-2 small (124M), learned absolute positions. Parity max logit error 0.0027.
- `results/gate5_gpt2medium.json`: GPT-2 medium (355M), learned absolute positions. Revision not pinned in advance; resolved `6dcaa7a9…`. Parity max logit error 0.0065.
- `results/gate5_pythia160m.json`: Pythia-160m (GPT-NeoX, `step143000`), rotary positions on 25% of each head's dimensions. Parity max logit error 0.0075.

All three passed parity against the real Hugging Face forward. Every head in every layer is replaced. Three texts: tinyshakespeare, a *What's New In Python 3.13* excerpt (written after all three models' training data), and the 13 repo stories concatenated. That is 15 chunks of 1,024 tokens per model. Metric: pooled far perplexity on positions ≥ 48. Nothing was tuned: every layout, the τ = 8…512 ladder and the budgets are frozen from Gate 4.

"4.7%" and "12.5%" below are the share of a 1,024-token cache each budget buys (6,144 and 16,384 scalars per head).

### Pooled far perplexity (lower is better)

| memory per head | GPT-2 small 4.7% | 12.5% | GPT-2 medium 4.7% | 12.5% | Pythia-160m 4.7% | 12.5% |
|---|---:|---:|---:|---:|---:|---:|
| full cache | 38.4 | 38.4 | 24.1 | 24.1 | 23.2 | 23.2 |
| sink1 + window | 88.1 | 54.1 | 237.7 | 207.7 | 39.5 | 29.5 |
| **sink1 + window + 7 lowpass levels** | 65.8 | 46.4 | 304.2 | 178.8 | 44.1 | 29.8 |
| H2O (heavy hitters + recent; our online implementation) | 116.0 | 74.8 | 108.7 | 72.7 | 35.2 | **26.4** |
| merge (H2O layout, victims folded into most-similar slot with a mass; our implementation, CaM/D2O family) | **63.9** | **46.2** | **51.6** | **30.1** | **35.0** | 26.6 |

Lowpass uses 6,023 / 16,263 scalars, merge 6,063 / 16,383, the others 6,144 / 16,384.

### Lowpass against each baseline (chunks won out of 15)

| comparison | GPT-2 small 4.7% | 12.5% | GPT-2 medium 4.7% | 12.5% | Pythia 4.7% | 12.5% |
|---|---:|---:|---:|---:|---:|---:|
| lowpass vs sink + window | **15/15** | **15/15** | 0/15 | 13/15 | 0/15 | 4/15 |
| lowpass vs H2O | **15/15** | **15/15** | 0/15 | 0/15 | 0/15 | 2/15 |
| lowpass vs merge | 7/15 | 7/15 | 0/15 | 0/15 | 0/15 | 2/15 |

### Per text at the 4.7% budget

| text | model | full | sink+window | lowpass | merge | H2O |
|---|---|---:|---:|---:|---:|---:|
| Shakespeare | GPT-2 small | 67.4 | 126.2 | **97.3** | 97.5 | 198.8 |
| Python 3.13 docs | GPT-2 small | 24.7 | 72.5 | 55.3 | **51.6** | 84.5 |
| unseen stories | GPT-2 small | 30.2 | 63.5 | 42.4 | **41.9** | 74.5 |
| Shakespeare | GPT-2 medium | 38.3 | 284.7 | 331.3 | **71.0** | 152.1 |
| Python 3.13 docs | GPT-2 medium | 16.1 | 209.6 | 275.8 | **41.8** | 91.2 |
| unseen stories | GPT-2 medium | 21.4 | 213.2 | 312.0 | **41.6** | 78.7 |
| Shakespeare | Pythia-160m | 34.4 | 56.9 | 65.7 | **47.1** | 49.6 |
| Python 3.13 docs | Pythia-160m | 14.2 | 27.4 | 30.7 | 25.4 | **24.2** |
| unseen stories | Pythia-160m | 28.3 | 39.6 | 40.9 | **36.6** | 37.4 |

**Classification, all three models: `FAIL_LOWPASS_LOSES_TO_A_STRONGER_BASELINE_AT_B48`.**

- **Lowpass works only on GPT-2 small.** There it beats sink + window and our H2O in every chunk at both budgets and ties merge (7/15 chunks each way, merge slightly ahead pooled).
- **On GPT-2 medium, lowpass fails.** At 4.7% it is worse than plain sink + window in every chunk (304 vs 238). At 12.5% it beats sink + window in 13/15 chunks but closes only 7% of the gap to the full cache (179 vs 208, full cache 24). H2O and merge beat it in every chunk at both budgets.
- **On Pythia, lowpass fails.** At 4.7% it is worse than sink + window in every chunk (44.1 vs 39.5). At 12.5% it roughly ties it (29.8 vs 29.5).
- **Merge is the only method that is strong on all three models.** It is best, or within 1% of best, in all six model × budget cells. On GPT-2 medium at 12.5% it reaches 30.1 against a full cache of 24.1, while every other method stays above 72. It is our own implementation, not a reproduction of CaM or D2O.
- **GPT-2 medium does not survive on a sink and a recent window.** Going from 47 to 127 recent tokens only moves sink + window from 238 to 208, far from the full cache's 24. So what medium loses is not recent context but specific older tokens. Methods that keep individual old tokens (H2O) or merge them by similarity (merge) recover much more. Which tokens those are, and whether medium has sinks other than token 0, was not measured.
- **Our H2O is not simply broken.** It scored worse than sink + window on GPT-2 small, which looked suspicious, but it is the best row on Pythia at 12.5% and far better than sink + window on GPT-2 medium. It suits some models and not others.
- **The rotary explanation is withdrawn as the main story.** After the Pythia run the README suggested that lowpass failed there because averaging position-rotated keys cancels them. GPT-2 medium uses absolute positions, like GPT-2 small, and lowpass fails there too. Rotation may still hurt on Pythia, but it is not the general reason. A better working hypothesis: lowpass averages all evicted tokens of a time band into one key and value, which works when evicted tokens matter mainly as bulk mass (GPT-2 small, a strong single sink) and fails when a model needs particular old tokens to stay distinguishable (GPT-2 medium). Not measured.
- **Determinism check:** the GPT-2 small Shakespeare chunks for lowpass and sink + window reproduce Gate 4's per-chunk NLLs exactly.
- **Not run:** no grouped-KV model, no generation test, no speed or memory benchmark.

## Where the line stands

```text
Gate 0  exact address/write compiler                        PASS (1e-15)
PRs 2-4 resident memories on one head, <=29-token texts     ranked each other; all worse than
                                                            predicting zero; equal-budget cache is exact
Gate 2  window + temporal summaries vs equal-budget window  PASS, but mostly lost sink mass
Gate 3  sink control, one head                              PASS; lowpass > band 10/10
Gate 4  all 144 heads, perplexity, vs sink + window         PASS 8/8; 12.5% cache -> 74.3 vs 73.2 full
Gate 5  GPT-2 small: new texts, vs H2O and merge            FAIL primary: ties merge, beats H2O
Gate 5  GPT-2 medium                                        FAIL: lowpass loses to H2O and merge 0/15;
                                                            merge 30.1 vs full 24.1 at 12.5%
Gate 5  Pythia-160m (rotary positions)                      FAIL: lowpass worse than sink + window;
                                                            H2O and merge win
```

**Bottom line.** Keeping evicted tokens instead of dropping them matters a lot on all three models, but how you keep them decides everything.

- The similarity **merge** helped on all three models and was the best or near-best method in every setting. On GPT-2 medium it was the only method that stayed near the full cache.
- This repo's **leaky multi-timescale lowpass** helped only on GPT-2 small. On GPT-2 medium and Pythia-160m it did worse than plain sink + window at the tight budget.
- The Sihti-style partitioned **bands** lost to lowpass in all 18 comparisons in Gates 3–4.

So the brain-window idea (Rytmi's fast/slow traces, TATWATASW's time windows) produced a bounded-memory layout that worked on one small model and did not generalise. Its Gate 4 success came from a model where evicted tokens matter mostly as bulk mass. The method that generalised across these three models is merge-style compression, which already exists in the literature. Nothing here removes a real KV cache or speeds up generation.

## Why this connects to the older line

The intended translation is narrower than “transformers are brains”:

```text
transformer residual state   -> resident state
Q/K geometry                 -> address / susceptibility
V/O geometry                 -> write direction
KV history                   -> candidate resident temporal state
fast/slow pair               -> local temporal level + change
fast - slow                  -> query-addressable temporal direction coordinate
```

The first real-head experiment said temporal resonance preserved more history energy but still lost output direction. The Rytmi-window follow-up improved both metrics. The raw-coordinate control now says something sharper: **the same stored temporal information can be easy or hard for the same instantaneous query to retrieve depending on the coordinates exposed by the resident state.**

That makes query-addressable temporal representation the next mechanism to study. It is still not evidence for unplugging attention.

## Run

Core synthetic tests require only NumPy:

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_gate0 --out /tmp/gate0.json
python -m experiments.verify_receipts /tmp/gate0.json results/gate0.json
```

The optional real GPT-2 shadow run requires PyTorch and Transformers plus model-download access:

```bash
python -m pip install -e ".[test,pretrained]"
python -m experiments.run_gpt2_shadow --out /tmp/gpt2_shadow.json
```

## Repository map

- `src/transformer_to_x/tiny_head.py` — deterministic causal reference head.
- `src/transformer_to_x/compiler.py` — exact address/write compiler.
- `src/transformer_to_x/trace_bank.py` — static, decay, resonant, and fast/slow Rytmi-window resident memories plus matched read-coordinate controls.
- `src/transformer_to_x/pretrained_shadow.py` — model-independent pretrained-head geometry and fixed-state shadow evaluator.
- `src/transformer_to_x/gpt2_adapter.py` — GPT-2 packed-head extraction, capture hooks, and parity diagnostics.
- `src/transformer_to_x/window_residue.py` — head-space exact KV window + temporal residue (band/lowpass) memory.
- `src/transformer_to_x/metrics.py` — deterministic numerical metrics.
- `experiments/run_gate0.py` — canonical exact-compiler receipt.
- `experiments/run_gpt2_shadow.py` — optional real GPT-2 shadow experiment, including the frozen Rytmi-window grid and coordinate controls.
- `experiments/run_gpt2_window_residue.py` — Gate 2 window + residue runner.
- `experiments/gpt2_long_contexts.json` — Gate 2 long selection/held-out texts.
- `experiments/gpt2_long_contexts_fresh.json` — Gate 3 fresh held-out texts.
- `experiments/run_gpt2_sink_control.py` — Gate 3 sink-control runner.
- `results/gpt2_window_residue.json` — frozen Gate 2 receipt.
- `results/gpt2_sink_control.json` — frozen Gate 3 receipt.
- `src/transformer_to_x/gpt2_numpy.py` — NumPy GPT-2 with vectorised bounded-memory attention for every head.
- `experiments/run_gpt2_perplexity.py` — Gate 4 perplexity runner.
- `experiments/data/tinyshakespeare_head.txt` — Gate 4 text fixture.
- `results/gpt2_perplexity.json` — frozen Gate 4 receipt.
- `results/gate5_gpt2.json` — frozen Gate 5 receipt (GPT-2 small).
- `results/gate5_gpt2medium.json` — frozen Gate 5 receipt (GPT-2 medium).
- `results/gate5_pythia160m.json` — frozen Gate 5 receipt (Pythia-160m).
- `src/transformer_to_x/cache_policies.py` — H2O and merge baselines.
- `src/transformer_to_x/neox_numpy.py` — NumPy GPT-NeoX (Pythia) with rotary positions (parity PASS on Pythia-160m).
- `experiments/run_gate5.py` — Gate 5 runner.
- `experiments/data/python313_whatsnew_excerpt.txt`, `experiments/data/unseen_stories.txt` — Gate 5 text fixtures (Python docs excerpt © Python Software Foundation, PSF licence).
- `experiments/gpt2_shadow_contexts.json` — frozen selection and held-out texts.
- `experiments/verify_receipts.py` — portable frozen-receipt comparison with explicit floating tolerance.
- `results/gate0.json` — frozen Gate 0 evidence.
- `results/gpt2_shadow.json` — frozen baseline real-head shadow evidence and artifact provenance.
- `results/gpt2_shadow_rytmi.json` — frozen Rytmi-window real-head shadow evidence and artifact provenance.
- `results/gpt2_shadow_raw_fast_slow.json` — frozen raw fast/slow coordinate-control evidence and artifact provenance.
- `docs/superpowers/specs/` — approved/implemented designs.
- `docs/superpowers/plans/` — implementation plans.
