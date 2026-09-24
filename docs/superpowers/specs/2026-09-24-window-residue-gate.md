# Window + temporal residue gate — pre-registration

Written and committed before the first real GPT-2 score.

## Why this gate exists

Every earlier real-head table (PRs #2-#4) compared resident memories at 6,144
scalars on held-out contexts of at most 29 tokens. A GPT-2 small head stores
64 + 64 = 128 scalars per token, so 6,144 scalars is an exact KV window of 48
tokens. On those contexts the plain equal-budget cache is exact (relative RMSE 0).
The earlier tables ranked resident memories against each other; none of them was
compared against keeping the cache. This gate adds that comparison, and moves to
contexts long enough (~250 tokens) that a 48-token window is not exact.

## Hypothesis (from Sihti + TATWATASW)

Keep an exact recent window (the present). Feed only the tokens that fall out of
it into a ladder of leaky integrators at octave time constants. Read the ladder in
**band** (residue) coordinates: band l = level l − level l−1. Each band is a
non-negative time window at an intermediate lag, and the bands telescope to the
slowest level, so they partition the evicted past — the temporal analogue of
Sihti's lossless residue stack. Each band is read as one pseudo-token
(mean key, mean value, logit + log mass) in the same softmax as the window.

## Frozen setup

- Model, revision, layer 5 / head 0, parity tolerance: unchanged from PR #2.
- Contexts: `experiments/gpt2_long_contexts.json`, 3 selection / 5 held-out.
- Budgets: `kv_window_full` W=48 → 6,144. Hybrid W=40 + 7 levels
  (key trace + value trace + mass each) → 6,023 ≤ 6,144. Masses counted even
  though they are a deterministic function of time.
- Tau ladders (7 octave levels): (2..128), (4..256), (8..512). Chosen on
  selection contexts only, by far-token relative RMSE of the band read.
- Primary metric: relative RMSE over positions ≥ 48 (where the full window is
  inexact), pooled over the 5 held-out contexts. All-token metrics also reported.

## Questions and pass rules

1. **Primary.** `window_residue_band` far RMSE < `kv_window_full` far RMSE.
   PASS → `PASS_RESIDUE_BEATS_EQUAL_BUDGET_WINDOW`, else
   `FAIL_WINDOW_WINS_AT_EQUAL_BUDGET`. Per-context wins reported.
2. Does the stack add anything? band < `kv_window_hybrid_only` (same W=40,
   stack stored but never read).
3. Partition vs nested: band < `window_residue_lowpass` (same state).
   Caveat: this is not a pure coordinate control like PR #4. Lowpass levels are
   nested, so an old token contributes to several pseudo-tokens at once;
   bands partition. A band win says "partitioned time windows beat overlapping
   accumulations", which is the Sihti claim, not "basis alone".

Reference rows: `zero` (relative RMSE exactly 1.0) and `rytmi_resident_pr3`
(PR #3's selected residual-space Rytmi memory, unchanged, 6,144 scalars).
Short-context check: `kv_window_full` on the original held-out sentences.

## Expectation, written down

Honest prior: question 2 probably passes (anything beats nothing for lags > 40),
and the primary is a coin flip leaning FAIL — eight exact extra tokens are a
strong competitor, and many GPT-2 heads are local. A FAIL on the primary is
still a clean, publishable result: at this budget the brain-style resident
state does not buy anything over the cache it is meant to replace.

## Boundary

No KV cache removed, no live attention replaced, no generation or perplexity.
