# Gate 3 — sink control, pre-registration

Written and committed before the first real score.

## What Gate 2 left open

Gate 2 passed its primary (band summaries beat an equal-budget 48-token window)
but every nonzero method, including the exact 48-token window, scored worse than
predicting zero, and overlapping lowpass summaries beat partitioned bands 5/5.
One candidate explanation: this head parks attention on its first token(s), whose
value contributions are small, and dropping them from a window renormalises
the softmax onto content tokens and inflates the output. The summaries would then
win by keeping old mass, not by temporal structure. Cropping old keys can inflate
the output through lost mass in general, so a sink is not assumed. It is measured.

## Frozen, nothing selected

- Model/revision/layer 5/head 0/parity tolerance: unchanged.
- Tau ladder (8, 16, 32, 64, 128, 256, 512): selected in Gate 2, now frozen.
- Primary read: lowpass (better in Gate 2). Band is secondary.
  Gate 2's five held-out contexts are hereby demoted to a selection set.
- Scoring fixture: `experiments/gpt2_long_contexts_fresh.json`, five new
  ~210-word contexts, never scored by any gate. Gate 2's fixture reported alongside.
- Sink = first token of the sequence (GPT-2 adds no BOS here), never evicted.
- Far metric: positions >= 48, pooled relative RMSE.

## Budgets (128 scalars per exact K/V pair, 129 per summary level)

| row | exact pairs | levels | scalars |
|---|---:|---:|---:|
| kv_window_full | 48 | 0 | 6,144 |
| sink_window_full (sink + 47) | 48 | 0 | 6,144 |
| sink_hybrid_lowpass / band (sink + 39) | 40 | 7 | 6,023 |
| sink_window_hybrid_only (levels stored, not read) | 40 | 7 | 6,023 |
| nosink_hybrid_lowpass / band (40) | 40 | 7 | 6,023 |

(A figure of 5,895 given in chat was wrong: that is 39 exact pairs, not 40.)

## Diagnostics reported (real head, far positions)

Mean attention on token 0 and on tokens 0–3; attention mass the 48-window drops
on tokens 0–3 vs elsewhere; token 0's value-output norm / median token's;
mean ||a_t0 v_0 W_O|| / ||y_t|| and ||y_t − a_t0 v_0 W_O|| / ||y_t||. Every
method reports the pooled RMS norm ratio pred/target and the mean per-token ratio.

## Questions

1. Sink check (diagnostic): does sink + 47 beat the plain 48-window and score < 1.0?
2. **Primary:** `sink_hybrid_lowpass` beats both `sink_window_full` and zero on
   the fresh fixture. PASS → `PASS_TEMPORAL_SUMMARY_BEATS_SINK_WINDOW_AND_ZERO`,
   else `FAIL_TEMPORAL_SUMMARY_NOT_USEFUL_OVER_SINK_WINDOW`.
3. Secondary: same rule for `sink_hybrid_band`.
4. Band vs lowpass with the sink in place.

## Written-down priors

Q1: likely PASS if the sink story is right. If the sink row still scores > 1.0,
the inflation comes from lost mass spread over many old tokens, and the sink
story is wrong or incomplete. Q2: lean FAIL. Once the sink is kept, eight more
exact recent tokens are hard to beat with seven averaged summaries.

## Boundary

A pass would be a head-output result on one head. It says nothing about
generation or perplexity, and no KV cache is removed.
