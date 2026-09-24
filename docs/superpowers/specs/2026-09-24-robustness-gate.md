# Gate 5 — stronger baselines, new texts, a RoPE model: pre-registration

Written and committed before the first real score.

## Why

Gate 4 showed sink1 + 39 + 7 lowpass levels beats sink1 + 47 on GPT-2 small
perplexity, 8/8 chunks, but only against StreamingLLM-style sink + window, on
one text, on one model with absolute positions.

## Frozen, nothing tuned

- Memory layouts, tau = 8..512, far_start = 48 and budgets: unchanged from Gate 4.
- Models: gpt2 (607a30d7...), pythia-160m (revision step143000), gpt2-medium
  (revision not pinned in advance; the resolved commit is recorded in the receipt).
- Parity per model vs the HF forward on the first chunk: max |logit error| <= 0.1
  and mean-NLL error <= 1e-3, else FAIL_PARITY and nothing is scored.
- Texts (sha256 pinned): tinyshakespeare head (Gate 4's text); What's New In
  Python 3.13 excerpt (2024; after both models' training data); the 13 Gate 2-3
  stories concatenated (written for this repo; unseen by any model, but seen by
  our earlier gates, and Gate 2 selected the tau ladder on three of them).
  Up to 6 chunks x 1,024 tokens per text, each chunk a fresh sequence.

## Rows (scalars per head, d_head = 64)

| row | layout | scalars |
|---|---|---:|
| b48_sink_window | sink1 + 47 | 6,144 |
| b48_lowpass | sink1 + 39 + 7 lowpass levels | 6,023 |
| b48_h2o | 48 slots: 24 recent + 24 heavy hitters | 6,144 |
| b48_merge | 47 slots with mass: H2O layout, victims merged into most-similar non-recent slot | 6,063 |
| b128_* | same at 128 exact pairs (127 slots for merge; sink1 + 119 for lowpass) | <= 16,384 |
| full_cache | — | — |

H2O is the online heavy-hitter policy (Zhang et al. 2023): scores are
accumulated attention probabilities; the victim is the non-recent slot with the
lowest score, evicted before the new token is attended. merge is OUR
implementation in the spirit of CaM/D2O, not a reproduction of either.

## Questions

Per model, pooled far perplexity over all chunks of all three texts:
lowpass vs sink_window, vs h2o, vs merge, at b48 and b128. Chunk wins, per-text
wins and the fraction of the baseline's gap to the full cache closed are reported.

**Primary (GPT-2 small):** b48_lowpass beats both b48_h2o and b48_merge.
PASS -> PASS_LOWPASS_BEATS_H2O_AND_MERGE_AT_B48, else
FAIL_LOWPASS_LOSES_TO_A_STRONGER_BASELINE_AT_B48. Other models get the same
classification string, reported as replication, not as the primary.

## Written-down priors

- lowpass vs sink_window replicates on all models: likely.
- vs H2O: uncertain. H2O keeps whichever old tokens heads actually use, and it
  keeps the sink automatically; it may beat averaged summaries at b48.
- vs merge: uncertain, lean lowpass by a small margin at b48, since merge also
  keeps old mass but pays slots for it.
- Pythia (partial rotary, 25% of dims): averaged rotated keys are untested; could fail.

## Boundary

Teacher-forced perplexity only. No generation, speed or memory benchmark. No
GQA model yet (grouped KV heads need a per-KV-head policy for a fair budget);
that is a later gate if this one holds.
