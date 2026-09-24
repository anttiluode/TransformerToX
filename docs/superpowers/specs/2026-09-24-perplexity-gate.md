# Gate 4 — perplexity with every head replaced, pre-registration

Written and committed before the first real score.

## Question

Gate 3 showed, on one sink-heavy head, that sink + 39 exact tokens + 7 lowpass
levels reconstructs the head output better than sink + 47 exact tokens at equal
budget. Does that survive when EVERY head in GPT-2 small uses the memory, and
does it lower next-token perplexity? Head-output reconstruction on one head
does not imply this: other heads may be local, and errors compound over layers.

## Frozen

- GPT-2 small at the pinned revision, run in float64 NumPy. Parity vs the real
  HF forward (full cache) on chunk 0: max |logit error| <= 5e-2 and mean-NLL
  error <= 1e-3, else FAIL_PARITY and nothing is scored.
- Text: `experiments/data/tinyshakespeare_head.txt`, first 44,961 chars of
  karpathy/char-rnn tinyshakespeare (sha256 3f0590...cacef). Eight
  non-overlapping 1,024-token chunks, each a fresh sequence starting at position 0.
- Memory per head, all 144 heads identical: sink tokens pinned, sliding window,
  7 leaky levels at tau = 8..512 fed only by evicted tokens (frozen from Gate 2).
- Metric: pooled mean next-token NLL over positions >= 48 (predictions of tokens
  49..1023), reported as perplexity. All-position perplexity also reported.

## Rows

| row | exact pairs | levels | scalars/head |
|---|---:|---:|---:|
| full_cache | all | 0 | — |
| b48_window48 | 48 | 0 | 6,144 |
| b48_sink1_window47 | 48 | 0 | 6,144 |
| b48_sink4_window44 | 48 | 0 | 6,144 |
| b48_sink1_window39_only | 40 | 7 stored, unread | 6,023 |
| b48_sink1_window39_lowpass | 40 | 7 | 6,023 |
| b48_sink1_window39_band | 40 | 7 | 6,023 |
| b128_sink1_window127 | 128 | 0 | 16,384 |
| b128_sink1_window119_lowpass | 120 | 7 | 16,263 |

## Questions

1. **Primary:** b48 lowpass far perplexity < b48 sink1+47.
   PASS -> PASS_LOWPASS_LOWERS_PERPLEXITY_AT_EQUAL_BUDGET, else
   FAIL_SINK_WINDOW_WINS_PERPLEXITY. Chunk wins (of 8) and the fraction of the
   sink-window-to-full-cache gap closed are reported.
2. Secondary: the same comparison at 16,384 scalars.
3. b48 band vs sink1+47, and lowpass vs band.
4. References: sink1 vs no sink; sink1 vs sink4.

## Written-down priors

Sink vs no sink: large perplexity improvement expected (StreamingLLM).
Primary: genuinely uncertain, lean slight PASS. Gate 3 said older mass matters
for at least one head, but many heads are local and would prefer 8 more exact
recent tokens. Secondary (b128): smaller effect than b48, because less old mass
is dropped. A pass with a tiny gap-closed fraction (< 5%) would be reported as
real but practically small.

## Boundary

Teacher-forced perplexity on one text with one model. Tinyshakespeare may
overlap GPT-2's training data; that affects absolute perplexity for all rows
equally but should be kept in mind. No generation, no speed or memory benchmark,
no comparison against published KV-compression methods beyond sink + window.
