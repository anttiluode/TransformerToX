"""Gate 4: perplexity with EVERY GPT-2 head replaced by a bounded memory.

Pre-registered in docs/superpowers/specs/2026-09-24-perplexity-gate.md BEFORE
the first real score. Nothing is selected: tau ladder, sink count, read mode and
budgets are all frozen from Gates 2-3.

The model runs in float64 NumPy (``transformer_to_x.gpt2_numpy``) from the
pinned GPT-2 weights. A parity check against the real Hugging Face forward pass
(full cache) must pass before any memory row is scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.run_gpt2_shadow import MODEL_ID, MODEL_REVISION, load_real_runtime
from transformer_to_x.gpt2_numpy import GPT2Numpy, MemorySpec, token_nll
from transformer_to_x.window_residue import rhos_from_taus

TEXT_PATH = Path(__file__).with_name("data") / "tinyshakespeare_head.txt"
TEXT_SHA256 = "3f0590676115fc4d4c2edd6466f021f46b5bf5d4acd77cb63fdcc338ac3cacef"

# ---- frozen before the first real score -------------------------------------
CHUNK = 1024
N_CHUNKS = 8
FAR_START = 48                 # score next-token NLL from positions >= 48
TAUS = (8, 16, 32, 64, 128, 256, 512)
PARITY_MAX_LOGIT_ERR = 5e-2    # float32 HF vs float64 NumPy
PARITY_MAX_MEAN_NLL_ERR = 1e-3
# -----------------------------------------------------------------------------


def method_specs() -> dict[str, MemorySpec]:
    r = rhos_from_taus(TAUS)
    return {
        "full_cache": MemorySpec(),
        # budget 6,144 scalars per head (48 exact pairs)
        "b48_window48": MemorySpec(window=48),
        "b48_sink1_window47": MemorySpec(window=47, sink=1),
        "b48_sink4_window44": MemorySpec(window=44, sink=4),
        "b48_sink1_window39_only": MemorySpec(window=39, sink=1, rhos=r, read_mode="window_only"),
        "b48_sink1_window39_lowpass": MemorySpec(window=39, sink=1, rhos=r, read_mode="lowpass"),
        "b48_sink1_window39_band": MemorySpec(window=39, sink=1, rhos=r, read_mode="band"),
        # budget 16,384 scalars per head (128 exact pairs)
        "b128_sink1_window127": MemorySpec(window=127, sink=1),
        "b128_sink1_window119_lowpass": MemorySpec(window=119, sink=1, rhos=r, read_mode="lowpass"),
    }


def load_text() -> str:
    text = TEXT_PATH.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != TEXT_SHA256:
        raise ValueError("frozen text fixture changed")
    return text


def make_chunks(ids: np.ndarray) -> list[np.ndarray]:
    if ids.shape[0] < CHUNK * N_CHUNKS:
        raise ValueError(f"need {CHUNK * N_CHUNKS} tokens, got {ids.shape[0]}")
    return [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(N_CHUNKS)]


def parity(model_hf, np_model: GPT2Numpy, ids: np.ndarray) -> dict:
    import torch

    with torch.no_grad():
        hf = model_hf(input_ids=torch.tensor(ids[None]), use_cache=False).logits[0].double().numpy()
    ours = np_model.logits(ids)
    max_err = float(np.abs(hf - ours).max())
    nll_err = float(abs(token_nll(hf, ids).mean() - token_nll(ours, ids).mean()))
    ok = max_err <= PARITY_MAX_LOGIT_ERR and nll_err <= PARITY_MAX_MEAN_NLL_ERR
    return {"classification": "PASS" if ok else "FAIL", "max_logit_abs_error": max_err,
            "mean_nll_abs_error": nll_err, "tolerances": [PARITY_MAX_LOGIT_ERR, PARITY_MAX_MEAN_NLL_ERR]}


def score_methods(np_model: GPT2Numpy, chunks, specs, far_start=FAR_START, log=print) -> dict:
    results = {}
    for name, spec in specs.items():
        t0 = time.time()
        far, allp = [], []
        for c in chunks:
            nll = token_nll(np_model.logits(c, spec), c)   # nll[i] predicts token i+1 from position i
            far.append(nll[far_start:])
            allp.append(nll)
        far_mean = [float(x.mean()) for x in far]
        pooled = float(np.concatenate(far).mean())
        results[name] = {
            "far_mean_nll": pooled,
            "far_perplexity": float(np.exp(pooled)),
            "all_perplexity": float(np.exp(np.concatenate(allp).mean())),
            "per_chunk_far_mean_nll": far_mean,
            "scalar_budget_per_head": spec.scalar_budget(np_model.wte.shape[1] // np_model.n_head),
        }
        log(f"  {name:32s} far ppl {results[name]['far_perplexity']:9.3f}  ({time.time() - t0:.0f}s)")
    return results


def questions(res: dict) -> dict:
    nll = {k: v["far_mean_nll"] for k, v in res.items()}
    per = {k: v["per_chunk_far_mean_nll"] for k, v in res.items()}
    wins = lambda a, b: int(sum(x < y for x, y in zip(per[a], per[b], strict=True)))

    def compare(a, b):
        gap = nll[b] - nll["full_cache"]
        return {
            "pass": bool(nll[a] < nll[b]),
            "far_perplexity": float(np.exp(nll[a])), "baseline_far_perplexity": float(np.exp(nll[b])),
            "full_cache_far_perplexity": float(np.exp(nll["full_cache"])),
            "chunk_wins": wins(a, b), "chunks": len(per[a]),
            "fraction_of_gap_to_full_cache_closed": float((nll[b] - nll[a]) / gap) if gap > 0 else None,
        }

    return {
        "primary_b48_lowpass_beats_sink_window": compare("b48_sink1_window39_lowpass", "b48_sink1_window47"),
        "secondary_b128_lowpass_beats_sink_window": compare("b128_sink1_window119_lowpass", "b128_sink1_window127"),
        "b48_band_beats_sink_window": compare("b48_sink1_window39_band", "b48_sink1_window47"),
        "b48_lowpass_beats_band": compare("b48_sink1_window39_lowpass", "b48_sink1_window39_band"),
        "reference_sink1_vs_no_sink": compare("b48_sink1_window47", "b48_window48"),
        "reference_sink1_vs_sink4": compare("b48_sink1_window47", "b48_sink4_window44"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/gpt2_perplexity.json"))
    args = ap.parse_args()
    model, tokenizer, meta = load_real_runtime()
    np_model = GPT2Numpy.from_hf(model)
    ids = np.asarray(tokenizer(load_text(), return_tensors=None)["input_ids"], dtype=np.int64)
    chunks = make_chunks(ids)
    receipt = {
        "gate": "gpt2_perplexity_all_heads",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "runtime": meta,
        "text": {"path": str(TEXT_PATH.name), "sha256": TEXT_SHA256, "tokens": int(ids.shape[0]),
                 "source": "karpathy/char-rnn tinyshakespeare input.txt, first 44,961 chars"},
        "frozen": {"chunk": CHUNK, "n_chunks": N_CHUNKS, "far_start": FAR_START, "taus": list(TAUS),
                   "specs": {k: {"window": s.window, "sink": s.sink, "levels": len(s.rhos), "read": s.read_mode}
                             for k, s in method_specs().items()}},
        "boundary": {"all_heads_all_layers_replaced": True, "changes_generation": False,
                     "measures_perplexity": True, "note": "teacher-forced perplexity, one text, GPT-2 small"},
    }
    print("parity check ...")
    receipt["parity"] = parity(model, np_model, chunks[0])
    print(receipt["parity"])
    if receipt["parity"]["classification"] != "PASS":
        receipt["classification"] = "FAIL_PARITY"
    else:
        res = score_methods(np_model, chunks, method_specs())
        receipt["methods"] = res
        receipt["questions"] = questions(res)
        receipt["classification"] = (
            "PASS_LOWPASS_LOWERS_PERPLEXITY_AT_EQUAL_BUDGET"
            if receipt["questions"]["primary_b48_lowpass_beats_sink_window"]["pass"]
            else "FAIL_SINK_WINDOW_WINS_PERPLEXITY")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {receipt['classification']}")


if __name__ == "__main__":
    main()
