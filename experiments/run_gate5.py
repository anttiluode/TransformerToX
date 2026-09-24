"""Gate 5: does the lowpass memory survive stronger baselines, new texts and a RoPE model?

Pre-registered in docs/superpowers/specs/2026-09-24-robustness-gate.md BEFORE
the first real score. One run per model:

    python -m experiments.run_gate5 --model gpt2        --out results/gate5_gpt2.json
    python -m experiments.run_gate5 --model pythia-160m --out results/gate5_pythia160m.json
    python -m experiments.run_gate5 --model gpt2-medium --out results/gate5_gpt2medium.json

Nothing is tuned. Memory layouts, tau ladder and budgets are frozen from Gate 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from transformer_to_x.cache_policies import PolicySpec
from transformer_to_x.gpt2_numpy import GPT2Numpy, MemorySpec, token_nll
from transformer_to_x.window_residue import rhos_from_taus

DATA = Path(__file__).with_name("data")

# ---- frozen before the first real score -------------------------------------
MODELS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "gpt2-medium": ("openai-community/gpt2-medium", None),   # revision recorded at run time
    "pythia-160m": ("EleutherAI/pythia-160m", "step143000"),
}
TEXTS = {
    "shakespeare": ("tinyshakespeare_head.txt", "3f0590676115fc4d4c2edd6466f021f46b5bf5d4acd77cb63fdcc338ac3cacef"),
    "python313_docs": ("python313_whatsnew_excerpt.txt", "2257dd709e98b75987eeccaf23be0ee661edd3c9e7e439ef6dcd85eef628bca3"),
    "unseen_stories": ("unseen_stories.txt", "ec84a62ca314655cd195f6b40600093b137168df315a1979d057de5d74b22de1"),
}
CHUNK = 1024
MAX_CHUNKS_PER_TEXT = 6
FAR_START = 48
TAUS = (8, 16, 32, 64, 128, 256, 512)
PARITY_MAX_LOGIT_ERR = 0.1
PARITY_MAX_MEAN_NLL_ERR = 1e-3
# -----------------------------------------------------------------------------


def row_specs():
    r = rhos_from_taus(TAUS)
    return {
        "full_cache": MemorySpec(),
        "b48_sink_window": MemorySpec(window=47, sink=1),
        "b48_lowpass": MemorySpec(window=39, sink=1, rhos=r, read_mode="lowpass"),
        "b48_h2o": PolicySpec("h2o", 48),
        "b48_merge": PolicySpec("merge", 47),
        "b128_sink_window": MemorySpec(window=127, sink=1),
        "b128_lowpass": MemorySpec(window=119, sink=1, rhos=r, read_mode="lowpass"),
        "b128_h2o": PolicySpec("h2o", 128),
        "b128_merge": PolicySpec("merge", 127),
    }


def load_texts() -> dict[str, str]:
    out = {}
    for name, (fname, sha) in TEXTS.items():
        text = (DATA / fname).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != sha:
            raise ValueError(f"text fixture changed: {fname}")
        out[name] = text
    return out


def chunk_ids(ids: np.ndarray) -> list[np.ndarray]:
    n = min(MAX_CHUNKS_PER_TEXT, ids.shape[0] // CHUNK)
    if n < 1:
        raise ValueError("text shorter than one chunk")
    return [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(n)]


def numpy_model(hf_model):
    arch = type(hf_model).__name__
    if arch == "GPT2LMHeadModel":
        return GPT2Numpy.from_hf(hf_model)
    if arch == "GPTNeoXForCausalLM":
        from transformer_to_x.neox_numpy import NeoXNumpy
        return NeoXNumpy.from_hf(hf_model)
    raise ValueError(f"unsupported architecture {arch}")


def d_head_of(np_model) -> int:
    width = np_model.wte.shape[1] if hasattr(np_model, "wte") else np_model.embed.shape[1]
    return width // np_model.n_head


def parity(hf_model, np_model, ids) -> dict:
    import torch

    with torch.no_grad():
        hf = hf_model(input_ids=torch.tensor(ids[None]), use_cache=False).logits[0].double().numpy()
    ours = np_model.logits(ids)
    max_err = float(np.abs(hf - ours).max())
    nll_err = float(abs(token_nll(hf, ids).mean() - token_nll(ours, ids).mean()))
    ok = max_err <= PARITY_MAX_LOGIT_ERR and nll_err <= PARITY_MAX_MEAN_NLL_ERR
    return {"classification": "PASS" if ok else "FAIL", "max_logit_abs_error": max_err,
            "mean_nll_abs_error": nll_err, "tolerances": [PARITY_MAX_LOGIT_ERR, PARITY_MAX_MEAN_NLL_ERR]}


def score(np_model, chunks_by_text, specs, far_start=FAR_START, log=print) -> dict:
    """Returns {row: {text: [per-chunk far NLL arrays]}} summarised."""
    res = {}
    for row, spec in specs.items():
        t0 = time.time()
        per_text = {}
        for text, chunks in chunks_by_text.items():
            per_text[text] = [token_nll(np_model.logits(c, spec), c)[far_start:] for c in chunks]
        pooled = np.concatenate([a for arrs in per_text.values() for a in arrs])
        res[row] = {
            "pooled_far_nll": float(pooled.mean()),
            "pooled_far_perplexity": float(np.exp(pooled.mean())),
            "per_text": {t: {"far_perplexity": float(np.exp(np.concatenate(a).mean())),
                             "per_chunk_far_nll": [float(x.mean()) for x in a]}
                         for t, a in per_text.items()},
            "scalar_budget_per_head": spec.scalar_budget(d_head_of(np_model)),
        }
        log(f"  {row:18s} pooled far ppl {res[row]['pooled_far_perplexity']:9.3f}  ({time.time() - t0:.0f}s)")
    return res


def compare(res, a, b) -> dict:
    chunks_a = [x for t in res[a]["per_text"].values() for x in t["per_chunk_far_nll"]]
    chunks_b = [x for t in res[b]["per_text"].values() for x in t["per_chunk_far_nll"]]
    full = res["full_cache"]["pooled_far_nll"]
    gap = res[b]["pooled_far_nll"] - full
    return {
        "pass": bool(res[a]["pooled_far_nll"] < res[b]["pooled_far_nll"]),
        "perplexity": res[a]["pooled_far_perplexity"], "baseline_perplexity": res[b]["pooled_far_perplexity"],
        "chunk_wins": int(sum(x < y for x, y in zip(chunks_a, chunks_b, strict=True))), "chunks": len(chunks_a),
        "text_wins": {t: bool(res[a]["per_text"][t]["far_perplexity"] < res[b]["per_text"][t]["far_perplexity"])
                      for t in res[a]["per_text"]},
        "fraction_of_baseline_gap_closed": float((res[b]["pooled_far_nll"] - res[a]["pooled_far_nll"]) / gap)
        if gap > 0 else None,
    }


def questions(res) -> dict:
    q = {}
    for b in ("b48", "b128"):
        for base in ("sink_window", "h2o", "merge"):
            q[f"{b}_lowpass_vs_{base}"] = compare(res, f"{b}_lowpass", f"{b}_{base}")
    return q


def classify(q) -> str:
    beats = q["b48_lowpass_vs_h2o"]["pass"] and q["b48_lowpass_vs_merge"]["pass"]
    return "PASS_LOWPASS_BEATS_H2O_AND_MERGE_AT_B48" if beats else "FAIL_LOWPASS_LOSES_TO_A_STRONGER_BASELINE_AT_B48"


def load_hf(key):
    import torch  # noqa: F401
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, rev = MODELS[key]
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    meta = {"model_id": model_id, "requested_revision": rev,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "architecture": type(model).__name__}
    return model, tok, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="gpt2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hf, tok, meta = load_hf(args.model)
    npm = numpy_model(hf)
    texts = load_texts()
    chunks = {name: chunk_ids(np.asarray(tok(t)["input_ids"], dtype=np.int64)) for name, t in texts.items()}
    receipt = {
        "gate": "gate5_robustness", "model_key": args.model, "runtime": meta,
        "texts": {n: {"file": TEXTS[n][0], "sha256": TEXTS[n][1], "chunks": len(c)} for n, c in chunks.items()},
        "frozen": {"chunk": CHUNK, "max_chunks_per_text": MAX_CHUNKS_PER_TEXT, "far_start": FAR_START,
                   "taus": list(TAUS), "rows": {k: repr(v) for k, v in row_specs().items()}},
        "boundary": {"all_heads_all_layers_replaced": True, "teacher_forced_only": True,
                     "merge_is_our_implementation_not_CaM_or_D2O": True},
    }
    print(f"{args.model}: chunks per text {[len(c) for c in chunks.values()]}; parity check ...")
    first = next(iter(chunks.values()))[0]
    receipt["parity"] = parity(hf, npm, first)
    print(receipt["parity"])
    if receipt["parity"]["classification"] != "PASS":
        receipt["classification"] = "FAIL_PARITY"
    else:
        res = score(npm, chunks, row_specs())
        receipt["rows"] = res
        receipt["questions"] = questions(res)
        receipt["classification"] = classify(receipt["questions"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {receipt['classification']}")


if __name__ == "__main__":
    main()
