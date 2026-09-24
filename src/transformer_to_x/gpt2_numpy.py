"""NumPy GPT-2 forward pass with pluggable, vectorised per-head memory attention.

Used by Gate 4 (perplexity). Every attention head in every layer is replaced by
the same bounded memory layout used in Gates 2-3:

    exact sink tokens  +  exact sliding window  +  L leaky summary levels
    fed only by evicted tokens, read as lowpass or band pseudo-tokens.

The vectorised attention here must equal the step-by-step
``WindowResidueMemory`` reference; tests check that directly.

Positions: GPT-2 uses learned absolute position embeddings added at the input,
so no position re-indexing is needed when tokens are evicted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemorySpec:
    """Per-head memory layout. window=None means the full causal cache."""

    window: int | None = None
    sink: int = 0
    rhos: tuple[float, ...] = ()
    read_mode: str = "lowpass"  # "lowpass" | "band" | "window_only"

    def __post_init__(self) -> None:
        if self.window is not None and int(self.window) <= 0:
            raise ValueError("window must be positive or None")
        if int(self.sink) < 0:
            raise ValueError("sink must be non-negative")
        if self.read_mode not in ("lowpass", "band", "window_only"):
            raise ValueError("bad read_mode")
        r = np.asarray(self.rhos, dtype=float)
        if r.size and (np.any(r < 0) or np.any(r >= 1) or np.any(np.diff(r) <= 0)):
            raise ValueError("rhos must be strictly increasing in [0, 1)")
        if self.window is None and (self.sink or r.size):
            raise ValueError("full cache takes no sink or levels")

    def scalar_budget(self, d_head: int) -> int | None:
        if self.window is None:
            return None
        return (self.sink + self.window) * 2 * d_head + len(self.rhos) * (2 * d_head + 1)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def memory_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, spec: MemorySpec) -> np.ndarray:
    """q, k, v: (H, T, d). Returns the per-head mixture (H, T, d)."""
    H, T, d = q.shape
    idx = np.arange(T)
    t_col, j_row = idx[:, None], idx[None, :]
    causal = j_row <= t_col
    if spec.window is None:
        exact = causal
    else:
        # Mirrors WindowResidueMemory: the first `sink` tokens are pinned; the
        # window holds the last `window` tokens AFTER the sink.
        in_sink = j_row < spec.sink
        in_window = (j_row >= spec.sink) & (j_row > t_col - spec.window)
        exact = causal & (in_sink | in_window)

    logits = scale * np.einsum("htd,hjd->htj", q, k)
    logits = np.where(exact[None], logits, -np.inf)

    L = len(spec.rhos)
    if spec.window is None or L == 0 or spec.read_mode == "window_only":
        return _softmax_rows(logits) @ v

    # Leaky levels over evicted tokens. Token j (j >= sink) is evicted at step
    # t = j + window and then decays by rho each step.
    rho = np.asarray(spec.rhos, dtype=float)
    Sk = np.zeros((T, L, H, d))
    Sv = np.zeros((T, L, H, d))
    mass = np.zeros((T, L))
    sk = np.zeros((L, H, d))
    sv = np.zeros((L, H, d))
    m = np.zeros(L)
    for t in range(T):
        j = t - spec.window
        if j >= spec.sink:
            sk = rho[:, None, None] * sk + k[None, :, j, :]
            sv = rho[:, None, None] * sv + v[None, :, j, :]
            m = rho * m + 1.0
        Sk[t], Sv[t], mass[t] = sk, sv, m
    if spec.read_mode == "band":
        Sk = np.diff(Sk, axis=1, prepend=np.zeros((T, 1, H, d)))
        Sv = np.diff(Sv, axis=1, prepend=np.zeros((T, 1, H, d)))
        mass = np.diff(mass, axis=1, prepend=np.zeros((T, 1)))
    active = mass > 1e-12                                     # (T, L)
    safe = np.where(active, mass, 1.0)
    kbar = Sk / safe[:, :, None, None]                        # (T, L, H, d)
    vbar = Sv / safe[:, :, None, None]
    plog = scale * np.einsum("htd,tlhd->htl", q, kbar) + np.log(safe)[None]
    plog = np.where(active[None], plog, -np.inf)

    allw = _softmax_rows(np.concatenate([logits, plog], axis=-1))
    w_exact, w_pseudo = allw[..., :T], allw[..., T:]
    return w_exact @ v + np.einsum("htl,tlhd->htd", w_pseudo, vbar)


def attend(q, k, v, scale, spec):
    """Dispatch to the bounded-memory attention or a sequential cache policy."""
    from .cache_policies import PolicySpec, policy_attention

    if isinstance(spec, PolicySpec):
        return policy_attention(q, k, v, scale, spec)
    return memory_attention(q, k, v, scale, spec)


@dataclass
class GPT2Numpy:
    wte: np.ndarray
    wpe: np.ndarray
    blocks: list[dict[str, np.ndarray]]
    ln_f_w: np.ndarray
    ln_f_b: np.ndarray
    n_head: int
    eps: float = 1e-5

    @classmethod
    def from_hf(cls, model) -> "GPT2Numpy":
        cfg = model.config
        if getattr(cfg, "scale_attn_by_inverse_layer_idx", False) or getattr(cfg, "reorder_and_upcast_attn", False):
            raise ValueError("unsupported GPT-2 attention variant")
        if cfg.activation_function != "gelu_new":
            raise ValueError("expected gelu_new")
        f = lambda t: t.detach().cpu().double().numpy()
        tr = model.transformer
        blocks = []
        for b in tr.h:
            blocks.append({
                "ln1_w": f(b.ln_1.weight), "ln1_b": f(b.ln_1.bias),
                "attn_w": f(b.attn.c_attn.weight), "attn_b": f(b.attn.c_attn.bias),
                "proj_w": f(b.attn.c_proj.weight), "proj_b": f(b.attn.c_proj.bias),
                "ln2_w": f(b.ln_2.weight), "ln2_b": f(b.ln_2.bias),
                "fc_w": f(b.mlp.c_fc.weight), "fc_b": f(b.mlp.c_fc.bias),
                "mproj_w": f(b.mlp.c_proj.weight), "mproj_b": f(b.mlp.c_proj.bias),
            })
        return cls(wte=f(tr.wte.weight), wpe=f(tr.wpe.weight), blocks=blocks,
                   ln_f_w=f(tr.ln_f.weight), ln_f_b=f(tr.ln_f.bias),
                   n_head=int(cfg.n_head), eps=float(cfg.layer_norm_epsilon))

    def _ln(self, x, w, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * w + b

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def logits(self, ids, spec=MemorySpec()) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        T = ids.shape[0]
        if T > self.wpe.shape[0]:
            raise ValueError("sequence longer than position table")
        x = self.wte[ids] + self.wpe[:T]
        D = x.shape[1]
        H = self.n_head
        dh = D // H
        scale = 1.0 / np.sqrt(dh)
        for b in self.blocks:
            h = self._ln(x, b["ln1_w"], b["ln1_b"])
            qkv = h @ b["attn_w"] + b["attn_b"]
            q, k, v = (qkv[:, i * D:(i + 1) * D].reshape(T, H, dh).transpose(1, 0, 2) for i in range(3))
            mix = attend(q, k, v, scale, spec)
            x = x + mix.transpose(1, 0, 2).reshape(T, D) @ b["proj_w"] + b["proj_b"]
            h = self._ln(x, b["ln2_w"], b["ln2_b"])
            x = x + self._gelu(h @ b["fc_w"] + b["fc_b"]) @ b["mproj_w"] + b["mproj_b"]
        return self._ln(x, self.ln_f_w, self.ln_f_b) @ self.wte.T


def token_nll(logits: np.ndarray, ids) -> np.ndarray:
    """Negative log-likelihood of ids[1:] given logits[:-1]. Shape (T-1,)."""
    ids = np.asarray(ids)
    lg = logits[:-1]
    m = lg.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(lg - m).sum(-1, keepdims=True)))[:, 0]
    return lse - lg[np.arange(lg.shape[0]), ids[1:]]
