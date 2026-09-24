"""NumPy GPT-NeoX (Pythia) forward pass: rotary positions, parallel residual.

Keys are rotated at their own position before they enter any memory, so the
bounded memories and cache policies see exactly what a real RoPE KV cache holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gpt2_numpy import MemorySpec, attend


def _erf(x):
    try:
        from scipy.special import erf
    except ImportError:  # pragma: no cover
        import math
        return np.vectorize(math.erf)(x)
    return erf(x)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rope_cos_sin(T: int, rot_dim: int, base: float):
    inv = 1.0 / (base ** (np.arange(0, rot_dim, 2, dtype=np.float64) / rot_dim))
    freqs = np.outer(np.arange(T, dtype=np.float64), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def apply_partial_rope(x, cos, sin, rot_dim):
    """x: (H, T, d). Rotates the first rot_dim dims (HF rotate_half convention)."""
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    xr = xr * cos[None] + rotate_half(xr) * sin[None]
    return np.concatenate([xr, xp], axis=-1)


@dataclass
class NeoXNumpy:
    embed: np.ndarray
    blocks: list[dict[str, np.ndarray]]
    ln_f_w: np.ndarray
    ln_f_b: np.ndarray
    unembed: np.ndarray          # (V, D)
    n_head: int
    rot_dim: int
    rope_base: float
    eps: float
    parallel: bool = True

    @classmethod
    def from_hf(cls, model) -> "NeoXNumpy":
        cfg = model.config
        if getattr(cfg, "hidden_act", "gelu") != "gelu":
            raise ValueError("expected exact gelu")
        if getattr(cfg, "rope_scaling", None):
            raise ValueError("rope scaling unsupported")
        f = lambda t: t.detach().cpu().double().numpy()
        nx = model.gpt_neox
        blocks = []
        for b in nx.layers:
            a = b.attention
            blocks.append({
                "ln1_w": f(b.input_layernorm.weight), "ln1_b": f(b.input_layernorm.bias),
                "ln2_w": f(b.post_attention_layernorm.weight), "ln2_b": f(b.post_attention_layernorm.bias),
                "qkv_w": f(a.query_key_value.weight), "qkv_b": f(a.query_key_value.bias),
                "o_w": f(a.dense.weight), "o_b": f(a.dense.bias),
                "up_w": f(b.mlp.dense_h_to_4h.weight), "up_b": f(b.mlp.dense_h_to_4h.bias),
                "down_w": f(b.mlp.dense_4h_to_h.weight), "down_b": f(b.mlp.dense_4h_to_h.bias),
            })
        d_head = cfg.hidden_size // cfg.num_attention_heads
        pct = getattr(cfg, "partial_rotary_factor", None) or getattr(cfg, "rotary_pct", 1.0)
        base = getattr(cfg, "rope_theta", None) or getattr(cfg, "rotary_emb_base", 10000)
        return cls(embed=f(nx.embed_in.weight), blocks=blocks,
                   ln_f_w=f(nx.final_layer_norm.weight), ln_f_b=f(nx.final_layer_norm.bias),
                   unembed=f(model.embed_out.weight), n_head=int(cfg.num_attention_heads),
                   rot_dim=int(d_head * pct), rope_base=float(base), eps=float(cfg.layer_norm_eps),
                   parallel=bool(getattr(cfg, "use_parallel_residual", True)))

    def _ln(self, x, w, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * w + b

    def logits(self, ids, spec=MemorySpec()) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        T = ids.shape[0]
        x = self.embed[ids]
        D = x.shape[1]
        H = self.n_head
        dh = D // H
        cos, sin = rope_cos_sin(T, self.rot_dim, self.rope_base)
        scale = 1.0 / np.sqrt(dh)
        for b in self.blocks:
            h = self._ln(x, b["ln1_w"], b["ln1_b"])
            qkv = (h @ b["qkv_w"].T + b["qkv_b"]).reshape(T, H, 3 * dh).transpose(1, 0, 2)
            q, k, v = qkv[..., :dh], qkv[..., dh:2 * dh], qkv[..., 2 * dh:]
            q = apply_partial_rope(q, cos, sin, self.rot_dim)
            k = apply_partial_rope(k, cos, sin, self.rot_dim)
            mix = attend(q, k, v, scale, spec)
            attn_out = mix.transpose(1, 0, 2).reshape(T, D) @ b["o_w"].T + b["o_b"]
            mlp_in = self._ln(x, b["ln2_w"], b["ln2_b"]) if self.parallel else None
            if not self.parallel:
                x = x + attn_out
                mlp_in = self._ln(x, b["ln2_w"], b["ln2_b"])
            u = mlp_in @ b["up_w"].T + b["up_b"]
            mlp_out = (0.5 * u * (1.0 + _erf(u / np.sqrt(2.0)))) @ b["down_w"].T + b["down_b"]
            x = x + (attn_out + mlp_out if self.parallel else mlp_out)
        return self._ln(x, self.ln_f_w, self.ln_f_b) @ self.unembed.T
