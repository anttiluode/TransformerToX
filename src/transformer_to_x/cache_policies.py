"""Sequential cache policies used as stronger baselines in Gate 5.

h2o    Heavy-Hitter Oracle (Zhang et al. 2023), online form. B slots, each an
       exact past token. The most recent R = B // 2 tokens are always kept; the
       rest are the tokens with the largest accumulated attention received so
       far. When a new token arrives with the cache full, the non-recent slot
       with the smallest accumulated score is evicted first, then the query
       attends over the B kept slots. Budget B * 2d.
merge  Our implementation of the "merge instead of drop" family (CaM, D2O):
       same H2O layout and victim choice, but the victim is folded into the
       non-recent slot with the most similar key (cosine), as a mass-weighted
       average of keys and values. Every slot carries a mass m and is read with
       logit + log m (the same pseudo-token rule as the lowpass summaries).
       Budget B * (2d + 1). This is NOT a faithful reproduction of CaM or D2O.

Both are per-head and causal; the query at t sees token t itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    policy: str          # "h2o" | "merge"
    slots: int
    recent: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in ("h2o", "merge"):
            raise ValueError("policy must be h2o or merge")
        if int(self.slots) < 2:
            raise ValueError("slots must be >= 2")
        r = self.recent_size
        if not 1 <= r < self.slots:
            raise ValueError("recent must be in [1, slots)")

    @property
    def recent_size(self) -> int:
        return int(self.recent) if self.recent is not None else int(self.slots) // 2

    def scalar_budget(self, d_head: int) -> int:
        per = 2 * d_head + (1 if self.policy == "merge" else 0)
        return int(self.slots) * per


def policy_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, spec: PolicySpec) -> np.ndarray:
    H, T, d = q.shape
    B, R = int(spec.slots), spec.recent_size
    merge = spec.policy == "merge"
    hi = np.arange(H)
    K = np.zeros((H, B, d))
    V = np.zeros((H, B, d))
    pos = np.full((H, B), -1, dtype=np.int64)       # -1 = empty
    score = np.zeros((H, B))
    mass = np.zeros((H, B))
    out = np.zeros((H, T, d))
    n = 0
    for t in range(T):
        if n < B:
            slot = np.full(H, n)
            n += 1
        else:
            # victim: non-recent slot with the smallest accumulated score
            nonrecent = pos <= t - R                    # recent = last R-1 kept + new token t
            cand = np.where(nonrecent, score, np.inf)
            slot = np.argmin(cand, axis=1)
            if merge:
                vk, vv = K[hi, slot], V[hi, slot]
                vm, vs = mass[hi, slot], score[hi, slot]
                kn = K / (np.linalg.norm(K, axis=2, keepdims=True) + 1e-12)
                sim = np.einsum("hbd,hd->hb", kn, vk / (np.linalg.norm(vk, axis=1, keepdims=True) + 1e-12))
                ok = nonrecent.copy()
                ok[hi, slot] = False
                sim = np.where(ok, sim, -np.inf)
                tgt = np.argmax(sim, axis=1)
                has = np.isfinite(sim[hi, tgt])
                tm = mass[hi, tgt]
                tot = tm + vm
                w_old = np.where(has, tm / np.where(has, tot, 1.0), 1.0)[:, None]
                w_new = 1.0 - w_old
                K[hi, tgt] = np.where(has[:, None], w_old * K[hi, tgt] + w_new * vk, K[hi, tgt])
                V[hi, tgt] = np.where(has[:, None], w_old * V[hi, tgt] + w_new * vv, V[hi, tgt])
                mass[hi, tgt] = np.where(has, tot, mass[hi, tgt])
                score[hi, tgt] = np.where(has, score[hi, tgt] + vs, score[hi, tgt])
        K[hi, slot] = k[:, t]
        V[hi, slot] = v[:, t]
        pos[hi, slot] = t
        score[hi, slot] = 0.0
        mass[hi, slot] = 1.0
        valid = pos >= 0
        logits = scale * np.einsum("hbd,hd->hb", K, q[:, t])
        if merge:
            logits = logits + np.log(np.where(valid, mass, 1.0))
        logits = np.where(valid, logits, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        score += p
        out[:, t] = np.einsum("hb,hbd->hd", p, V)
    return out
