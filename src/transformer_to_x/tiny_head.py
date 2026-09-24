from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadWeights:
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray

    def __post_init__(self) -> None:
        w_q = np.asarray(self.w_q, dtype=float)
        w_k = np.asarray(self.w_k, dtype=float)
        w_v = np.asarray(self.w_v, dtype=float)
        w_o = np.asarray(self.w_o, dtype=float)
        if w_q.ndim != 2:
            raise ValueError("w_q must be a 2D matrix")
        if w_k.shape != w_q.shape or w_v.shape != w_q.shape:
            raise ValueError("w_q, w_k, and w_v must have identical shapes")
        d_model, d_head = w_q.shape
        if w_o.shape != (d_head, d_model):
            raise ValueError("w_o must have shape (d_head, d_model)")
        for name, value in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_o", w_o)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        object.__setattr__(self, "w_q", w_q.copy())
        object.__setattr__(self, "w_k", w_k.copy())
        object.__setattr__(self, "w_v", w_v.copy())
        object.__setattr__(self, "w_o", w_o.copy())

    @property
    def d_model(self) -> int:
        return int(self.w_q.shape[0])

    @property
    def d_head(self) -> int:
        return int(self.w_q.shape[1])


@dataclass(frozen=True)
class HeadRun:
    logits: np.ndarray
    attention: np.ndarray
    output: np.ndarray


def causal_mask(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    return np.triu(np.ones((length, length), dtype=bool), k=1)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError("logits must be a 2D matrix")
    row_max = np.max(values, axis=1, keepdims=True)
    shifted = values - row_max
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    if np.any(denom <= 0.0) or not np.all(np.isfinite(denom)):
        raise ValueError("softmax rows must contain at least one finite logit")
    return exp / denom


@dataclass(frozen=True)
class TinyCausalHead:
    weights: HeadWeights

    @classmethod
    def deterministic(
        cls,
        *,
        seed: int,
        d_model: int,
        d_head: int,
    ) -> "TinyCausalHead":
        if d_model <= 0 or d_head <= 0:
            raise ValueError("d_model and d_head must be positive")
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(float(d_model))
        return cls(
            HeadWeights(
                w_q=rng.normal(scale=scale, size=(d_model, d_head)),
                w_k=rng.normal(scale=scale, size=(d_model, d_head)),
                w_v=rng.normal(scale=scale, size=(d_model, d_head)),
                w_o=rng.normal(scale=scale, size=(d_head, d_model)),
            )
        )

    def reference(self, x: np.ndarray) -> HeadRun:
        tokens = np.asarray(x, dtype=float)
        if tokens.ndim != 2 or tokens.shape[1] != self.weights.d_model:
            raise ValueError("x must have shape (sequence, d_model)")
        if tokens.shape[0] <= 0:
            raise ValueError("sequence must be non-empty")
        if not np.all(np.isfinite(tokens)):
            raise ValueError("x must contain only finite values")

        q = tokens @ self.weights.w_q
        k = tokens @ self.weights.w_k
        v = tokens @ self.weights.w_v
        logits = (q @ k.T) / np.sqrt(float(self.weights.d_head))
        logits = logits.copy()
        logits[causal_mask(tokens.shape[0])] = -np.inf
        attention = stable_softmax(logits)
        output = attention @ v @ self.weights.w_o
        return HeadRun(logits=logits, attention=attention, output=output)
