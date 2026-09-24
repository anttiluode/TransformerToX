from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiny_head import HeadRun, HeadWeights, causal_mask, stable_softmax


@dataclass(frozen=True)
class CompiledHead:
    address_operator: np.ndarray
    write_operator: np.ndarray
    d_model: int
    d_head: int

    @classmethod
    def from_weights(cls, weights: HeadWeights) -> "CompiledHead":
        return cls(
            address_operator=weights.w_q @ weights.w_k.T,
            write_operator=weights.w_v @ weights.w_o,
            d_model=weights.d_model,
            d_head=weights.d_head,
        )

    def __post_init__(self) -> None:
        address = np.asarray(self.address_operator, dtype=float)
        write = np.asarray(self.write_operator, dtype=float)
        expected = (self.d_model, self.d_model)
        if address.shape != expected or write.shape != expected:
            raise ValueError("compiled operators must both have shape (d_model, d_model)")
        if not np.all(np.isfinite(address)) or not np.all(np.isfinite(write)):
            raise ValueError("compiled operators must contain only finite values")
        object.__setattr__(self, "address_operator", address.copy())
        object.__setattr__(self, "write_operator", write.copy())

    def run(self, x: np.ndarray) -> HeadRun:
        tokens = np.asarray(x, dtype=float)
        if tokens.ndim != 2 or tokens.shape[1] != self.d_model:
            raise ValueError("x must have shape (sequence, d_model)")
        if tokens.shape[0] <= 0:
            raise ValueError("sequence must be non-empty")
        if not np.all(np.isfinite(tokens)):
            raise ValueError("x must contain only finite values")

        logits = (tokens @ self.address_operator @ tokens.T) / np.sqrt(float(self.d_head))
        logits = logits.copy()
        logits[causal_mask(tokens.shape[0])] = -np.inf
        attention = stable_softmax(logits)
        output = attention @ (tokens @ self.write_operator)
        return HeadRun(logits=logits, attention=attention, output=output)
