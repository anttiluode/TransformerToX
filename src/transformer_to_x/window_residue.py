"""Exact KV window + temporal residue stack, stored in head (K/V) space.

This is the "Sihti in time" memory. A real KV cache costs 2*d_head scalars per
token. Under a fixed scalar budget we split storage between

* an exact window of the last W keys/values (the "present"), and
* L leaky integrator levels that absorb only the tokens EVICTED from the window
  (what bled out of the present).

Level l keeps key/value traces S_l = rho_l * S_l + k_evicted and a mass
m_l = rho_l * m_l + 1, with rho_1 < rho_2 < ... < rho_L.

Two read coordinates over the SAME stored state:

band    (residue coordinates, the Sihti move): band 1 is level 1, band l is
        level l minus level l-1. Because rho_l > rho_(l-1), every band weights
        each past token by rho_l^a - rho_(l-1)^a >= 0, i.e. each band is a
        genuine time window at an intermediate lag. Bands telescope: their sum
        is exactly the slowest level, so they PARTITION the evicted past.
lowpass (accumulated coordinates): each nested level is read on its own, so an
        old token is represented inside several pseudo-tokens at once.

Each band/level becomes one pseudo-token with mean key k = S_k / mass, mean
value v = S_v / mass and attention logit  scale * q.k + log(mass). That is the
first-order (Jensen) summary of sum_j w_j exp(scale q.k_j); it is exact when
all tokens in the band share a key. Window tokens are read with ordinary exact
attention; all logits share one softmax.
"""

from __future__ import annotations

import numpy as np

_MASS_FLOOR = 1e-12


def _positive_int(value, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return vector


def rhos_from_taus(taus) -> tuple[float, ...]:
    values = tuple(float(t) for t in taus)
    if not values or any(not np.isfinite(t) or t < 1.0 for t in values):
        raise ValueError("taus must be finite and >= 1")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("taus must be strictly increasing")
    return tuple(1.0 - 1.0 / t for t in values)


def _softmax_mix(logits: np.ndarray, values: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return weights @ values


class WindowResidueMemory:
    """Exact last-W K/V window plus L leaky levels over evicted tokens.

    read_mode: "band" | "lowpass" | "window_only". window_only stores the same
    levels but never reads them (ablation: does the stack add anything?).
    levels may be empty (rhos=()), which gives a plain sliding-window cache.
    """

    READ_MODES = ("band", "lowpass", "window_only")

    def __init__(
        self,
        *,
        d_head: int,
        window: int,
        rhos: tuple[float, ...] = (),
        read_mode: str = "band",
        sink: int = 0,
    ) -> None:
        self.d_head = _positive_int(d_head, "d_head")
        self.window = _positive_int(window, "window")
        # sink: the first `sink` tokens of the sequence are kept exactly and never
        # evicted (StreamingLLM-style). They cost 2*d_head scalars each.
        self.sink = _positive_int(sink, "sink", allow_zero=True)
        if read_mode not in self.READ_MODES:
            raise ValueError(f"read_mode must be one of {self.READ_MODES}")
        values = np.asarray(tuple(rhos), dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("rhos must be a finite 1D sequence")
        if np.any(values < 0.0) or np.any(values >= 1.0):
            raise ValueError("each rho must satisfy 0 <= rho < 1")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("rhos must be strictly increasing (fast to slow)")
        self.rhos = tuple(float(v) for v in values)
        self.read_mode = read_mode
        self._rho = values[:, None]
        self.reset()

    @property
    def levels(self) -> int:
        return len(self.rhos)

    @property
    def scalar_state_budget(self) -> int:
        # window: W keys + W values; each level: key trace + value trace + mass.
        return (self.sink + self.window) * 2 * self.d_head + self.levels * (2 * self.d_head + 1)

    def reset(self) -> None:
        self._sink_k: list[np.ndarray] = []
        self._sink_v: list[np.ndarray] = []
        self._win_k: list[np.ndarray] = []
        self._win_v: list[np.ndarray] = []
        self._sk = np.zeros((self.levels, self.d_head))
        self._sv = np.zeros((self.levels, self.d_head))
        self._mass = np.zeros(self.levels)

    def update(self, k_t, v_t) -> None:
        k = _finite_vector(k_t, self.d_head, "k_t")
        v = _finite_vector(v_t, self.d_head, "v_t")
        if len(self._sink_k) < self.sink:
            self._sink_k.append(k.copy())
            self._sink_v.append(v.copy())
            return
        self._win_k.append(k.copy())
        self._win_v.append(v.copy())
        if len(self._win_k) > self.window:
            old_k = self._win_k.pop(0)
            old_v = self._win_v.pop(0)
            if self.levels:
                self._sk = self._rho * self._sk + old_k[None, :]
                self._sv = self._rho * self._sv + old_v[None, :]
                self._mass = self._rho[:, 0] * self._mass + 1.0

    def resident_tokens(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pseudo-token (keys, values, masses) exposed to the reader."""
        empty = (np.zeros((0, self.d_head)), np.zeros((0, self.d_head)), np.zeros(0))
        if self.levels == 0 or self.read_mode == "window_only":
            return empty
        if self.read_mode == "lowpass":
            sk, sv, mass = self._sk, self._sv, self._mass
        else:
            sk = np.diff(self._sk, axis=0, prepend=np.zeros((1, self.d_head)))
            sv = np.diff(self._sv, axis=0, prepend=np.zeros((1, self.d_head)))
            mass = np.diff(self._mass, prepend=0.0)
        keep = mass > _MASS_FLOOR
        if not np.any(keep):
            return empty
        return sk[keep] / mass[keep, None], sv[keep] / mass[keep, None], mass[keep]

    def band_masses(self) -> np.ndarray:
        return np.diff(self._mass, prepend=0.0)

    def read(self, q_t, score_scale: float) -> np.ndarray:
        q = _finite_vector(q_t, self.d_head, "q_t")
        scale = float(score_scale)
        keys = np.asarray(self._sink_k + self._win_k).reshape(-1, self.d_head)
        values = np.asarray(self._sink_v + self._win_v).reshape(-1, self.d_head)
        logits = scale * (keys @ q)
        rk, rv, rm = self.resident_tokens()
        if rm.size:
            keys = np.vstack([keys, rk])
            values = np.vstack([values, rv])
            logits = np.concatenate([logits, scale * (rk @ q) + np.log(rm)])
        if logits.size == 0:
            return np.zeros(self.d_head)
        return _softmax_mix(logits, values)


class ZeroMemory:
    """Predicts a zero head contribution. Relative RMSE is exactly 1."""

    def __init__(self, *, d_head: int) -> None:
        self.d_head = _positive_int(d_head, "d_head")

    scalar_state_budget = 0

    def reset(self) -> None:
        pass

    def update(self, k_t, v_t) -> None:
        pass

    def read(self, q_t, score_scale: float) -> np.ndarray:
        return np.zeros(self.d_head)


def run_head_space_context(geometry, x: np.ndarray, memory) -> np.ndarray:
    """Stream one context through a head-space memory; return (T, d_model) contributions."""
    tokens = np.asarray(x, dtype=float)
    if tokens.ndim != 2 or tokens.shape[1] != geometry.d_model or tokens.shape[0] == 0:
        raise ValueError("x must have shape (nonempty_sequence, d_model)")
    q = tokens @ geometry.w_q + geometry.b_q
    k = tokens @ geometry.w_k + geometry.b_k
    v = tokens @ geometry.w_v + geometry.b_v
    memory.reset()
    mixtures = []
    for t in range(tokens.shape[0]):
        memory.update(k[t], v[t])
        mixtures.append(memory.read(q[t], geometry.score_scale))
    return np.asarray(mixtures) @ geometry.w_o
