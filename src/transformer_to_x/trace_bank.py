from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _validate_d_model(d_model: int) -> int:
    if not isinstance(d_model, (int, np.integer)) or int(d_model) <= 0:
        raise ValueError("d_model must be a positive integer")
    return int(d_model)


def _validate_vector(value: np.ndarray, *, d_model: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (d_model,):
        raise ValueError(f"{name} must have shape ({d_model},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalized_pseudo_event_read(components: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Read any fixed-state memory through one shared content-addressed summary.

    Each state component is treated as a pseudo-event. Cosine-like scores choose
    components by query direction, then the original (unnormalized) component
    vectors are mixed. Empty components are ignored.
    """

    state = np.asarray(components, dtype=float)
    if state.ndim != 2:
        raise ValueError("components must be a 2D matrix")
    q = _validate_vector(query, d_model=state.shape[1], name="query")

    norms = np.linalg.norm(state, axis=1)
    active = norms > 0.0
    if not np.any(active):
        return np.zeros(state.shape[1], dtype=float)

    unit_state = np.zeros_like(state)
    unit_state[active] = state[active] / norms[active, None]

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        scores = np.zeros(state.shape[0], dtype=float)
    else:
        scores = unit_state @ (q / q_norm)
    scores = scores.astype(float, copy=False)
    scores[~active] = -np.inf

    finite_scores = scores[active]
    shifted = finite_scores - np.max(finite_scores)
    active_weights = np.exp(shifted)
    active_weights /= np.sum(active_weights)

    weights = np.zeros(state.shape[0], dtype=float)
    weights[active] = active_weights
    return weights @ state


class StaticMemory:
    """Order-invariant matched-budget attacker with deterministic content routing."""

    def __init__(self, *, d_model: int, state_slots: int) -> None:
        self.d_model = _validate_d_model(d_model)
        if not isinstance(state_slots, (int, np.integer)) or int(state_slots) <= 0:
            raise ValueError("state_slots must be a positive integer")
        self.state_slots = int(state_slots)
        self._state = np.zeros((self.state_slots, self.d_model), dtype=float)

        rng = np.random.default_rng(0x51A71C)
        routes = rng.normal(size=(self.state_slots, self.d_model))
        route_norms = np.linalg.norm(routes, axis=1, keepdims=True)
        self._routes = routes / route_norms

    @property
    def scalar_state_budget(self) -> int:
        return int(self._state.size)

    @property
    def state_shape(self) -> tuple[int, int]:
        return self._state.shape

    def reset(self) -> None:
        self._state.fill(0.0)

    def update(self, x_t: np.ndarray) -> None:
        event = _validate_vector(x_t, d_model=self.d_model, name="x_t")
        slot = int(np.argmax(self._routes @ event))
        self._state[slot] += event

    def read_components(self) -> np.ndarray:
        return self._state.copy()

    def read(self, query: np.ndarray) -> np.ndarray:
        return _normalized_pseudo_event_read(self._state, query)


class DecayMemory:
    """Fixed-size bank of exponentially decaying event traces."""

    def __init__(self, *, d_model: int, rhos: tuple[float, ...]) -> None:
        self.d_model = _validate_d_model(d_model)
        if not rhos:
            raise ValueError("rhos must contain at least one decay mode")
        values = np.asarray(rhos, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("rhos must be a finite 1D sequence")
        if np.any(values < 0.0) or np.any(values >= 1.0):
            raise ValueError("each rho must satisfy 0 <= rho < 1")
        self.rhos = tuple(float(v) for v in values)
        self._rho = values[:, None]
        self._state = np.zeros((len(self.rhos), self.d_model), dtype=float)

    @property
    def scalar_state_budget(self) -> int:
        return int(self._state.size)

    @property
    def state_shape(self) -> tuple[int, int]:
        return self._state.shape

    def reset(self) -> None:
        self._state.fill(0.0)

    def update(self, x_t: np.ndarray) -> None:
        event = _validate_vector(x_t, d_model=self.d_model, name="x_t")
        self._state = self._rho * self._state + event[None, :]

    def read_components(self) -> np.ndarray:
        return self._state.copy()

    def read(self, query: np.ndarray) -> np.ndarray:
        return _normalized_pseudo_event_read(self._state, query)


class ResonantMemory:
    """Damped complex trace bank stored explicitly as real/imaginary vectors."""

    def __init__(
        self,
        *,
        d_model: int,
        modes: tuple[tuple[float, float], ...],
    ) -> None:
        self.d_model = _validate_d_model(d_model)
        if not modes:
            raise ValueError("modes must contain at least one (rho, omega) pair")

        radii: list[float] = []
        omegas: list[float] = []
        for mode in modes:
            if len(mode) != 2:
                raise ValueError("each mode must be a (rho, omega) pair")
            rho = float(mode[0])
            omega = float(mode[1])
            if not np.isfinite(rho) or not np.isfinite(omega):
                raise ValueError("resonant mode parameters must be finite")
            if rho < 0.0 or rho >= 1.0:
                raise ValueError("each resonant rho must satisfy 0 <= rho < 1")
            radii.append(rho)
            omegas.append(omega)

        self.modes = tuple((float(r), float(w)) for r, w in zip(radii, omegas))
        self._rho = np.asarray(radii, dtype=float)[:, None]
        self._cos = np.cos(np.asarray(omegas, dtype=float))[:, None]
        self._sin = np.sin(np.asarray(omegas, dtype=float))[:, None]
        self._real = np.zeros((len(self.modes), self.d_model), dtype=float)
        self._imag = np.zeros_like(self._real)

    @property
    def scalar_state_budget(self) -> int:
        return int(self._real.size + self._imag.size)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (2 * len(self.modes), self.d_model)

    def reset(self) -> None:
        self._real.fill(0.0)
        self._imag.fill(0.0)

    def update(self, x_t: np.ndarray) -> None:
        event = _validate_vector(x_t, d_model=self.d_model, name="x_t")
        old_real = self._real
        old_imag = self._imag
        self._real = self._rho * (self._cos * old_real - self._sin * old_imag)
        self._imag = self._rho * (self._sin * old_real + self._cos * old_imag)
        self._real += event[None, :]

    def read_components(self) -> np.ndarray:
        components = np.empty(self.state_shape, dtype=float)
        components[0::2] = self._real
        components[1::2] = self._imag
        return components

    def read(self, query: np.ndarray) -> np.ndarray:
        return _normalized_pseudo_event_read(self.read_components(), query)


class RytmiWindowMemory:
    """Fast/slow temporal windows with matched coordinate-system controls.

    Each pair stores two full d_model traces under the recurrence
    trace_t = rho * trace_(t-1) + x_t. The default read exposes
    (fast - slow, slow) for each pair. The raw_fast_slow control exposes
    (fast, slow), while slow_only hides the fast trace from the reader. All
    modes keep the exact same stored fast/slow state.
    """

    def __init__(
        self,
        *,
        d_model: int,
        pairs: tuple[tuple[float, float], ...],
        read_mode: str = "direction",
    ) -> None:
        self.d_model = _validate_d_model(d_model)
        if not pairs:
            raise ValueError("pairs must contain at least one (rho_fast, rho_slow) pair")
        if read_mode not in {"direction", "raw_fast_slow", "slow_only"}:
            raise ValueError(
                "read_mode must be 'direction', 'raw_fast_slow', or 'slow_only'"
            )

        fast: list[float] = []
        slow: list[float] = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("each pair must be (rho_fast, rho_slow)")
            rho_fast = float(pair[0])
            rho_slow = float(pair[1])
            if not np.isfinite(rho_fast) or not np.isfinite(rho_slow):
                raise ValueError("window decay parameters must be finite")
            if not (0.0 <= rho_fast < rho_slow < 1.0):
                raise ValueError("each pair must satisfy 0 <= rho_fast < rho_slow < 1")
            fast.append(rho_fast)
            slow.append(rho_slow)

        self.pairs = tuple((f, s) for f, s in zip(fast, slow))
        self.read_mode = read_mode
        self._rho_fast = np.asarray(fast, dtype=float)[:, None]
        self._rho_slow = np.asarray(slow, dtype=float)[:, None]
        self._fast = np.zeros((len(self.pairs), self.d_model), dtype=float)
        self._slow = np.zeros_like(self._fast)

    @property
    def scalar_state_budget(self) -> int:
        return int(self._fast.size + self._slow.size)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (2 * len(self.pairs), self.d_model)

    def reset(self) -> None:
        self._fast.fill(0.0)
        self._slow.fill(0.0)

    def update(self, x_t: np.ndarray) -> None:
        event = _validate_vector(x_t, d_model=self.d_model, name="x_t")
        self._fast = self._rho_fast * self._fast + event[None, :]
        self._slow = self._rho_slow * self._slow + event[None, :]

    def read_components(self) -> np.ndarray:
        if self.read_mode == "slow_only":
            return self._slow.copy()
        components = np.empty(self.state_shape, dtype=float)
        if self.read_mode == "raw_fast_slow":
            components[0::2] = self._fast
        else:
            components[0::2] = self._fast - self._slow
        components[1::2] = self._slow
        return components

    def read(self, query: np.ndarray) -> np.ndarray:
        return _normalized_pseudo_event_read(self.read_components(), query)


def assert_equal_state_budget(memories: Iterable[object]) -> int:
    items = list(memories)
    if not items:
        raise ValueError("at least one memory is required")
    try:
        budgets = [int(memory.scalar_state_budget) for memory in items]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("every memory must expose scalar_state_budget") from exc
    if any(budget <= 0 for budget in budgets):
        raise ValueError("state budgets must be positive")
    if len(set(budgets)) != 1:
        raise ValueError(f"state budget mismatch: {budgets}")
    return budgets[0]
