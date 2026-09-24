from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np

from .metrics import max_abs_error, mean_cosine_similarity, relative_rmse
from .tiny_head import causal_mask, stable_softmax


DECAY_CONFIGS = (
    (0.0, 0.25, 0.50, 0.65, 0.75, 0.82, 0.88, 0.92),
    (0.0, 0.50, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875),
    (0.75, 0.85, 0.90, 0.94, 0.96, 0.975, 0.985, 0.995),
)

RESONANT_CONFIGS = (
    ((0.90, 0.20), (0.94, 0.40), (0.97, 0.70), (0.985, 1.00)),
    ((0.90, 0.35), (0.94, 0.70), (0.97, 1.05), (0.985, 1.40)),
    ((0.90, 0.50), (0.94, 1.00), (0.97, 1.50), (0.985, 2.00)),
)


@dataclass(frozen=True)
class PretrainedHeadGeometry:
    d_model: int
    d_head: int
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    b_q: np.ndarray
    b_k: np.ndarray
    b_v: np.ndarray
    w_o: np.ndarray
    score_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.d_model, (int, np.integer)) or int(self.d_model) <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(self.d_head, (int, np.integer)) or int(self.d_head) <= 0:
            raise ValueError("d_head must be a positive integer")
        object.__setattr__(self, "d_model", int(self.d_model))
        object.__setattr__(self, "d_head", int(self.d_head))
        expected = {
            "w_q": (self.d_model, self.d_head),
            "w_k": (self.d_model, self.d_head),
            "w_v": (self.d_model, self.d_head),
            "b_q": (self.d_head,),
            "b_k": (self.d_head,),
            "b_v": (self.d_head,),
            "w_o": (self.d_head, self.d_model),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, value.copy())
        scale = float(self.score_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("score_scale must be positive and finite")
        object.__setattr__(self, "score_scale", scale)


@dataclass(frozen=True)
class HeadReference:
    attention: np.ndarray
    mixture: np.ndarray
    contribution: np.ndarray


@dataclass(frozen=True)
class ShadowMetrics:
    relative_rmse: float
    mean_cosine_similarity: float
    max_abs_error: float
    token_relative_rmse: tuple[float, ...]
    token_cosine_similarity: tuple[float, ...]


@dataclass(frozen=True)
class ShadowRun:
    prediction: np.ndarray
    metrics: ShadowMetrics
    state_budget: int


def _validate_tokens(x: np.ndarray, d_model: int) -> np.ndarray:
    tokens = np.asarray(x, dtype=float)
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] != d_model:
        raise ValueError("x must have shape (nonempty_sequence, d_model)")
    if not np.all(np.isfinite(tokens)):
        raise ValueError("x must contain only finite values")
    return tokens


def reconstruct_head(geometry: PretrainedHeadGeometry, x: np.ndarray) -> HeadReference:
    tokens = _validate_tokens(x, geometry.d_model)
    q = tokens @ geometry.w_q + geometry.b_q
    k = tokens @ geometry.w_k + geometry.b_k
    v = tokens @ geometry.w_v + geometry.b_v
    logits = (q @ k.T) * geometry.score_scale
    logits = logits.copy()
    logits[causal_mask(tokens.shape[0])] = -np.inf
    attention = stable_softmax(logits)
    mixture = attention @ v
    contribution = mixture @ geometry.w_o
    return HeadReference(attention=attention, mixture=mixture, contribution=contribution)


def resident_address_query(geometry: PretrainedHeadGeometry, x_t: np.ndarray) -> np.ndarray:
    event = np.asarray(x_t, dtype=float)
    if event.shape != (geometry.d_model,) or not np.all(np.isfinite(event)):
        raise ValueError("x_t must be a finite d_model vector")
    return (event @ geometry.w_q + geometry.b_q) @ geometry.w_k.T


def resident_write_operator(geometry: PretrainedHeadGeometry) -> np.ndarray:
    return geometry.w_v @ geometry.w_o


def resident_value_offset(geometry: PretrainedHeadGeometry) -> np.ndarray:
    return geometry.b_v @ geometry.w_o


def _build_shadow_metrics(reference: np.ndarray, prediction: np.ndarray) -> ShadowMetrics:
    return ShadowMetrics(
        relative_rmse=relative_rmse(reference, prediction),
        mean_cosine_similarity=mean_cosine_similarity(reference, prediction),
        max_abs_error=max_abs_error(reference, prediction),
        token_relative_rmse=tuple(relative_rmse(r, p) for r, p in zip(reference, prediction, strict=True)),
        token_cosine_similarity=tuple(mean_cosine_similarity(r, p) for r, p in zip(reference, prediction, strict=True)),
    )


def run_shadow_context(
    geometry: PretrainedHeadGeometry,
    x: np.ndarray,
    reference_y: np.ndarray,
    memory: object,
) -> ShadowRun:
    tokens = _validate_tokens(x, geometry.d_model)
    reference = np.asarray(reference_y, dtype=float)
    if reference.shape != (tokens.shape[0], geometry.d_model):
        raise ValueError("reference_y must have shape (sequence, d_model)")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_y must contain only finite values")
    budget = getattr(memory, "scalar_state_budget", None)
    if not isinstance(budget, (int, np.integer)) or int(budget) <= 0:
        raise ValueError("memory must expose a positive integer scalar_state_budget")
    for method in ("reset", "update", "read"):
        if not callable(getattr(memory, method, None)):
            raise ValueError(f"memory must provide {method}()")

    memory.reset()
    write = resident_write_operator(geometry)
    offset = resident_value_offset(geometry)
    predictions: list[np.ndarray] = []
    for x_t in tokens:
        address = resident_address_query(geometry, x_t)
        memory.update(x_t)
        resident_x = np.asarray(memory.read(address), dtype=float)
        if resident_x.shape != (geometry.d_model,) or not np.all(np.isfinite(resident_x)):
            raise ValueError("memory.read must return a finite d_model vector")
        predictions.append(resident_x @ write + offset)
    prediction = np.asarray(predictions, dtype=float)
    return ShadowRun(
        prediction=prediction,
        metrics=_build_shadow_metrics(reference, prediction),
        state_budget=int(budget),
    )


def evaluate_contexts(
    geometry: PretrainedHeadGeometry,
    contexts: Sequence[tuple[np.ndarray, np.ndarray]],
    memory_factory: Callable[[], object],
) -> dict[str, object]:
    if not contexts:
        raise ValueError("at least one context is required")
    runs = [run_shadow_context(geometry, x, reference_y, memory_factory()) for x, reference_y in contexts]
    references = np.concatenate([np.asarray(reference_y, dtype=float) for _, reference_y in contexts], axis=0)
    predictions = np.concatenate([run.prediction for run in runs], axis=0)
    return {
        "relative_rmse": relative_rmse(references, predictions),
        "mean_cosine_similarity": mean_cosine_similarity(references, predictions),
        "max_abs_error": max_abs_error(references, predictions),
        "runs": runs,
    }


def select_temporal_config(
    configs: Sequence[object],
    selection_contexts: tuple[PretrainedHeadGeometry, Sequence[tuple[np.ndarray, np.ndarray]]],
    memory_factory_for_config: Callable[[object], object],
):
    if not configs:
        raise ValueError("configs must be non-empty")
    geometry, contexts = selection_contexts
    scored = []
    for index, config in enumerate(configs):
        result = evaluate_contexts(
            geometry,
            contexts,
            lambda config=config: memory_factory_for_config(config),
        )
        scored.append((float(result["relative_rmse"]), index, config))
    return min(scored, key=lambda item: (item[0], item[1]))[2]
