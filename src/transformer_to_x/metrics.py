from __future__ import annotations

import numpy as np


def _matching_finite_arrays(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(est)):
        raise ValueError("metrics require finite values")
    return ref, est


def max_abs_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = _matching_finite_arrays(reference, estimate)
    return float(np.max(np.abs(est - ref))) if ref.size else 0.0


def relative_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = _matching_finite_arrays(reference, estimate)
    if ref.size == 0:
        return 0.0
    error_rms = float(np.sqrt(np.mean(np.square(est - ref))))
    reference_rms = float(np.sqrt(np.mean(np.square(ref))))
    if reference_rms == 0.0:
        return 0.0 if error_rms == 0.0 else float("inf")
    return error_rms / reference_rms


def mean_cosine_similarity(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = _matching_finite_arrays(reference, estimate)
    if ref.ndim == 1:
        ref = ref[None, :]
        est = est[None, :]
    elif ref.ndim != 2:
        raise ValueError("cosine similarity expects vectors or a matrix of row vectors")

    ref_norm = np.linalg.norm(ref, axis=1)
    est_norm = np.linalg.norm(est, axis=1)
    both_zero = (ref_norm == 0.0) & (est_norm == 0.0)
    one_zero = (ref_norm == 0.0) ^ (est_norm == 0.0)
    denom = ref_norm * est_norm
    cosine = np.empty(ref.shape[0], dtype=float)
    nonzero = denom > 0.0
    cosine[nonzero] = (
        np.sum(ref[nonzero] * est[nonzero], axis=1) / denom[nonzero]
    )
    cosine[both_zero] = 1.0
    cosine[one_zero] = 0.0
    return float(np.mean(cosine))
