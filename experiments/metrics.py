from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from config import FAR_TARGETS

@dataclass
class MetricResult:
    """Authentication metrics computed from genuine and impostor scores."""

    auc: float
    eer: float
    eer_threshold: float
    far_at_eer: float
    frr_at_eer: float
    accuracy_at_eer: float
    genuine_mean: float
    genuine_std: float
    genuine_median: float
    impostor_mean: float
    impostor_std: float
    impostor_median: float
    d_prime: float
    genuine_trials: int
    impostor_trials: int
    tar_at_far: dict[str, float]
    thresholds: list[float]
    far: list[float]
    frr: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _arrays(genuine: Iterable[float], impostor: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    genuine_array = np.asarray(list(genuine), dtype=np.float64).reshape(-1)
    impostor_array = np.asarray(list(impostor), dtype=np.float64).reshape(-1)
    if genuine_array.size == 0 or impostor_array.size == 0:
        raise ValueError("Both genuine and impostor scores must be non-empty.")
    if not np.isfinite(genuine_array).all() or not np.isfinite(impostor_array).all():
        raise ValueError("Scores must contain only finite values.")
    return genuine_array, impostor_array


def _roc(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(genuine.size, dtype=np.int8), np.zeros(impostor.size, dtype=np.int8)])
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1]
    thresholds = sorted_scores[distinct]
    tp = np.cumsum(sorted_labels)[distinct]
    fp = (distinct + 1) - tp
    fn_total = genuine.size
    tn_total = impostor.size
    tpr = tp / fn_total
    fpr = fp / tn_total
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0], np.r_[np.inf, thresholds, -np.inf]


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr, kind="mergesort")
    return float(np.trapezoid(tpr[order], fpr[order])) if hasattr(np, "trapezoid") else float(np.trapezoid(tpr[order], fpr[order]))


def _finite_boundary_threshold(thresholds: np.ndarray, index: int) -> float:
    """Return a finite representative threshold for an ROC boundary."""
    candidate_indices = (index, index + 1, index - 1, index + 2)
    for candidate in candidate_indices:
        if 0 <= candidate < len(thresholds) and np.isfinite(thresholds[candidate]):
            return float(thresholds[candidate])
    return 0.0


def _eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Estimate EER by interpolating the first FAR/FRR crossing.

    The ROC representation includes ``+/-inf`` sentinel thresholds.  Those
    sentinels are useful for rates but must never leak into the exported EER
    threshold, so boundary crossings use the nearest finite score threshold.
    """
    frr = 1.0 - tpr
    difference = fpr - frr
    exact = np.flatnonzero(difference == 0.0)
    if exact.size:
        index = int(exact[0])
        return float(fpr[index]), _finite_boundary_threshold(thresholds, index)

    crossing = np.flatnonzero(
        (difference[:-1] < 0.0) & (difference[1:] > 0.0)
        | (difference[:-1] > 0.0) & (difference[1:] < 0.0)
    )
    if crossing.size:
        i = int(crossing[0])
        x0, x1 = difference[i], difference[i + 1]
        alpha = 0.0 if x1 == x0 else float(-x0 / (x1 - x0))
        eer = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
        left, right = thresholds[i], thresholds[i + 1]
        if np.isfinite(left) and np.isfinite(right):
            threshold = float(left + alpha * (right - left))
        else:
            threshold = _finite_boundary_threshold(thresholds, i)
        return eer, threshold

    index = int(np.argmin(np.abs(difference)))
    return float((fpr[index] + frr[index]) / 2.0), _finite_boundary_threshold(thresholds, index)


def far_frr(genuine: Iterable[float], impostor: Iterable[float], threshold: float) -> tuple[float, float]:
    genuine_array, impostor_array = _arrays(genuine, impostor)
    return float(np.mean(impostor_array >= threshold)), float(np.mean(genuine_array < threshold))


def _threshold_for_far(impostor: np.ndarray, target: float) -> float:
    if not 0 <= target <= 1:
        raise ValueError(f"FAR target must be in [0, 1], got {target}.")
    if target == 0:
        return float(np.nextafter(impostor.max(), np.inf))
    quantile = 1.0 - target
    # ``higher`` is available in current NumPy versions.  Keep a small
    # compatibility fallback for older installations without exposing the
    # legacy keyword to static type checkers.
    try:
        quantile_fn = getattr(np, "quantile")
        value = quantile_fn(impostor, quantile, method="higher")
    except TypeError:  # NumPy < 1.22
        value = getattr(np, "quantile")(impostor, quantile, interpolation="higher")
    return float(value)


def compute_metrics(
    genuine: Iterable[float],
    impostor: Iterable[float],
    far_targets: Iterable[float] = FAR_TARGETS,
) -> MetricResult:
    """Compute robust threshold metrics for similarity scores."""
    genuine_array, impostor_array = _arrays(genuine, impostor)
    fpr, tpr, thresholds = _roc(genuine_array, impostor_array)
    auc = _auc(fpr, tpr)
    eer, eer_threshold = _eer(fpr, tpr, thresholds)
    far_eer, frr_eer = far_frr(genuine_array, impostor_array, eer_threshold)
    accuracy_at_eer = float((np.sum(genuine_array >= eer_threshold) + np.sum(impostor_array < eer_threshold)) / (genuine_array.size + impostor_array.size))

    tar_at_far: dict[str, float] = {}
    for target in far_targets:
        threshold = _threshold_for_far(impostor_array, float(target))
        tar_at_far[f"{float(target):g}"] = float(np.mean(genuine_array >= threshold))

    genuine_std = float(genuine_array.std(ddof=1)) if genuine_array.size > 1 else 0.0
    impostor_std = float(impostor_array.std(ddof=1)) if impostor_array.size > 1 else 0.0
    pooled = np.sqrt((genuine_std**2 + impostor_std**2) / 2.0)
    # A zero pooled variance means the distributions are degenerate.  Use a
    # finite value so JSON exports remain valid and deterministic.
    d_prime = float((genuine_array.mean() - impostor_array.mean()) / pooled) if pooled > 0 else 0.0

    # ``_roc`` already contains the rates at every distinct finite score
    # threshold.  Reuse those cumulative rates instead of rescanning every
    # trial for every threshold (which is prohibitively expensive for the
    # class-level impostor matrix).
    operating_thresholds = thresholds[1:-1]
    far = fpr[1:-1]
    frr = 1.0 - tpr[1:-1]

    return MetricResult(
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
        far_at_eer=far_eer,
        frr_at_eer=frr_eer,
        accuracy_at_eer=accuracy_at_eer,
        genuine_mean=float(genuine_array.mean()),
        genuine_std=genuine_std,
        genuine_median=float(np.median(genuine_array)),
        impostor_mean=float(impostor_array.mean()),
        impostor_std=impostor_std,
        impostor_median=float(np.median(impostor_array)),
        d_prime=d_prime,
        genuine_trials=int(genuine_array.size),
        impostor_trials=int(impostor_array.size),
        tar_at_far=tar_at_far,
        thresholds=operating_thresholds.tolist(),
        far=far.tolist(),
        frr=frr.tolist(),
    )
