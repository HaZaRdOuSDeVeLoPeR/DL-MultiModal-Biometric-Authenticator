from __future__ import annotations
from typing import Iterable, Sequence

import numpy as np

from config import (
    BOOTSTRAP_DEFAULT_CONFIDENCE,
    BOOTSTRAP_DEFAULT_ITERATIONS,
    BOOTSTRAP_DEFAULT_SEED,
    BOOTSTRAP_MAX_GENUINE_SAMPLES,
    BOOTSTRAP_MAX_IMPOSTOR_SAMPLES,
    PCA_MAX_POINTS,
)

from collections import defaultdict
import torch
import torch.nn.functional as F
from .metrics import compute_metrics, _roc
from .protocols import build_modality_evaluation_data
from src.authentication import split_enrollment_verification
from .protocols import subject_from_class_key


def _score_arrays(genuine: Iterable[float], impostor: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    genuine_array = np.asarray(list(genuine), dtype=np.float64).reshape(-1)
    impostor_array = np.asarray(list(impostor), dtype=np.float64).reshape(-1)
    if genuine_array.size == 0 or impostor_array.size == 0:
        raise ValueError("Both genuine and impostor scores must be non-empty.")
    if not np.isfinite(genuine_array).all() or not np.isfinite(impostor_array).all():
        raise ValueError("Scores must contain only finite values.")
    return genuine_array, impostor_array


def _percentile_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be finite and non-empty.")
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha))


def bootstrap_metric_intervals(
    genuine: Iterable[float],
    impostor: Iterable[float],
    iterations: int = BOOTSTRAP_DEFAULT_ITERATIONS,
    confidence: float = BOOTSTRAP_DEFAULT_CONFIDENCE,
    seed: int = BOOTSTRAP_DEFAULT_SEED,
    max_genuine_samples: int = BOOTSTRAP_MAX_GENUINE_SAMPLES,
    max_impostor_samples: int = BOOTSTRAP_MAX_IMPOSTOR_SAMPLES,
) -> dict[str, object]:
    """Estimate deterministic percentile confidence intervals for AUC and EER.

    Bootstrap resampling is performed independently within the genuine and
    impostor populations. Large impostor arrays are deterministically reduced
    to ``max_impostor_samples`` before resampling so analysis does not turn
    into an accidental second full-scale evaluation. The returned metadata
    records the effective sample sizes and iteration count.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if max_genuine_samples <= 0 or max_impostor_samples <= 0:
        raise ValueError("Bootstrap sample limits must be positive.")
    genuine_array, impostor_array = _score_arrays(genuine, impostor)
    rng = np.random.default_rng(seed)

    if genuine_array.size > max_genuine_samples:
        indices = rng.choice(genuine_array.size, max_genuine_samples, replace=False)
        genuine_array = genuine_array[np.sort(indices)]
    if impostor_array.size > max_impostor_samples:
        indices = rng.choice(impostor_array.size, max_impostor_samples, replace=False)
        impostor_array = impostor_array[np.sort(indices)]

    auc_values: list[float] = []
    eer_values: list[float] = []
    for _ in range(iterations):
        genuine_sample = genuine_array[rng.integers(0, genuine_array.size, genuine_array.size)]
        impostor_sample = impostor_array[rng.integers(0, impostor_array.size, impostor_array.size)]
        result = compute_metrics(genuine_sample, impostor_sample)
        auc_values.append(result.auc)
        eer_values.append(result.eer)

    auc_interval = _percentile_interval(auc_values, confidence)
    eer_interval = _percentile_interval(eer_values, confidence)
    return {
        "confidence": float(confidence),
        "iterations": int(iterations),
        "seed": int(seed),
        "genuine_samples": int(genuine_array.size),
        "impostor_samples": int(impostor_array.size),
        "auc": {"lower": auc_interval[0], "upper": auc_interval[1]},
        "eer": {"lower": eer_interval[0], "upper": eer_interval[1]},
    }


def det_curve(genuine: Iterable[float], impostor: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return false-accept, false-reject, and threshold arrays for a DET plot."""
    genuine_array, impostor_array = _score_arrays(genuine, impostor)
    fpr, tpr, thresholds = _roc(genuine_array, impostor_array)
    fnr = 1.0 - tpr
    # Keep boundary operating points as well as interior points. Perfectly
    # separated scores can otherwise produce an empty DET curve; the plot
    # applies an epsilon clip before transforming rates to probit coordinates.
    finite = np.isfinite(thresholds)
    return fpr[finite], fnr[finite], thresholds[finite]


def pca_projection(
    embeddings: np.ndarray,
    labels: Iterable[object] | None = None,
    max_points: int = PCA_MAX_POINTS,
) -> tuple[np.ndarray, list[str]]:
    """Compute a deterministic two-dimensional PCA projection.

    The first ``max_points`` rows are used when an artifact is larger than
    the plotting budget. This preserves extraction order and makes plots
    repeatable without sampling held-out examples based on their labels.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("embeddings must be a 2-D array with at least two rows and columns.")
    if max_points <= 1:
        raise ValueError("max_points must be greater than one.")
    values = values[:max_points]
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T
    if labels is None:
        output_labels = ["all"] * len(projection)
    else:
        output_labels = [str(label) for label in list(labels)[: len(projection)]]
        if len(output_labels) != len(projection):
            raise ValueError("labels length must match embeddings.")
    return projection.astype(np.float32), output_labels


def fusion_delta_records(records: Iterable[dict]) -> list[dict]:
    """Build per-pair fusion improvement rows from fusion and baseline records.

    Negative EER deltas and positive AUC deltas indicate improvement over a
    baseline. Both baselines are reported separately because each uses a
    different architecture and modality in a fingerprint/iris model pair.
    """
    rows = list(records)
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        pair = (str(row["fingerprint_model"]), str(row["iris_model"]))
        grouped.setdefault(pair, {})[str(row["method"])] = row

    output: list[dict] = []
    for (fingerprint_model, iris_model), methods in sorted(grouped.items()):
        fingerprint = methods.get("fingerprint_baseline")
        iris = methods.get("iris_baseline")
        if fingerprint is None or iris is None:
            continue
        best_eer = min(float(fingerprint["eer"]), float(iris["eer"]))
        best_auc = max(float(fingerprint["auc"]), float(iris["auc"]))
        for method in ("score_level", "feature_level"):
            fusion = methods.get(method)
            if fusion is None:
                continue
            output.append({
                "fingerprint_model": fingerprint_model,
                "iris_model": iris_model,
                "method": method,
                "cohort": fusion.get("cohort", "joint_held_out"),
                "eer_delta_vs_fingerprint": float(fusion["eer"]) - float(fingerprint["eer"]),
                "eer_delta_vs_iris": float(fusion["eer"]) - float(iris["eer"]),
                "eer_improvement_vs_best": best_eer - float(fusion["eer"]),
                "auc_delta_vs_fingerprint": float(fusion["auc"]) - float(fingerprint["auc"]),
                "auc_delta_vs_iris": float(fusion["auc"]) - float(iris["auc"]),
                "auc_improvement_vs_best": float(fusion["auc"]) - best_auc,
            })
    return output


__all__ = [
    "bootstrap_metric_intervals",
    "det_curve",
    "fusion_delta_records",
    "pca_projection",
]



def rank_modality_trials(
    data: dict,
    protocol,
    instance_level_enrollment: bool,
    instance_level_verification: bool,
    top_n: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Return top-N genuine and impostor comparisons with image traceability.

    The full similarity matrix is computed with PyTorch, while only the top-N
    impostor candidates per query are retained. This keeps the notebook useful
    for large fingerprint cohorts without materializing millions of Python
    dictionaries. The returned rows contain both sample paths and IDs.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    enrollment, verification = split_enrollment_verification(data, protocol)
    templates, queries = build_modality_evaluation_data(
        data,
        protocol,
        instance_level_enrollment=instance_level_enrollment,
        instance_level_verification=instance_level_verification,
    )
    keys = sorted(templates)
    template_matrix = F.normalize(torch.stack([templates[k] for k in keys]).float(), p=2, dim=1)
    query_matrix = F.normalize(queries["embeddings"].float(), p=2, dim=1)
    similarity = query_matrix @ template_matrix.T

    # Enrollment paths are grouped according to the template semantics.
    enrollment_paths: dict[str, list[str]] = defaultdict(list)
    if instance_level_enrollment:
        for key, path in zip(enrollment["class_keys"], enrollment["paths"]):
            enrollment_paths[str(key)].append(str(path))
    else:
        for key, path in zip(enrollment["class_keys"], enrollment["paths"]):
            enrollment_paths[subject_from_class_key(str(key))].append(str(path))

    query_keys = [str(k) for k in queries["class_keys"]]
    query_subjects = [str(s) for s in queries.get("subjects", query_keys)]
    query_paths = list(verification["paths"])
    # Subject-level verification queries are aggregated, so one query can have
    # multiple source images. Reconstruct those paths in the same grouping.
    if not instance_level_verification:
        grouped_paths: dict[tuple[str, int], list[str]] = defaultdict(list)
        for key, impression, path in zip(
            verification["class_keys"], verification["impressions"], verification["paths"]
        ):
            grouped_paths[(subject_from_class_key(str(key)), int(impression))].append(str(path))
        query_paths = [
            "; ".join(grouped_paths[(subject, int(impression))])
            for subject, impression in zip(query_subjects, queries["impressions"])
        ]

    key_to_index = {key: i for i, key in enumerate(keys)}
    genuine_rows: list[dict] = []
    impostor_candidates: list[dict] = []
    for row, (query_key, subject) in enumerate(zip(query_keys, query_subjects)):
        if instance_level_enrollment and instance_level_verification:
            genuine_index = key_to_index[query_key]
        elif instance_level_enrollment:
            matching = [i for i, key in enumerate(keys) if subject_from_class_key(key) == subject]
            # The verification sample's class maps to its class-level enrollment template.
            genuine_index = key_to_index.get(query_key, matching[0] if matching else -1)
        else:
            if subject not in key_to_index:
                raise ValueError(f"Verification subject has no enrollment template: {subject}")
            genuine_index = key_to_index[subject]
        if genuine_index < 0:
            raise ValueError(f"Unable to locate genuine template for query {query_key}")

        genuine_score = float(similarity[row, genuine_index])
        genuine_rows.append({
            "score": genuine_score,
            "enrollment_subject_id": subject,
            "verification_subject_id": subject,
            "enrollment_key": keys[genuine_index],
            "verification_key": query_key,
            "enrollment_paths": enrollment_paths[keys[genuine_index]],
            "verification_path": query_paths[row],
            "genuine": True,
        })

        impostor_mask = torch.ones(similarity.shape[1], dtype=torch.bool)
        impostor_mask[genuine_index] = False
        candidate_count = min(top_n, int(impostor_mask.sum().item()))
        values, indices = torch.topk(similarity[row].masked_fill(~impostor_mask, -torch.inf), k=candidate_count)
        for score, index in zip(values.tolist(), indices.tolist()):
            impostor_candidates.append({
                "score": float(score),
                "enrollment_subject_id": subject_from_class_key(keys[index]) if "_" in keys[index] else keys[index],
                "verification_subject_id": subject,
                "enrollment_key": keys[index],
                "verification_key": query_key,
                "enrollment_paths": enrollment_paths[keys[index]],
                "verification_path": query_paths[row],
                "genuine": False,
            })

    genuine_rows.sort(key=lambda row: row["score"], reverse=True)
    impostor_candidates.sort(key=lambda row: row["score"], reverse=True)
    return genuine_rows[:top_n], impostor_candidates[:top_n]


__all__.append("rank_modality_trials")
