from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from .metrics import _roc
from .analysis import det_curve
from .analysis import pca_projection

from config import (
    PLOT_DPI,
    SCORE_HISTOGRAM_BINS,
    PLOT_SCORE_FIGSIZE,
    PLOT_ROC_FIGSIZE,
    PLOT_FAR_FRR_FIGSIZE,
    PLOT_DET_FIGSIZE,
    PLOT_PCA_FIGSIZE,
    PLOT_TRAINING_FIGSIZE,
    PLOT_METRIC_FIGSIZE,
    PLOT_FUSION_DELTA_HEIGHT,
    PLOT_FUSION_DELTA_MIN_WIDTH,
    PLOT_SCATTER_SIZE,
)


def _pyplot():
    matplotlib.use("Agg")
    return plt


def _save(figure, path: Path, plot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=PLOT_DPI)
    plot.close(figure)


def _label_bars(axis, containers=None, fmt="{:.3f}") -> None:
    """Display each bar's value above the bar."""
    containers = containers if containers is not None else axis.containers

    for container in containers:
        axis.bar_label(
            container,
            fmt=fmt,
            padding=3,
            fontsize=8,
        )


def plot_score_distribution(path: Path, genuine: Iterable[float], impostor: Iterable[float], title: str) -> None:
    plot = _pyplot()
    figure, axis = plot.subplots(figsize=PLOT_SCORE_FIGSIZE)
    axis.hist(np.asarray(list(genuine)), bins=SCORE_HISTOGRAM_BINS, alpha=0.65, density=True, label="Genuine")
    axis.hist(np.asarray(list(impostor)), bins=SCORE_HISTOGRAM_BINS, alpha=0.65, density=True, label="Impostor")
    axis.set(title=title, xlabel="Cosine similarity", ylabel="Density")
    axis.legend()
    _save(figure, path, plot)


def plot_roc(path: Path, genuine: Iterable[float], impostor: Iterable[float], title: str) -> None:
    """Plot ROC curve with a logarithmic FAR axis."""
    plot = _pyplot()
    genuine_array = np.asarray(list(genuine), dtype=float)
    impostor_array = np.asarray(list(impostor), dtype=float)
    fpr, tpr, _ = _roc(genuine_array, impostor_array)

    figure, axis = plot.subplots(figsize=PLOT_ROC_FIGSIZE)

    # Logarithmic axes cannot display zero.
    floor = 1e-5
    fpr_plot = np.maximum(fpr, floor)

    axis.plot(fpr_plot, tpr, linewidth=2, label="ROC Curve")
    axis.plot(
        [floor, 1], [0, 1],
        "k--", linewidth=1, label="Random",
    )

    axis.set_xscale("log")
    axis.set(
        xlim=(floor, 1),
        ylim=(0, 1),
        title=title,
        xlabel="False Acceptance Rate (FAR)",
        ylabel="True Acceptance Rate (TAR)",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    _save(figure, path, plot)


def plot_far_frr(path: Path, thresholds: Iterable[float], far: Iterable[float], frr: Iterable[float], title: str) -> None:
    plot = _pyplot()
    figure, axis = plot.subplots(figsize=PLOT_FAR_FRR_FIGSIZE)
    axis.plot(list(thresholds), list(far), label="FAR")
    axis.plot(list(thresholds), list(frr), label="FRR")
    axis.set(title=title, xlabel="Threshold", ylabel="Rate", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path, plot)


def plot_det(path: Path, genuine: Iterable[float], impostor: Iterable[float], title: str) -> None:
    """Plot DET curve with a logarithmic FAR axis."""
    plot = _pyplot()
    false_accept, false_reject, _ = det_curve(genuine, impostor)

    # Convert to percentages.
    far_percent = np.asarray(false_accept, dtype=float) * 100
    frr_percent = np.asarray(false_reject, dtype=float) * 100

    floor = 1e-3  # 0.001% FAR
    far_plot = np.maximum(far_percent, floor)

    figure, axis = plot.subplots(figsize=PLOT_DET_FIGSIZE)

    axis.plot(far_plot, frr_percent, linewidth=2)
    axis.plot(
        [floor, 100], [100, 0],
        "k--", linewidth=1, alpha=0.5,
    )

    axis.set_xscale("log")
    axis.set(
        xlim=(floor, 100),
        ylim=(0, 100),
        title=title,
        xlabel="False Acceptance Rate (%)",
        ylabel="False Rejection Rate (%)",
    )
    axis.grid(True, which="both", alpha=0.25)
    _save(figure, path, plot)


def plot_pca(path: Path, embeddings: np.ndarray, labels: Iterable[object] | None = None, title: str = "Embedding PCA") -> None:
    """Plot a deterministic two-dimensional embedding projection."""
    plot = _pyplot()
    projection, output_labels = pca_projection(embeddings, labels)
    figure, axis = plot.subplots(figsize=PLOT_PCA_FIGSIZE)
    unique = list(dict.fromkeys(output_labels))
    if len(unique) <= 20:
        for label in unique:
            mask = np.asarray([value == label for value in output_labels])
            axis.scatter(projection[mask, 0], projection[mask, 1], s=PLOT_SCATTER_SIZE, alpha=0.45, label=label)
        axis.legend(fontsize=7, ncol=2)
    else:
        axis.scatter(projection[:, 0], projection[:, 1], s=PLOT_SCATTER_SIZE, alpha=0.45)
    axis.set(title=title, xlabel="Principal component 1", ylabel="Principal component 2")
    axis.grid(alpha=0.2)
    _save(figure, path, plot)


def plot_fusion_delta(path: Path, records: Iterable[dict], metric: str = "eer_improvement_vs_best", title: str = "Fusion improvement") -> None:
    """Plot fusion improvement over the stronger same-pair baseline."""
    plot = _pyplot()
    rows = list(records)
    labels = [f"{row['fingerprint_model']}+{row['iris_model']}\n{row['method']}" for row in rows]
    values = [float(row[metric]) for row in rows]
    figure, axis = plot.subplots(figsize=(max(PLOT_FUSION_DELTA_MIN_WIDTH, len(rows) * 0.45), PLOT_FUSION_DELTA_HEIGHT))
    axis.bar(np.arange(len(rows)), values)
    _label_bars(axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title=title, ylabel=metric, xticks=np.arange(len(rows)), xticklabels=labels)
    axis.tick_params(axis="x", labelrotation=70, labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path, plot)


def plot_training_history(path: Path, histories: Iterable[dict], title: str) -> None:
    plot = _pyplot()
    figure, (loss_axis, time_axis) = plot.subplots(1, 2, figsize=PLOT_TRAINING_FIGSIZE)
    for history in histories:
        label = f"{history['modality']} / {history['model_name']}"
        loss_axis.plot(history.get("train_loss", []), label=label)
        time_axis.plot(history.get("epoch_times", []), label=label)
    loss_axis.set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    time_axis.set(title="Epoch duration", xlabel="Epoch", ylabel="Seconds")
    loss_axis.legend(fontsize=8)
    time_axis.legend(fontsize=8)
    figure.suptitle(title)
    _save(figure, path, plot)


def plot_fusion_method_comparison(path: Path, records: Iterable[dict], metric: str, title: str) -> None:
    """Plot a grouped comparison across every fingerprint/iris model pair.

    Only the two fusion methods are shown (score-level and feature-level).
    Each fingerprint-model × iris-model combination gets one group.
    """
    plot = _pyplot()
    rows = [
        row for row in records
        if row.get("method") in ("score_level", "feature_level")
    ]

    pair_order = []
    for row in rows:
        pair = (str(row["fingerprint_model"]), str(row["iris_model"]))
        if pair not in pair_order:
            pair_order.append(pair)

    methods = ("score_level", "feature_level")
    method_labels = {
        "score_level": "Score-level",
        "feature_level": "Feature-level",
    }
    values = {
        method: [] for method in methods
    }
    lookup = {
        (str(row["fingerprint_model"]), str(row["iris_model"]), row["method"]): float(row[metric])
        for row in rows
    }

    for pair in pair_order:
        for method in methods:
            key = (pair[0], pair[1], method)
            if key not in lookup:
                raise ValueError(f"Missing {method} record for fusion pair {pair[0]} + {pair[1]}")
            values[method].append(lookup[key])

    x = np.arange(len(pair_order), dtype=float)
    width = 0.38
    figure_width = max(PLOT_FUSION_DELTA_MIN_WIDTH, len(pair_order) * 0.8)
    figure, axis = plot.subplots(figsize=(figure_width, PLOT_FUSION_DELTA_HEIGHT))

    for index, method in enumerate(methods):
        offset = (-width / 2) if index == 0 else (width / 2)
        axis.bar(
            x + offset,
            values[method],
            width=width,
            label=method_labels[method],
        )

    _label_bars(axis)

    labels = [f"{fp}+{iris}" for fp, iris in pair_order]
    axis.set(
        title=title,
        ylabel=metric.upper(),
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(figure, path, plot)


def plot_metric_bars(path: Path, records: Iterable[dict], metric: str, title: str) -> None:
    """Plot a grouped comparison bar chart from metric records."""
    plot = _pyplot()
    rows = list(records)
    labels = [str(row.get("model_name", row.get("method", ""))) for row in rows]
    values = [float(row[metric]) for row in rows]
    figure, axis = plot.subplots(figsize=PLOT_METRIC_FIGSIZE)
    axis.bar(labels, values)
    _label_bars(axis)
    axis.set(title=title, ylabel=metric.upper())
    axis.tick_params(axis="x", rotation=25)
    _save(figure, path, plot)
