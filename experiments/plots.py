from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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


def _save(figure, path: Path, plot, *, uses_inset_axes: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Matplotlib tight_layout does not fully support inset axes. For figures
    # with insets, let bbox_inches handle the final bounding box instead.
    if not uses_inset_axes:
        figure.tight_layout()
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
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
    axis.legend(loc="upper right")
    _save(figure, path, plot)


def plot_roc(path: Path, genuine: Iterable[float], impostor: Iterable[float], title: str) -> None:
    """Conventional ROC: linear FAR/TAR axes plus a zoom of the high-performing region."""
    plot = _pyplot()
    genuine_array = np.asarray(list(genuine), dtype=float)
    impostor_array = np.asarray(list(impostor), dtype=float)
    fpr, tpr, _ = _roc(genuine_array, impostor_array)

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    figure, axis = plot.subplots(figsize=PLOT_ROC_FIGSIZE)
    axis.plot(fpr, tpr, linewidth=2, label="ROC curve")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        title=title,
        xlabel="False Acceptance Rate (FAR)",
        ylabel="True Acceptance Rate (TAR)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")

    # Magnify the region where a strong classifier's ROC departs from the
    # ideal top-left corner. Keep the full ROC as the primary, comparable plot.
    inset = inset_axes(axis, width="40%", height="40%", loc="lower center",
                       borderpad=2.0)
    inset.plot(fpr, tpr, linewidth=1.6)
    inset.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.65)
    inset.set_xlim(0.0, 0.10)
    inset.set_ylim(0.90, 1.0)
    inset.set_title("Low-FAR detail", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, alpha=0.25)
    mark_inset(axis, inset, loc1=2, loc2=4, fc="none", ec="0.45", linewidth=0.8)

    _save(figure, path, plot, uses_inset_axes=True)


def plot_far_frr(path: Path, thresholds: Iterable[float], far: Iterable[float], frr: Iterable[float], title: str) -> None:
    """Plot FAR/FRR versus cosine threshold, with logarithmic error-rate axis."""
    plot = _pyplot()
    threshold_values = np.asarray(list(thresholds), dtype=float)
    far_values = np.asarray(list(far), dtype=float)
    frr_values = np.asarray(list(frr), dtype=float)

    # Rates are fractions in [0, 1]. A display floor of 1e-3 makes low EER
    # intersections visible without changing the underlying metric values.
    floor = 1e-3
    far_display = np.clip(far_values, floor, 1.0)
    frr_display = np.clip(frr_values, floor, 1.0)

    figure, axis = plot.subplots(figsize=PLOT_FAR_FRR_FIGSIZE)
    axis.plot(threshold_values, far_display, label="FAR")
    axis.plot(threshold_values, frr_display, label="FRR")
    axis.set_yscale("log")
    axis.set(
        title=title,
        xlabel="Cosine similarity threshold",
        ylabel="Error rate (log scale)",
        ylim=(floor, 1.0),
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="upper center")
    _save(figure, path, plot)



def plot_det(
    path: Path,
    genuine: Iterable[float],
    impostor: Iterable[float],
    title: str,
) -> None:
    """DET curve with normal-deviate axes and a clean low-error inset."""
    plot = _pyplot()

    false_accept, false_reject, _ = det_curve(genuine, impostor)
    far = np.asarray(false_accept, dtype=float)
    frr = np.asarray(false_reject, dtype=float)

    # Probit coordinates; clip endpoints because norm.ppf(0/1) is infinite.
    eps = 1e-5
    far_x = norm.ppf(np.clip(far, eps, 1.0 - eps))
    frr_y = norm.ppf(np.clip(frr, eps, 1.0 - eps))

    figure, axis = plot.subplots(figsize=PLOT_DET_FIGSIZE)
    axis.plot(far_x, frr_y, linewidth=2, label="DET curve")

    # Random classifier: FAR = FRR, hence the diagonal in probit space.
    bound = norm.ppf(1.0 - eps)
    axis.plot(
        [-bound, bound],
        [bound, -bound],
        "k--",
        linewidth=1,
        alpha=0.75,
        label="Random classifier",
    )

    # Use standard percentage tick values, converted to probit coordinates.
    tick_percent = np.array(
        [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 40, 50, 60, 80, 90, 95, 98, 99],
        dtype=float,
    )
    tick_positions = norm.ppf(tick_percent / 100.0)
    tick_labels = [f"{v:g}" for v in tick_percent]

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels)
    axis.set(
        title=title,
        xlabel="False Acceptance Rate (%)",
        ylabel="False Rejection Rate (%)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")

    # Zoom the low-error region. Do not draw connector lines over the main plot.
    inset = inset_axes(
        axis,
        width="40%",
        height="40%",
        loc="upper right",
        borderpad=2,
    )
    inset.plot(far_x, frr_y, linewidth=1.5)
    inset.plot(
        [-bound, bound],
        [-bound, bound],
        "k--",
        linewidth=0.8,
        alpha=0.65,
    )

    inset_min = norm.ppf(eps)
    inset_max = norm.ppf(0.05)
    inset.set_xlim(inset_min, inset_max)
    inset.set_ylim(inset_min, inset_max)

    inset_ticks_percent = np.array([0.001, 0.01, 0.1, 1, 5])
    inset_ticks = norm.ppf(inset_ticks_percent / 100.0)
    inset.set_xticks(inset_ticks)
    inset.set_xticklabels([f"{v:g}" for v in inset_ticks_percent])
    inset.set_yticks(inset_ticks)
    inset.set_yticklabels([f"{v:g}" for v in inset_ticks_percent])

    inset.set_title("Low-error detail", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, alpha=0.25)

    _save(figure, path, plot, uses_inset_axes=True)



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


def plot_fusion_delta(
    path: Path,
    records: Iterable[dict],
    metric: str = "eer_improvement_vs_best",
    title: str = "Fusion improvement",
) -> None:
    """Plot grouped score-level vs feature-level improvements for each model pair."""
    plot = _pyplot()

    rows = [
        row
        for row in records
        if row.get("method") in ("score_level", "feature_level")
    ]

    if not rows:
        raise ValueError("No score-level or feature-level fusion records to plot.")

    # Preserve the model-pair order from the input records.
    pair_order = []
    for row in rows:
        pair = (
            str(row["fingerprint_model"]),
            str(row["iris_model"]),
        )
        if pair not in pair_order:
            pair_order.append(pair)

    methods = ("score_level", "feature_level")
    method_labels = {
        "score_level": "Score-level",
        "feature_level": "Feature-level",
    }

    # Index values by (fingerprint model, iris model, fusion method).
    lookup = {
        (
            str(row["fingerprint_model"]),
            str(row["iris_model"]),
            row["method"],
        ): float(row[metric])
        for row in rows
    }

    values = {method: [] for method in methods}

    for fp_model, iris_model in pair_order:
        for method in methods:
            key = (fp_model, iris_model, method)

            if key not in lookup:
                raise ValueError(
                    f"Missing {method} record for "
                    f"{fp_model} + {iris_model}"
                )

            values[method].append(lookup[key])

    x = np.arange(len(pair_order), dtype=float)
    width = 0.38

    figure_width = max(
        PLOT_FUSION_DELTA_MIN_WIDTH,
        len(pair_order) * 0.8,
    )
    figure, axis = plot.subplots(
        figsize=(figure_width, PLOT_FUSION_DELTA_HEIGHT)
    )

    # Two adjacent bars per fingerprint–iris model combination.
    for index, method in enumerate(methods):
        offset = -width / 2 if index == 0 else width / 2

        axis.bar(
            x + offset,
            values[method],
            width=width,
            label=method_labels[method],
        )

    _label_bars(axis)

    labels = [
        f"FP:{fp}\nIRIS:{iris}"
        for fp, iris in pair_order
    ]

    axis.set(
        title=title,
        ylabel=metric,
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=60, labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")

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

    labels = [f"FP:{fp}\nIRIS:{iris}" for fp, iris in pair_order]
    axis.set(
        title=title,
        ylabel=metric.upper(),
        xticks=x,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=60, labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
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
