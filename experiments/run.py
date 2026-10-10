from __future__ import annotations

import argparse
from dataclasses import asdict

import config

from config import (
    EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS,
    EXPERIMENT_DEFAULT_BOOTSTRAP_SEED,
    EXPERIMENT_DEFAULT_MODALITY,
    EXPERIMENT_PAIRS,
    MODALITIES,
    RESULTS_DIR,
    VIRTUAL_PAIRING_LIMIT
)
from .analysis import (
    bootstrap_metric_intervals,
    fusion_delta_records
)
from .fusion import (
    MODELS,
    create_virtual_pairings,
    evaluate_fusion_pair,
    load_split_manifest,
    load_virtual_population_embeddings,
    run_fusion_pair,
)
from .io import (
    experiment_dir,
    load_embedding_artifact,
    load_history,
    write_json,
    write_metrics_csv,
    write_metrics_json,
    write_scores
)
from .plots import (
    plot_det,
    plot_far_frr,
    plot_fusion_delta,
    plot_metric_bars,
    plot_fusion_method_comparison,
    plot_pca,
    plot_roc,
    plot_score_distribution,
    plot_training_history
)
from .protocols import PROTOCOLS, evaluate_single_modality
from .protocols import score_with_templates


def _models(value: str | None) -> list[str]:
    if not value:
        return list(MODELS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if not selected:
        raise ValueError("At least one model must be selected.")
    return selected


def run_modality(
    modality: str,
    models: list[str],
    make_plots: bool = True,
    bootstrap_iterations: int = EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPERIMENT_DEFAULT_BOOTSTRAP_SEED
) -> list[dict]:
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    output = experiment_dir(modality)
    records: list[dict] = []
    bootstrap_records: list[dict] = []
    histories: list[dict] = []
    for model_index, model_name in enumerate(models):
        print(f"Running (Enrollment + Authentication) Test: {modality}/{model_name}...")
        data = load_embedding_artifact(modality, model_name)
        result, templates, verification = evaluate_single_modality(data, PROTOCOLS[modality])
        record = {"model_name": model_name, "modality": modality, **result.to_dict()}
        for key in ("thresholds", "far", "frr"):
            record.pop(key, None)
        records.append(record)

        genuine, impostor = score_with_templates(
            templates,
            verification,
            template_instance_level=config.STANDALONE_VIRTUAL_ENROLLMENT,
        )
        write_scores(output / "scores" / f"{model_name}.npz", genuine, impostor)
        if bootstrap_iterations:
            bootstrap_records.append({
                "model_name": model_name,
                "modality": modality,
                **bootstrap_metric_intervals(
                    genuine,
                    impostor,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + model_index
                )
            })
        if make_plots:
            plot_dir = output / "plots"
            plot_score_distribution(
                plot_dir / "distributions" / f"{model_name}_scores.png",
                genuine,
                impostor,
                f"{modality.title()} — {model_name}"
            )
            plot_roc(
                plot_dir / "roc" / f"{model_name}_roc.png",
                genuine,
                impostor,
                f"{modality.title()} ROC — {model_name}"
            )
            plot_det(
                plot_dir / "det" / f"{model_name}_det.png",
                genuine,
                impostor,
                f"{modality.title()} DET — {model_name}"
            )
            plot_far_frr(
                plot_dir / "far_frr" / f"{model_name}_far_frr.png",
                result.thresholds,
                result.far,
                result.frr,
                f"{modality.title()} FAR / FRR — {model_name}"
            )
            embedding_array = data["embeddings"].detach().cpu().numpy()
            if embedding_array.ndim == 2 and embedding_array.shape[1] >= 2:
                plot_pca(
                    plot_dir / "pca" / f"{model_name}_pca.png",
                    embedding_array,
                    data.get("class_keys"),
                    f"{modality.title()} embeddings PCA — {model_name}"
                )
        histories.append(load_history(model_name, modality))
        print("Completed.")

    write_metrics_json(output / "metrics.json", records)
    write_metrics_csv(output / "metrics.csv", records)
    if bootstrap_records:
        write_json(output / "bootstrap_intervals.json", bootstrap_records)
    if make_plots:
        plot_training_history(
            RESULTS_DIR / "plots" / "training" / f"{modality}.png",
            histories,
            f"{modality.title()} training"
        )
        plot_metric_bars(
            output / "plots" / "comparisons" / "eer_comparison.png",
            records,
            "eer",
            f"{modality.title()} EER comparison"
        )
        plot_metric_bars(
            output / "plots" / "comparisons" / "accuracy_comparison.png",
            records,
            "accuracy_at_eer",
            f"{modality.title()} accuracy at EER threshold"
        )
    write_json(
        output / "manifest.json",
        {
            "modality": modality,
            "models": models,
            "protocol": asdict(PROTOCOLS[modality]),
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "instance_level_enrollment": config.STANDALONE_VIRTUAL_ENROLLMENT,
            "instance_level_verification": config.STANDALONE_VIRTUAL_VERIFICATION,
        }
    )
    return records



def select_headline_fusion_pairs(
    split_manifest: dict,
    experiment_pairs: int = EXPERIMENT_PAIRS,
) -> tuple[list[dict[str, int]], int, int, int]:
    """Pair independently held-out subjects into deterministic virtual IDs."""
    fingerprint_eval = sorted(
        int(subject) for subject in split_manifest["fingerprint"]["eval_subjects"]
    )
    iris_eval = sorted(
        int(subject) for subject in split_manifest["iris"]["eval_subjects"]
    )
    fingerprint_count = len(fingerprint_eval)
    iris_count = len(iris_eval)
    available = min(fingerprint_count, iris_count)

    if experiment_pairs <= 0:
        raise ValueError("EXPERIMENT_PAIRS must be positive.")
    if experiment_pairs > VIRTUAL_PAIRING_LIMIT:
        raise ValueError(
            f"EXPERIMENT_PAIRS cannot exceed VIRTUAL_PAIRING_LIMIT "
            f"({VIRTUAL_PAIRING_LIMIT})."
        )
    if experiment_pairs > available:
        raise ValueError(
            "EXPERIMENT_PAIRS cannot exceed the available evaluation subjects. "
            f"Available: {available} (fingerprint eval: {fingerprint_count}, "
            f"iris eval: {iris_count}). Requested: {experiment_pairs}."
        )

    pairings = create_virtual_pairings(
        fingerprint_eval[:experiment_pairs],
        iris_eval[:experiment_pairs],
    )
    return pairings, available, fingerprint_count, iris_count


def _bootstrap_fusion_records(
    result: dict,
    fp_model: str,
    iris_model: str,
    methods: tuple[str, ...],
    iterations: int,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    for method_index, method in enumerate(methods):
        scores = result[method]
        records.append({
            "fingerprint_model": fp_model,
            "iris_model": iris_model,
            "method": method,
            **bootstrap_metric_intervals(
                scores["genuine"],
                scores["impostor"],
                iterations=iterations,
                seed=seed + method_index,
            ),
        })
    return records



def _write_fusion_delta_outputs(output, records: list[dict], make_plots: bool) -> None:
    delta_records = fusion_delta_records(records)
    write_metrics_json(output / "fusion_delta.json", delta_records)
    write_metrics_csv(output / "fusion_delta.csv", delta_records)
    if make_plots and delta_records:
        plot_fusion_delta(
            output / "plots" / "comparisons" / "fusion_eer_improvement.png",
            delta_records,
            metric="eer_improvement_vs_best",
            title="Synthetic fusion EER improvement vs best baseline",
        )
        plot_fusion_delta(
            output / "plots" / "comparisons" / "fusion_auc_improvement.png",
            delta_records,
            metric="auc_improvement_vs_best",
            title="Synthetic fusion AUC improvement vs best baseline",
        )

def result_protocol_description(instance_enrollment: bool, instance_verification: bool) -> str:
    enrollment = "instance-level" if instance_enrollment else "subject-level"
    verification = "instance-level" if instance_verification else "subject-level"
    return f"synthetic virtual fusion: {enrollment} enrollment, {verification} verification"

def run_fusion(
    models: list[str],
    make_plots: bool = True,
    include_nonheldout_diagnostic: bool = False,
    bootstrap_iterations: int = EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPERIMENT_DEFAULT_BOOTSTRAP_SEED,
) -> list[dict]:
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    output = experiment_dir("fusion")
    manifest = load_split_manifest()
    pairings, available_max, fingerprint_eval_count, iris_eval_count = (
        select_headline_fusion_pairs(manifest)
    )

    records: list[dict] = []
    bootstrap_records: list[dict] = []
    methods = (
        "score_level",
        "feature_level",
        "fingerprint_baseline",
        "iris_baseline",
    )
    pair_index = 0
    for fp_model in models:
        for iris_model in models:
            print(f"Running (Enrollment + Authentication) Test: fusion/fingerprint_{fp_model}_iris_{iris_model}...")
            result = run_fusion_pair(fp_model, iris_model, pairings)
            if bootstrap_iterations:
                bootstrap_records.extend(_bootstrap_fusion_records(
                    result,
                    fp_model,
                    iris_model,
                    methods,
                    bootstrap_iterations,
                    bootstrap_seed + pair_index * len(methods),
                ))
            for method in methods:
                metrics = result[method]["metrics"].to_dict()
                row = {
                    "fingerprint_model": fp_model,
                    "iris_model": iris_model,
                    "method": method,
                    "cohort": "synthetic_virtual_fusion",
                    "subject_count": len(pairings),
                    "trial_count": result["trial_count"],
                    "fusion_instance_level_enrollment": config.FUSION_VIRTUAL_ENROLLMENT,
                    "fusion_instance_level_verification": config.FUSION_VIRTUAL_VERIFICATION,
                    **metrics,
                }
                for key in ("thresholds", "far", "frr"):
                    row.pop(key, None)
                records.append(row)
                write_scores(
                    output / "scores" / f"{fp_model}__{iris_model}__{method}.npz",
                    result[method]["genuine"],
                    result[method]["impostor"],
                )
                if make_plots:
                    plot_det(
                        output / "plots" / "det" / f"{fp_model}__{iris_model}__{method}_det.png",
                        result[method]["genuine"],
                        result[method]["impostor"],
                        f"Synthetic fusion DET — {fp_model} + {iris_model} ({method})",
                    )
                    if method in ("score_level", "feature_level"):
                        method_title = f"Synthetic fusion — {fp_model} + {iris_model} ({method})"
                        plot_score_distribution(
                            output / "plots" / "distributions" / f"{fp_model}__{iris_model}__{method}.png",
                            result[method]["genuine"],
                            result[method]["impostor"],
                            method_title,
                        )
                        plot_roc(
                            output / "plots" / "roc" / f"{fp_model}__{iris_model}__{method}_roc.png",
                            result[method]["genuine"],
                            result[method]["impostor"],
                            f"{method_title} ROC",
                        )
                        plot_far_frr(
                            output / "plots" / "far_frr" / f"{fp_model}__{iris_model}__{method}_far_frr.png",
                            result[method]["metrics"].thresholds,
                            result[method]["metrics"].far,
                            result[method]["metrics"].frr,
                            f"{method_title} FAR / FRR",
                        )
            if make_plots and result.get("feature_pca_embeddings") is not None:
                plot_pca(
                    output / "plots" / "pca" / f"{fp_model}__{iris_model}_pca.png",
                    result["feature_pca_embeddings"],
                    result.get("feature_pca_labels"),
                    f"Synthetic fusion embeddings PCA — {fp_model} + {iris_model}",
                )
            pair_index += 1
            print("Completed.")

    headline_trial_count = records[0]["trial_count"] if records else 0
    write_json(
        output / "pairing.json",
        {
            "mapping": pairings,
            "experiment_pairs": EXPERIMENT_PAIRS,
            "fingerprint_eval_subject_count": fingerprint_eval_count,
            "iris_eval_subject_count": iris_eval_count,
            "available_eval_limit": available_max,
            "selected_virtual_ids": [pairing["virtual_id"] for pairing in pairings],
            "selected_count": len(pairings),
            "trial_count": headline_trial_count,
            "headline_protocol": result_protocol_description(
                config.FUSION_VIRTUAL_ENROLLMENT,
                config.FUSION_VIRTUAL_VERIFICATION,
            ),
            "warning": "Fingerprint and iris IDs are from different subjects; virtual identities are not real multimodal people.",
        },
    )
    write_metrics_json(output / "metrics.json", records)
    write_metrics_csv(output / "metrics.csv", records)
    _write_fusion_delta_outputs(output, records, make_plots)
    if bootstrap_records:
        write_json(output / "bootstrap_intervals.json", bootstrap_records)
    if make_plots:
        fusion_rows = [
            row for row in records if row["method"] in ("score_level", "feature_level")
        ]
        plot_fusion_method_comparison(
            output / "plots" / "comparisons" / "eer_comparison.png",
            fusion_rows,
            "eer",
            "Synthetic fusion EER comparison — all model combinations",
        )
        plot_fusion_method_comparison(
            output / "plots" / "comparisons" / "fusion_accuracy_comparison.png",
            fusion_rows,
            "accuracy_at_eer",
            "Synthetic fusion accuracy at EER threshold — all model combinations",
        )
    write_json(
        output / "manifest.json",
        {
            "models": models,
            "experiment_pairs": EXPERIMENT_PAIRS,
            "fingerprint_eval_subject_count": fingerprint_eval_count,
            "iris_eval_subject_count": iris_eval_count,
            "available_eval_limit": available_max,
            "mapping": pairings,
            "selected_virtual_ids": [pairing["virtual_id"] for pairing in pairings],
            "selected_count": len(pairings),
            "trial_count": headline_trial_count,
            "virtual_population": VIRTUAL_PAIRING_LIMIT,
            "headline_cohort": result_protocol_description(
                config.FUSION_VIRTUAL_ENROLLMENT,
                config.FUSION_VIRTUAL_VERIFICATION,
            ),
            "fusion_instance_level_enrollment": config.FUSION_VIRTUAL_ENROLLMENT,
            "fusion_instance_level_verification": config.FUSION_VIRTUAL_VERIFICATION,
            "include_nonheldout_diagnostic": include_nonheldout_diagnostic,
            "nonheldout_diagnostic_status": "not_run; switch reserved for explicitly labeled diagnostic",
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
    )
    return records


def run_nonheldout_diagnostic(models: list[str]) -> list[dict]:
    """Run a clearly labeled full virtual-population diagnostic.

    The modality artifacts may contain training subjects, so this output is
    never mixed with the headline held-out virtual-fusion metrics.
    """
    output = experiment_dir("fusion")
    split_manifest = load_split_manifest()
    diagnostic_subjects = list(range(VIRTUAL_PAIRING_LIMIT))
    records: list[dict] = []
    for fp_model in models:
        for iris_model in models:
            fp_data = load_virtual_population_embeddings(
                "fingerprint", fp_model, diagnostic_subjects, split_manifest
            )
            iris_data = load_virtual_population_embeddings(
                "iris", iris_model, diagnostic_subjects, split_manifest
            )
            diagnostic_pairings = create_virtual_pairings(
                diagnostic_subjects,
                diagnostic_subjects,
            )
            result = evaluate_fusion_pair(fp_data, iris_data, diagnostic_pairings)
            for method in ("score_level", "feature_level"):
                metrics = result[method]["metrics"].to_dict()
                row = {
                    "fingerprint_model": fp_model,
                    "iris_model": iris_model,
                    "method": method,
                    "cohort": "nonheldout_diagnostic_virtual_population",
                    "subject_count": len(diagnostic_subjects),
                    "trial_count": result["trial_count"],
                    "fusion_instance_level_enrollment": config.FUSION_VIRTUAL_ENROLLMENT,
                    "fusion_instance_level_verification": config.FUSION_VIRTUAL_VERIFICATION,
                    **metrics,
                }
                for key in ("thresholds", "far", "frr"):
                    row.pop(key, None)
                records.append(row)
    write_metrics_json(output / "nonheldout_diagnostic_metrics.json", records)
    write_metrics_csv(output / "nonheldout_diagnostic_metrics.csv", records)
    write_json(
        output / "nonheldout_diagnostic_manifest.json",
        {
            "cohort": "nonheldout_diagnostic_virtual_population",
            "subject_count": len(diagnostic_subjects),
            "warning": "Diagnostic only; train-subject artifacts may be included and this is not an unseen-subject result.",
        },
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible biometric experiments.")
    parser.add_argument(
        "--modality",
        choices=(*MODALITIES, "fusion", "all"),
        default=EXPERIMENT_DEFAULT_MODALITY,
    )
    parser.add_argument("--models", help="Comma-separated model names.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS,
        help="Number of bootstrap resamples; 0 disables confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=EXPERIMENT_DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--include-nonheldout-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations must be non-negative")
    models = _models(args.models)

    if args.modality in ("fingerprint", "iris", "all"):
        modalities = ("fingerprint", "iris") if args.modality == "all" else (args.modality,)
        for modality in modalities:
            run_modality(
                modality,
                models,
                not args.no_plots,
                args.bootstrap_iterations,
                args.bootstrap_seed,
            )
    if args.modality in ("fusion", "all"):
        run_fusion(
            models,
            not args.no_plots,
            args.include_nonheldout_diagnostic,
            args.bootstrap_iterations,
            args.bootstrap_seed,
        )
        if args.include_nonheldout_diagnostic:
            run_nonheldout_diagnostic(models)


if __name__ == "__main__":
    main()
