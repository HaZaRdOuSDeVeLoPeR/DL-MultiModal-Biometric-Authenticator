from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

import config

from config import (
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    MODEL_NAMES,
    FUSION_FINGERPRINT_WEIGHT,
)
from src.authentication import (
    FINGERPRINT_PROTOCOL,
    IRIS_PROTOCOL,
    load_embeddings,
    sample_key_for,
)
from .metrics import compute_metrics
from .protocols import (
    build_modality_evaluation_data,
)

MODELS = tuple(MODEL_NAMES)

def create_virtual_pairings(
    fingerprint_subjects: Iterable[int],
    iris_subjects: Iterable[int],
) -> list[dict[str, int]]:
    """Pair independently held-out source subjects into virtual identities."""
    fingerprint_ids = [int(subject) for subject in fingerprint_subjects]
    iris_ids = [int(subject) for subject in iris_subjects]
    if len(fingerprint_ids) != len(iris_ids):
        raise ValueError("Virtual fusion requires equal fingerprint and iris cohort sizes.")
    return [
        {
            "virtual_id": virtual_id,
            "fingerprint_id": fingerprint_id,
            "iris_id": iris_id,
        }
        for virtual_id, (fingerprint_id, iris_id) in enumerate(
            zip(fingerprint_ids, iris_ids)
        )
    ]

def _subject_id(value: int) -> str:
    return str(int(value))

def _filter_data(data: dict, subjects: Iterable[int]) -> dict:
    wanted = {_subject_id(subject) for subject in subjects}
    indices = [i for i, value in enumerate(data["subjects"]) if _subject_id(value) in wanted]
    if not indices:
        raise ValueError("No embedding samples found for the requested fusion cohort.")
    sample_keys = data.get("sample_keys")
    if sample_keys is None:
        sample_keys = [
            sample_key_for(str(class_key), int(impression))
            for class_key, impression in zip(data["class_keys"], data["impressions"])
        ]
    return {
        "embeddings": data["embeddings"][indices],
        "subjects": data["subjects"][indices],
        "class_keys": [data["class_keys"][i] for i in indices],
        "sample_keys": [sample_keys[i] for i in indices],
        "impressions": data["impressions"][indices],
        "paths": [data["paths"][i] for i in indices],
    }

def _merge_embedding_data(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("No embedding artifacts matched the requested virtual cohort.")
    return {
        "embeddings": torch.cat([chunk["embeddings"] for chunk in chunks], dim=0),
        "subjects": torch.cat([chunk["subjects"] for chunk in chunks], dim=0),
        "class_keys": [key for chunk in chunks for key in chunk["class_keys"]],
        "sample_keys": [key for chunk in chunks for key in chunk["sample_keys"]],
        "impressions": torch.cat([chunk["impressions"] for chunk in chunks], dim=0),
        "paths": [path for chunk in chunks for path in chunk["paths"]],
    }

def load_virtual_population_embeddings(
    modality: str,
    model_name: str,
    subjects: Iterable[int],
    split_manifest: dict,
    root: Path = EMBEDDINGS_DIR,
) -> dict:
    """Load train/eval artifacts needed for an explicitly diagnostic cohort.

    This function is intentionally separate from the headline evaluation. A
    subject comes from the artifact corresponding to its stored modality split,
    so the resulting full-population run is not an unseen-subject estimate.
    """
    requested = {int(subject) for subject in subjects}
    chunks: list[dict] = []
    for split_name in ("train", "eval"):
        split_subjects = {
            int(subject)
            for subject in split_manifest[modality][f"{split_name}_subjects"]
        }
        selected = sorted(requested & split_subjects)
        if selected:
            artifact = load_embeddings(root / modality / model_name / f"{split_name}_embeddings.pt")
            chunks.append(_filter_data(artifact, selected))
    return _merge_embedding_data(chunks)

def _build_fusion_modality(
    data: dict,
    protocol,
    subject_key_fn,
    instance_level_enrollment: bool,
    instance_level_verification: bool,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build fusion-side templates/queries with virtual subject labels."""
    templates, queries = build_modality_evaluation_data(
        data,
        protocol,
        instance_level_enrollment=instance_level_enrollment,
        instance_level_verification=instance_level_verification,
        subject_key_fn=subject_key_fn,
    )

    # Make template keys globally unique while retaining the source biometric
    # class in the key. Subject-level templates use the virtual subject directly.
    mapped_templates: dict[str, torch.Tensor] = {}
    template_subjects: dict[str, str] = {}
    for key, embedding in templates.items():
        subject = str(subject_key_fn(key)) if instance_level_enrollment else str(key)
        mapped_key = f"{subject}::{key}" if instance_level_enrollment else subject
        mapped_templates[mapped_key] = embedding
        template_subjects[mapped_key] = subject

    if instance_level_verification:
        mapped_class_keys = []
        for key in queries["class_keys"]:
            subject = str(subject_key_fn(key))
            mapped_class_keys.append(f"{subject}::{key}")
    else:
        mapped_class_keys = [str(subject) for subject in queries["subjects"]]

    mapped_queries = dict(queries)
    mapped_queries["class_keys"] = mapped_class_keys
    mapped_queries["template_subjects"] = template_subjects
    mapped_queries["instance_level"] = instance_level_verification
    return mapped_templates, mapped_queries


def _pair_trials(fp_queries: dict, iris_queries: dict) -> list[tuple[str, int, int]]:
    subjects = sorted(set(fp_queries["groups"]) & set(iris_queries["groups"]))
    return [
        (subject, fp_index, iris_index)
        for subject in subjects
        for fp_index in fp_queries["groups"][subject]
        for iris_index in iris_queries["groups"][subject]
    ]

def _score_matrix(
    queries: torch.Tensor, templates: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, list[str]]:
    keys = sorted(templates)
    template_matrix = F.normalize(
        torch.stack([templates[key] for key in keys]).float(), p=2, dim=1
    )
    return F.normalize(queries.float(), p=2, dim=1) @ template_matrix.T, keys

def _genuine_mask(
    query_key: str,
    query_subject: str,
    template_key: str,
    template_instance_level: bool,
    template_subject: str,
) -> bool:
    if template_instance_level:
        if query_key.startswith(f"{query_subject}::"):
            return query_key == template_key
        return template_subject == query_subject
    return template_subject == query_subject

def evaluate_fusion_pair(
    fingerprint_data: dict,
    iris_data: dict,
    pairings: Iterable[dict[str, int]],
    fingerprint_weight: float = FUSION_FINGERPRINT_WEIGHT,
) -> dict:
    """Evaluate configurable synthetic fingerprint+iris fusion.

    The scoring path is vectorized and chunked. This is important for the
    instance-level protocol: 4,000 fused enrollment templates and 72,000
    cross-modal verification queries would otherwise require hundreds of
    millions of Python-level loop iterations.
    """
    pairing_records = sorted(
        (
            {
                "virtual_id": int(pairing["virtual_id"]),
                "fingerprint_id": int(pairing["fingerprint_id"]),
                "iris_id": int(pairing["iris_id"]),
            }
            for pairing in pairings
        ),
        key=lambda pairing: pairing["virtual_id"],
    )
    if not pairing_records:
        raise ValueError("At least one virtual fusion pairing is required.")

    def class_key_to_subject_id(subject_id: int) -> str:
        return f"{subject_id:03d}"

    fp_virtual_ids = {
        class_key_to_subject_id(p["fingerprint_id"]): str(p["virtual_id"])
        for p in pairing_records
    }
    iris_virtual_ids = {
        class_key_to_subject_id(p["iris_id"]): str(p["virtual_id"])
        for p in pairing_records
    }
    if len(fp_virtual_ids) != len(pairing_records) or len(iris_virtual_ids) != len(pairing_records):
        raise ValueError("Each source subject may appear in only one virtual fusion pairing.")

    fingerprint_subjects = [p["fingerprint_id"] for p in pairing_records]
    iris_subjects = [p["iris_id"] for p in pairing_records]

    fp_subject_key = lambda key: fp_virtual_ids[class_key_to_subject_id(int(str(key).split("_", 1)[0]))]
    iris_subject_key = lambda key: iris_virtual_ids[class_key_to_subject_id(int(str(key).split("_", 1)[0]))]

    fp_templates, fp_queries = _build_fusion_modality(
        _filter_data(fingerprint_data, fingerprint_subjects),
        FINGERPRINT_PROTOCOL,
        fp_subject_key,
        config.FUSION_VIRTUAL_ENROLLMENT,
        config.FUSION_VIRTUAL_VERIFICATION,
    )
    iris_templates, iris_queries = _build_fusion_modality(
        _filter_data(iris_data, iris_subjects),
        IRIS_PROTOCOL,
        iris_subject_key,
        config.FUSION_VIRTUAL_ENROLLMENT,
        config.FUSION_VIRTUAL_VERIFICATION,
    )

    trials = _pair_trials(fp_queries, iris_queries)
    if not trials:
        raise ValueError("No cross-modal trials could be constructed.")

    fp_scores, fp_keys = _score_matrix(fp_queries["embeddings"], fp_templates)
    iris_scores, iris_keys = _score_matrix(iris_queries["embeddings"], iris_templates)

    fp_subjects_by_template = fp_queries["template_subjects"]
    iris_subjects_by_template = iris_queries["template_subjects"]

    fp_weight = float(fingerprint_weight)
    if not 0.0 <= fp_weight <= 1.0:
        raise ValueError("fingerprint_weight must be between 0 and 1.")
    iris_weight = 1.0 - fp_weight
    fp_scale = np.sqrt(fp_weight)
    iris_scale = np.sqrt(iris_weight)

    # Build the Cartesian product of fingerprint and iris enrollment templates
    # within each virtual subject. Keep integer indices so the expensive score
    # construction below can use tensor gathers instead of Python string lookup.
    fused_fp_indices: list[int] = []
    fused_iris_indices: list[int] = []
    fused_subjects: list[str] = []
    for subject in sorted(set(fp_queries["groups"]) & set(iris_queries["groups"])):
        fp_subject_keys = [k for k in fp_keys if fp_subjects_by_template[k] == subject]
        iris_subject_keys = [k for k in iris_keys if iris_subjects_by_template[k] == subject]
        for fp_key in fp_subject_keys:
            fp_index = fp_keys.index(fp_key)
            for iris_key in iris_subject_keys:
                fused_fp_indices.append(fp_index)
                fused_iris_indices.append(iris_keys.index(iris_key))
                fused_subjects.append(subject)

    fused_fp_indices_t = torch.tensor(fused_fp_indices, dtype=torch.long)
    fused_iris_indices_t = torch.tensor(fused_iris_indices, dtype=torch.long)

    # Genuine lookup matrices are small compared with the full fused score
    # matrix (12M + 1.5M booleans for the default instance-level protocol).
    fp_query_keys = [str(key) for key in fp_queries["class_keys"]]
    iris_query_keys = [str(key) for key in iris_queries["class_keys"]]
    fp_query_subjects = [str(value) for value in fp_queries["subjects"]]
    iris_query_subjects = [str(value) for value in iris_queries["subjects"]]

    fp_genuine = torch.zeros((len(fp_query_keys), len(fp_keys)), dtype=torch.bool)
    iris_genuine = torch.zeros((len(iris_query_keys), len(iris_keys)), dtype=torch.bool)
    fp_key_to_index = {key: index for index, key in enumerate(fp_keys)}
    iris_key_to_index = {key: index for index, key in enumerate(iris_keys)}
    for row, (query_key, subject) in enumerate(zip(fp_query_keys, fp_query_subjects)):
        if config.FUSION_VIRTUAL_ENROLLMENT and fp_queries.get("instance_level", True):
            if query_key not in fp_key_to_index:
                raise ValueError(f"Verification key has no enrollment template: {query_key}")
            fp_genuine[row, fp_key_to_index[query_key]] = True
        else:
            # Subject-level verification against instance-level enrollment
            # treats every enrolled biometric instance of the same subject as
            # genuine. This is the mixed-granularity ablation.
            for index, key in enumerate(fp_keys):
                fp_genuine[row, index] = fp_subjects_by_template[key] == subject
    for row, (query_key, subject) in enumerate(zip(iris_query_keys, iris_query_subjects)):
        if config.FUSION_VIRTUAL_ENROLLMENT and iris_queries.get("instance_level", True):
            if query_key not in iris_key_to_index:
                raise ValueError(f"Verification key has no enrollment template: {query_key}")
            iris_genuine[row, iris_key_to_index[query_key]] = True
        else:
            for index, key in enumerate(iris_keys):
                iris_genuine[row, index] = iris_subjects_by_template[key] == subject

    # The cross-modal query pairs are kept as integer indices for fast gathers.
    trial_fp_indices = torch.tensor([fp_index for _, fp_index, _ in trials], dtype=torch.long)
    trial_iris_indices = torch.tensor([iris_index for _, _, iris_index in trials], dtype=torch.long)

    # Chunking keeps peak memory bounded while still moving the expensive
    # Cartesian score construction into optimized PyTorch tensor operations.
    chunk_size = int(getattr(config, "FUSION_SCORE_BATCH_SIZE", 1024))
    if chunk_size <= 0:
        raise ValueError("FUSION_SCORE_BATCH_SIZE must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_score_device = fp_scores.to(device)
    iris_score_device = iris_scores.to(device)
    fused_fp_indices_device = fused_fp_indices_t.to(device)
    fused_iris_indices_device = fused_iris_indices_t.to(device)

    genuine_chunks: list[np.ndarray] = []
    impostor_chunks: list[np.ndarray] = []

    # Feature-level fusion is mathematically identical to score-level fusion
    # here: both modality vectors/templates are L2-normalized and scaled by
    # sqrt(weight), so their concatenated cosine equals
    # w*FP_cosine + (1-w)*Iris_cosine. Reuse the exact score-level values.
    for start in range(0, len(trials), chunk_size):
        end = min(start + chunk_size, len(trials))
        fp_query = trial_fp_indices[start:end].to(device)
        iris_query = trial_iris_indices[start:end].to(device)
        fp_component = fp_score_device[fp_query][:, fused_fp_indices_device]
        iris_component = iris_score_device[iris_query][:, fused_iris_indices_device]
        fused_chunk = fp_weight * fp_component + iris_weight * iris_component

        fp_match = fp_genuine[trial_fp_indices[start:end]][:, fused_fp_indices_t]
        iris_match = iris_genuine[trial_iris_indices[start:end]][:, fused_iris_indices_t]
        genuine_mask = fp_match & iris_match

        genuine_chunks.append(fused_chunk[genuine_mask.to(device)].detach().cpu().numpy().astype(np.float32))
        impostor_chunks.append(fused_chunk[~genuine_mask.to(device)].detach().cpu().numpy().astype(np.float32))

    genuine_score = np.concatenate(genuine_chunks) if genuine_chunks else np.empty(0, dtype=np.float32)
    impostor_score = np.concatenate(impostor_chunks) if impostor_chunks else np.empty(0, dtype=np.float32)

    # Preserve the existing baseline semantics, but compute them directly from
    # the already-built modality score matrices rather than rescanning every
    # fused enrollment template for every trial.
    fp_baseline_genuine = []
    fp_baseline_impostor = []
    iris_baseline_genuine = []
    iris_baseline_impostor = []
    for start in range(0, len(trials), chunk_size):
        end = min(start + chunk_size, len(trials))
        fp_idx = trial_fp_indices[start:end]
        iris_idx = trial_iris_indices[start:end]
        fp_rows = fp_scores[fp_idx]
        iris_rows = iris_scores[iris_idx]
        fp_masks = fp_genuine[fp_idx]
        iris_masks = iris_genuine[iris_idx]
        fp_baseline_genuine.append(fp_rows[fp_masks].numpy().astype(np.float32))
        fp_baseline_impostor.append(fp_rows[~fp_masks].numpy().astype(np.float32))
        iris_baseline_genuine.append(iris_rows[iris_masks].numpy().astype(np.float32))
        iris_baseline_impostor.append(iris_rows[~iris_masks].numpy().astype(np.float32))

    fp_baseline_genuine = np.concatenate(fp_baseline_genuine) if fp_baseline_genuine else np.empty(0, dtype=np.float32)
    fp_baseline_impostor = np.concatenate(fp_baseline_impostor) if fp_baseline_impostor else np.empty(0, dtype=np.float32)
    iris_baseline_genuine = np.concatenate(iris_baseline_genuine) if iris_baseline_genuine else np.empty(0, dtype=np.float32)
    iris_baseline_impostor = np.concatenate(iris_baseline_impostor) if iris_baseline_impostor else np.empty(0, dtype=np.float32)

    def package(genuine: np.ndarray, impostor: np.ndarray) -> dict:
        return {
            "metrics": compute_metrics(genuine, impostor),
            "genuine": genuine,
            "impostor": impostor,
        }

    score_result = package(genuine_score, impostor_score)

    # Build feature-level fused enrollment vectors for the PCA visualization.
    # The score-level and feature-level authentication scores are equivalent
    # under the current sqrt-weighted normalized construction, but the actual
    # concatenated vectors are still useful for inspecting representation
    # structure.
    feature_vectors = []
    feature_labels = []
    for fp_index, iris_index, subject in zip(
        fused_fp_indices, fused_iris_indices, fused_subjects
    ):
        fp_vector = F.normalize(fp_templates[fp_keys[fp_index]].float(), p=2, dim=0)
        iris_vector = F.normalize(iris_templates[iris_keys[iris_index]].float(), p=2, dim=0)
        feature_vectors.append(torch.cat((fp_scale * fp_vector, iris_scale * iris_vector)))
        feature_labels.append(subject)
    feature_embeddings = (
        torch.stack(feature_vectors).cpu().numpy().astype(np.float32)
        if feature_vectors
        else np.empty((0, 0), dtype=np.float32)
    )

    return {
        "score_level": score_result,
        "feature_level": score_result,
        "fingerprint_baseline": package(fp_baseline_genuine, fp_baseline_impostor),
        "iris_baseline": package(iris_baseline_genuine, iris_baseline_impostor),
        "virtual_ids": [p["virtual_id"] for p in pairing_records],
        "pairings": pairing_records,
        "trial_count": len(trials),
        "feature_pca_embeddings": feature_embeddings,
        "feature_pca_labels": feature_labels,
        "virtual_protocol": (
            "synthetic_instance_level_fusion"
            if config.FUSION_VIRTUAL_ENROLLMENT and config.FUSION_VIRTUAL_VERIFICATION
            else "synthetic_subject_level_fusion"
            if not config.FUSION_VIRTUAL_ENROLLMENT and not config.FUSION_VIRTUAL_VERIFICATION
            else "synthetic_mixed_granularity_fusion"
        ),
        "fusion_instance_level_enrollment": config.FUSION_VIRTUAL_ENROLLMENT,
        "fusion_instance_level_verification": config.FUSION_VIRTUAL_VERIFICATION,
        "templates_per_subject": {
            "fingerprint": len([k for k in fp_keys if fp_subjects_by_template[k] == sorted(fp_subjects_by_template.values())[0]]) if fp_keys else 0,
            "iris": len([k for k in iris_keys if iris_subjects_by_template[k] == sorted(iris_subjects_by_template.values())[0]]) if iris_keys else 0,
        },
        "verification_templates_per_subject": {
            "fingerprint": len(fp_queries["groups"][sorted(fp_queries["groups"])[0]]) if fp_queries["groups"] else 0,
            "iris": len(iris_queries["groups"][sorted(iris_queries["groups"])[0]]) if iris_queries["groups"] else 0,
        },
    }

def load_split_manifest(path: Path | None = None) -> dict:
    path = path or (RESULTS_DIR / "subject_split.json")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def run_fusion_pair(
    fingerprint_model: str,
    iris_model: str,
    pairings: Iterable[dict[str, int]],
    embeddings_root: Path = EMBEDDINGS_DIR,
) -> dict:
    """Run headline fusion from held-out modality embedding artifacts."""
    fingerprint_data = load_embeddings(
        embeddings_root / "fingerprint" / fingerprint_model / "eval_embeddings.pt"
    )
    iris_data = load_embeddings(
        embeddings_root / "iris" / iris_model / "eval_embeddings.pt"
    )
    return evaluate_fusion_pair(fingerprint_data, iris_data, pairings)
