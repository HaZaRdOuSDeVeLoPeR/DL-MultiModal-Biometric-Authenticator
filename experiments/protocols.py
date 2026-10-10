from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

import config

from src.authentication import (
    FINGERPRINT_PROTOCOL,
    IRIS_PROTOCOL,
    ModalityProtocol,
    create_templates,
    normalize_embeddings,
    sample_key_for,
    split_enrollment_verification,
    validate_evaluation_data,
)
from .metrics import MetricResult, compute_metrics

PROTOCOLS = {
    "fingerprint": FINGERPRINT_PROTOCOL,
    "iris": IRIS_PROTOCOL,
}


def subject_from_class_key(class_key: str) -> str:
    return str(class_key).split("_", 1)[0]


def _filter_data(data: dict, subjects: set[int | str]) -> dict:
    subject_values = {str(subject) for subject in subjects}
    indices = [i for i, subject in enumerate(data["subjects"]) if str(int(subject)) in subject_values]
    if not indices:
        raise ValueError("Subject filter produced no embedding samples.")
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

def score_with_templates(
    templates: dict[str, torch.Tensor],
    verification_data: dict,
    template_instance_level: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return genuine and impostor cosine scores at the configured granularity."""
    embeddings = F.normalize(verification_data["embeddings"].float(), p=2, dim=1)
    keys = sorted(templates)
    matrix = F.normalize(torch.stack([templates[key] for key in keys]).float(), p=2, dim=1)
    similarity = embeddings @ matrix.T
    query_keys = [str(key) for key in verification_data["class_keys"]]
    query_subjects = [str(key) for key in verification_data.get("subjects", query_keys)]

    genuine_mask = torch.zeros_like(similarity, dtype=torch.bool)
    if template_instance_level and verification_data.get("instance_level", True):
        key_to_index = {key: i for i, key in enumerate(keys)}
        for row, key in enumerate(query_keys):
            if key not in key_to_index:
                raise ValueError(f"Verification key has no enrollment template: {key}")
            genuine_mask[row, key_to_index[key]] = True
    elif template_instance_level:
        for row, subject in enumerate(query_subjects):
            genuine_mask[row] = torch.tensor(
                [subject_from_class_key(key) == subject for key in keys],
                dtype=torch.bool,
            )
    else:
        for row, subject in enumerate(query_subjects):
            if subject not in templates:
                raise ValueError(f"Verification subject has no enrollment template: {subject}")
            genuine_mask[row, keys.index(subject)] = True

    return similarity[genuine_mask].numpy(), similarity[~genuine_mask].numpy()

def build_modality_evaluation_data(
    data: dict,
    protocol: ModalityProtocol,
    instance_level_enrollment: bool,
    instance_level_verification: bool,
    subject_key_fn: Callable[[str], str] = subject_from_class_key,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build enrollment templates and verification queries independently."""
    validate_evaluation_data(data, protocol)
    enrollment, verification = split_enrollment_verification(data, protocol)

    if instance_level_enrollment:
        templates = create_templates(enrollment)
    else:
        templates, _ = build_subject_template_data(
            enrollment, verification, subject_key_fn=subject_key_fn
        )

    if instance_level_verification:
        query_embeddings = normalize_embeddings(verification["embeddings"])
        query_subjects = [str(subject_key_fn(key)) for key in verification["class_keys"]]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, subject in enumerate(query_subjects):
            groups[subject].append(index)
        queries = {
            "embeddings": query_embeddings,
            "class_keys": [str(key) for key in verification["class_keys"]],
            "subjects": query_subjects,
            "groups": dict(groups),
            "instance_level": True,
            "impressions": verification["impressions"],
        }
    else:
        queries = build_subject_verification_template_data(
            verification, protocol, subject_key_fn=subject_key_fn
        )
        queries["class_keys"] = list(queries["subjects"])
        queries["instance_level"] = False

    return templates, queries

def evaluate_single_modality(
    data: dict,
    protocol: ModalityProtocol,
    subjects: set[int | str] | None = None,
) -> tuple[MetricResult, dict, dict]:
    """Run one modality using the configured enrollment/verification granularity."""
    if subjects is not None:
        data = _filter_data(data, subjects)
    templates, verification = build_modality_evaluation_data(
        data, protocol,
        instance_level_enrollment=config.STANDALONE_VIRTUAL_ENROLLMENT,
        instance_level_verification=config.STANDALONE_VIRTUAL_VERIFICATION,
    )
    genuine, impostor = score_with_templates(
        templates, verification,
        template_instance_level=config.STANDALONE_VIRTUAL_ENROLLMENT,
    )
    return compute_metrics(genuine, impostor), templates, verification

def build_subject_template_data(
    enrollment: dict,
    verification: dict,
    subject_key_fn: Callable[[str], str] = subject_from_class_key,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Pool extracted class embeddings for a post-training subject protocol.

    ``subject_key_fn`` normally extracts the source subject from a class key.
    Synthetic fusion can instead map source subjects to shared virtual IDs;
    classifier class keys themselves remain unchanged.
    """
    enrollment_embeddings = F.normalize(enrollment["embeddings"].float(), p=2, dim=1)
    verification_embeddings = F.normalize(verification["embeddings"].float(), p=2, dim=1)

    enrollment_groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(enrollment["class_keys"]):
        enrollment_groups[str(subject_key_fn(key))].append(i)

    templates = {
        subject: F.normalize(enrollment_embeddings[indices].mean(dim=0), p=2, dim=0)
        for subject, indices in enrollment_groups.items()
    }
    query_subjects = [str(subject_key_fn(key)) for key in verification["class_keys"]]
    query_groups: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(query_subjects):
        query_groups[subject].append(index)
    return templates, {
        "embeddings": verification_embeddings,
        "subjects": query_subjects,
        "paths": verification["paths"],
        "groups": dict(query_groups),
    }

def build_subject_verification_template_data(
    verification: dict,
    protocol: ModalityProtocol,
    subject_key_fn: Callable[[str], str] = subject_from_class_key,
) -> dict:
    """Build one subject-level verification template per impression.

    Each verification impression is aggregated across every biometric class
    belonging to the subject. For the configured protocols this produces:

    * fingerprint: 3 templates/subject, each averaging one sample from all 8 fingers
    * iris: 6 templates/subject, each averaging one sample from both eyes

    This keeps verification templates at the same subject-level biometric scope
    as the enrollment templates while retaining multiple held-out verification
    templates.
    """
    # ``verification`` is already the post-split subset, so the generic
    # evaluation validator (which expects all impressions 0..N-1) cannot be
    # applied here. Validate the subset's shape and expected verification
    # impression range explicitly instead.
    required = ("embeddings", "class_keys", "impressions")
    missing = [key for key in required if key not in verification]
    if missing:
        raise ValueError(f"Verification data is missing fields: {missing}")
    if not isinstance(verification["embeddings"], torch.Tensor) or verification["embeddings"].ndim != 2:
        raise ValueError("Verification embeddings must be a 2-D torch.Tensor.")
    expected_impressions = set(range(
        protocol.enrollment_impressions,
        protocol.impressions_per_class,
    ))
    actual_impressions = {int(value) for value in verification["impressions"]}
    if actual_impressions != expected_impressions:
        raise ValueError(
            f"Verification impressions are {sorted(actual_impressions)}; expected "
            f"{sorted(expected_impressions)} for {protocol.name}."
        )
    embeddings = normalize_embeddings(verification["embeddings"])

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, class_key in enumerate(verification["class_keys"]):
        subject = str(subject_key_fn(class_key))
        impression = int(verification["impressions"][index])
        groups[(subject, impression)].append(index)

    subjects = sorted({subject for subject, _ in groups})
    impressions = sorted({impression for _, impression in groups})
    if not subjects or not impressions:
        raise ValueError("No subject-level verification groups were constructed.")

    # Every subject must have exactly one sample for every verification
    # impression in every class (finger/eye). This prevents silently creating
    # templates from incomplete biometric coverage.
    expected_class_keys: dict[str, set[str]] = defaultdict(set)
    for class_key in verification["class_keys"]:
        expected_class_keys[str(subject_key_fn(class_key))].add(str(class_key))
    for subject in subjects:
        subject_class_count = len(expected_class_keys[subject])
        for impression in impressions:
            indices = groups.get((subject, impression), [])
            if len(indices) != subject_class_count:
                raise ValueError(
                    f"Subject {subject}, impression {impression} has {len(indices)} "
                    f"samples; expected {subject_class_count}."
                )

    query_embeddings: list[torch.Tensor] = []
    query_subjects: list[str] = []
    query_impressions: list[int] = []
    query_groups: dict[str, list[int]] = defaultdict(list)

    for subject in subjects:
        for impression in impressions:
            indices = groups[(subject, impression)]
            query = F.normalize(embeddings[indices].mean(dim=0), p=2, dim=0)
            query_groups[subject].append(len(query_embeddings))
            query_embeddings.append(query)
            query_subjects.append(subject)
            query_impressions.append(impression)

    return {
        "embeddings": torch.stack(query_embeddings),
        "subjects": query_subjects,
        "impressions": query_impressions,
        "groups": dict(query_groups),
    }
