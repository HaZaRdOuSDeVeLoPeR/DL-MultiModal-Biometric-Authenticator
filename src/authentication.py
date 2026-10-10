from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    EMBEDDINGS_DIR,
    FINGERPRINT_ENROLL_IMPRESSIONS,
    FINGERPRINT_VERIFY_IMPRESSIONS,
    FINGERPRINT_IMPRESSIONS_PER_FINGER,
    IRIS_ENROLL_IMPRESSIONS,
    IRIS_VERIFY_IMPRESSIONS,
    IRIS_IMPRESSIONS_PER_EYE,
)


@dataclass(frozen=True)
class ModalityProtocol:
    """Enrollment/verification protocol for one embedding modality."""

    name: str
    impressions_per_class: int
    enrollment_impressions: int
    verification_impressions: int

    def __post_init__(self) -> None:
        if self.enrollment_impressions + self.verification_impressions != self.impressions_per_class:
            raise ValueError(
                f"{self.name}: enrollment and verification counts must equal "
                f"{self.impressions_per_class}."
            )


FINGERPRINT_PROTOCOL = ModalityProtocol(
    name="fingerprint",
    impressions_per_class=FINGERPRINT_IMPRESSIONS_PER_FINGER,
    enrollment_impressions=FINGERPRINT_ENROLL_IMPRESSIONS,
    verification_impressions=FINGERPRINT_VERIFY_IMPRESSIONS,
)

IRIS_PROTOCOL = ModalityProtocol(
    name="iris",
    impressions_per_class=IRIS_IMPRESSIONS_PER_EYE,
    enrollment_impressions=IRIS_ENROLL_IMPRESSIONS,
    verification_impressions=IRIS_VERIFY_IMPRESSIONS,
)


# ============================================================
# Embedding loading and validation
# ============================================================


def sample_key_for(class_key: str, impression: int) -> str:
    """Build a traceability key; impression is not a classifier label."""
    return f"{class_key}_{int(impression):02d}"


def _with_sample_keys(data: dict) -> dict:
    """Derive sample keys for legacy artifacts without changing class keys."""
    if "sample_keys" not in data:
        data["sample_keys"] = [
            sample_key_for(str(class_key), int(impression))
            for class_key, impression in zip(data["class_keys"], data["impressions"])
        ]
    return data


def load_embeddings(file: Path) -> dict:
    """Load an embedding artifact and add sample metadata for old artifacts."""
    if not file.exists():
        raise FileNotFoundError(f"Embedding file not found: {file}")
    return _with_sample_keys(torch.load(file, map_location="cpu", weights_only=False))


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize a batch of embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding tensor, got {tuple(embeddings.shape)}")
    return F.normalize(embeddings.float(), p=2, dim=1)


def validate_evaluation_data(data: dict, protocol: ModalityProtocol) -> None:
    """Validate class/sample alignment for post-training authentication.

    A training class is subject+finger for fingerprints or subject+eye for
    iris. An impression/sample key identifies one observation within that
    class and is never used as the classifier target.
    """
    _with_sample_keys(data)
    required = ("embeddings", "subjects", "class_keys", "impressions", "paths", "sample_keys")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Embedding data is missing fields: {missing}")

    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Embedding metadata lengths do not match: {lengths}")
    if lengths["embeddings"] == 0:
        raise ValueError("Embedding data is empty.")
    if not isinstance(data["embeddings"], torch.Tensor) or data["embeddings"].ndim != 2:
        raise ValueError("'embeddings' must be a 2-D torch.Tensor.")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, class_key in enumerate(data["class_keys"]):
        grouped[str(class_key)].append(index)

    seen: set[str] = set()
    for class_key, indices in grouped.items():
        if len(indices) != protocol.impressions_per_class:
            raise ValueError(
                f"Class {class_key} has {len(indices)} impressions; expected "
                f"{protocol.impressions_per_class} for {protocol.name}."
            )
        impressions = sorted(int(data["impressions"][i]) for i in indices)
        expected_impressions = list(range(protocol.impressions_per_class))
        if impressions != expected_impressions:
            raise ValueError(
                f"Class {class_key} must contain impressions "
                f"0..{protocol.impressions_per_class - 1}."
            )
        for index in indices:
            expected_key = sample_key_for(class_key, int(data["impressions"][index]))
            actual_key = str(data["sample_keys"][index])
            if actual_key != expected_key:
                raise ValueError(
                    f"Sample metadata mismatch at index {index}: expected "
                    f"{expected_key}, got {actual_key}."
                )
            if actual_key in seen:
                raise ValueError(f"Duplicate sample key: {actual_key}")
            seen.add(actual_key)


def _data_subset(data: dict, indices: list[int]) -> dict:
    """Select aligned embedding and traceability metadata."""
    _with_sample_keys(data)
    return {
        "embeddings": data["embeddings"][indices],
        "subjects": data["subjects"][indices],
        "class_keys": [data["class_keys"][i] for i in indices],
        "sample_keys": [data["sample_keys"][i] for i in indices],
        "impressions": data["impressions"][indices],
        "paths": [data["paths"][i] for i in indices],
    }


def validate_fingerprint_evaluation(data: dict) -> None:
    """Backward-compatible fingerprint validation wrapper."""
    validate_evaluation_data(data, FINGERPRINT_PROTOCOL)


# ============================================================
# Enrollment / verification
# ============================================================


def split_enrollment_verification(
    data: dict,
    protocol: ModalityProtocol = FINGERPRINT_PROTOCOL,
) -> tuple[dict, dict]:
    """Split every class by impression into disjoint enrollment/verification."""
    validate_evaluation_data(data, protocol)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, class_key in enumerate(data["class_keys"]):
        grouped[str(class_key)].append(index)

    enrollment_indices: list[int] = []
    verification_indices: list[int] = []
    for class_key in sorted(grouped):
        indices = sorted(grouped[class_key], key=lambda i: int(data["impressions"][i]))
        enrollment_indices.extend(indices[:protocol.enrollment_impressions])
        verification_indices.extend(indices[protocol.enrollment_impressions:])
    return _data_subset(data, enrollment_indices), _data_subset(data, verification_indices)


# ============================================================
# Templates and scoring
# ============================================================


def create_templates(enrollment_data: dict) -> dict[str, torch.Tensor]:
    """Create one normalized template per subject+finger/eye class."""
    embeddings = normalize_embeddings(enrollment_data["embeddings"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, class_key in enumerate(enrollment_data["class_keys"]):
        grouped[str(class_key)].append(index)
    return {
        class_key: F.normalize(embeddings[indices].mean(dim=0), p=2, dim=0)
        for class_key, indices in grouped.items()
    }


def create_subject_templates(enrollment_data: dict, subject_key_fn=None) -> dict[str, torch.Tensor]:
    """Pool extracted class embeddings for an explicit subject-level protocol.

    This post-training operation can aggregate all fingers or both eyes for a
    subject. It never changes the subject+finger/eye training class labels.
    """
    if subject_key_fn is None:
        subject_key_fn = lambda key: str(key).split("_", 1)[0]
    embeddings = normalize_embeddings(enrollment_data["embeddings"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, class_key in enumerate(enrollment_data["class_keys"]):
        grouped[str(subject_key_fn(class_key))].append(index)
    return {
        subject_key: F.normalize(embeddings[indices].mean(dim=0), p=2, dim=0)
        for subject_key, indices in grouped.items()
    }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.dot(F.normalize(a.float(), p=2, dim=0), F.normalize(b.float(), p=2, dim=0)))


def generate_score_arrays(templates: dict[str, torch.Tensor], verification_data: dict, key_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one genuine score and all other-template impostor scores/query."""
    if key_fn is None:
        key_fn = lambda key: str(key)
    if not templates:
        raise ValueError("Cannot score against an empty template set.")
    embeddings = normalize_embeddings(verification_data["embeddings"])
    verification_keys = [str(key_fn(key)) for key in verification_data["class_keys"]]
    template_keys = sorted(templates)
    unknown = sorted(set(verification_keys) - set(template_keys))
    if unknown:
        raise ValueError(f"Verification keys have no enrollment template: {unknown[:5]}")
    matrix = F.normalize(torch.stack([templates[key] for key in template_keys]), p=2, dim=1)
    similarity = embeddings @ matrix.T
    key_to_index = {key: i for i, key in enumerate(template_keys)}
    genuine_indices = torch.tensor([key_to_index[key] for key in verification_keys], dtype=torch.long)
    rows = torch.arange(len(verification_keys), dtype=torch.long)
    genuine = similarity[rows, genuine_indices]
    mask = torch.ones_like(similarity, dtype=torch.bool)
    mask[rows, genuine_indices] = False
    return genuine.cpu(), similarity[mask].cpu()


def generate_scores(templates: dict[str, torch.Tensor], verification_data: dict) -> tuple[list[float], list[float]]:
    """Backward-compatible list-returning scorer."""
    genuine, impostor = generate_score_arrays(templates, verification_data)
    return genuine.tolist(), impostor.tolist()


def generate_detailed_scores(
    templates: dict[str, torch.Tensor],
    verification_embeddings: torch.Tensor,
    verification_class_keys: list[str],
    template_paths: dict[str, list[str]] | None = None,
    verification_paths: list[str] | None = None,
) -> list[dict]:
    """Generate metadata-rich comparisons for hard-impostor inspection."""
    if len(verification_class_keys) != len(verification_embeddings):
        raise ValueError("verification_class_keys length does not match embeddings.")
    if verification_paths is not None and len(verification_paths) != len(verification_embeddings):
        raise ValueError("verification_paths length does not match embeddings.")
    verification = normalize_embeddings(verification_embeddings)
    keys = sorted(templates)
    similarities = verification @ F.normalize(torch.stack([templates[key] for key in keys]).float(), p=2, dim=1).T
    output: list[dict] = []
    for row, verification_key in enumerate(verification_class_keys):
        for col, template_key in enumerate(keys):
            output.append({
                "template_class_key": template_key,
                "verification_class_key": verification_key,
                "score": float(similarities[row, col]),
                "genuine": template_key == verification_key,
                "verification_path": verification_paths[row] if verification_paths else None,
                "template_paths": (template_paths or {}).get(template_key, []),
            })
    return output


__all__ = [
    "ModalityProtocol", "FINGERPRINT_PROTOCOL", "IRIS_PROTOCOL", "sample_key_for",
    "load_embeddings", "normalize_embeddings", "validate_evaluation_data",
    "validate_fingerprint_evaluation", "split_enrollment_verification",
    "create_templates", "create_subject_templates", "cosine_similarity",
    "generate_score_arrays", "generate_scores", "generate_detailed_scores",
]
