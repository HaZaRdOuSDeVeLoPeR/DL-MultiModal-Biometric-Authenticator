from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    EMBEDDINGS_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    EMBEDDING_DIM,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    USE_CHANNELS_LAST,
    AMP_DTYPE,
    MODEL_NAMES,
    MODALITIES,
)
from src.models import BiometricEmbeddingModel, ModelName
from utils.device import get_device


MODELS: list[ModelName] = list(MODEL_NAMES)


# ============================================================
# DataLoader
# ============================================================


def create_dataloader(dataset: Dataset) -> DataLoader:
    """Create an inference loader while preserving metadata order."""
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**loader_kwargs)


# ============================================================
# Model loading
# ============================================================


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> BiometricEmbeddingModel:
    """Load a trained biometric model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BiometricEmbeddingModel(
        model_name=checkpoint["model_name"],
        embedding_dim=checkpoint["embedding_dim"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Embedding extraction
# ============================================================


@torch.inference_mode()
def extract_embeddings(model: BiometricEmbeddingModel, dataset: Dataset, device: torch.device) -> dict:
    """Extract normalized embeddings and aligned metadata."""
    dataloader = create_dataloader(dataset)
    embeddings: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    class_keys: list[str] = []
    sample_keys: list[str] = []
    impressions: list[torch.Tensor] = []
    paths: list[str] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        if USE_CHANNELS_LAST and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, AMP_DTYPE),
            enabled=device.type == "cuda",
        ):
            vectors = model.forward_embedding(images)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

        embeddings.append(vectors.cpu())
        subjects.append(batch["subject"].cpu())
        class_keys.extend(batch["class_key"])
        sample_keys.extend(batch["sample_key"])
        impressions.append(batch["impression"].cpu())
        paths.extend(batch["path"])

    return {
        "embeddings": torch.cat(embeddings),
        "subjects": torch.cat(subjects),
        "class_keys": class_keys,
        "sample_keys": sample_keys,
        "impressions": torch.cat(impressions),
        "paths": paths,
    }


# ============================================================
# Saving
# ============================================================


def save_embeddings(data: dict, output_file: Path, model_name: ModelName, modality: str, split_name: str) -> None:
    """Save embeddings and associated metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": data["embeddings"],
            "subjects": data["subjects"],
            "class_keys": data["class_keys"],
            "sample_keys": data["sample_keys"],
            "impressions": data["impressions"],
            "paths": data["paths"],
            "model_name": model_name,
            "modality": modality,
            "split": split_name,
            "embedding_dim": EMBEDDING_DIM,
        },
        output_file,
    )


# ============================================================
# Extract + save
# ============================================================


def extract_and_save_embeddings(
    model_name: ModelName,
    modality: str,
    split_name: str,
    dataset: Dataset,
    device: torch.device,
) -> None:
    """Extract and save one model/modality/split combination."""
    checkpoint_path = MODELS_DIR / modality / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(checkpoint_path, device)
    data = extract_embeddings(model, dataset, device)
    output_file = EMBEDDINGS_DIR / modality / model_name / f"{split_name}_embeddings.pt"
    save_embeddings(data, output_file, model_name, modality, split_name)
    print(f"{modality:<12} {model_name:<18} {split_name:<8} {tuple(data['embeddings'].shape)}")
    del model


# ============================================================
# Extract everything
# ============================================================


def extract_all_embeddings(datasets: dict[str, Dataset], device: torch.device | None = None) -> None:
    """Extract embeddings for every trained model, modality, and split."""
    if device is None:
        device = get_device()

    print("\n")
    print("=" * 60)
    print("EMBEDDING EXTRACTION")
    print("=" * 60)
    for modality in MODALITIES:
        for model_name in MODELS:
            for split_name in ("train", "eval"):
                dataset_key = f"{modality}_{split_name}"
                if dataset_key not in datasets:
                    raise KeyError(f"Dataset not found: {dataset_key}")
                extract_and_save_embeddings(model_name, modality, split_name, datasets[dataset_key], device)
    print("=" * 60)
