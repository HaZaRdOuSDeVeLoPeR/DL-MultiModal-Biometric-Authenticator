import time
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.device import get_device
from typing import Sized, cast

from config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    USE_AMP,
    PREFETCH_FACTOR,
    USE_CHANNELS_LAST,
    AMP_DTYPE,
    USE_TORCH_COMPILE,
    SAVE_OPTIMIZER_STATE,
    MODELS_DIR,
    FINGERPRINT_TRAIN_CLASSES,
    IRIS_TRAIN_CLASSES,
    METRICS_DIR,
    RANDOM_SEED,
    SAVE_BEST_ONLY,
    MODEL_NAMES,
    MODALITIES,
)

from src.models import (
    BiometricEmbeddingModel,
    ModelName,
)


# ============================================================
# Training history
# ============================================================


@dataclass
class TrainingHistory:
    """
    Training metrics recorded for one model/modality experiment.
    """

    train_loss: list[float]

    epoch_times: list[float]
    total_training_time: float

    best_loss: float
    best_epoch: int


# ============================================================
# DataLoader
# ============================================================


def create_dataloader(dataset: Dataset) -> DataLoader:
    """
    Create the training DataLoader.

    The dataset already contains the subject-disjoint training
    split and provides the biometric class ID for each sample.
    """

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**loader_kwargs)


# ============================================================
# Training
# ============================================================


def train_one_epoch(
    model: BiometricEmbeddingModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average cross-entropy loss.
    """

    model.train()

    total_loss = torch.zeros((), device=device)
    total = 0

    for batch in dataloader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        if USE_CHANNELS_LAST and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        labels = batch["class_id"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        # --------------------------------------------------
        # Mixed-precision training
        # --------------------------------------------------

        if USE_AMP and device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=getattr(torch, AMP_DTYPE),
            ):

                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --------------------------------------------------
        # Standard training
        # --------------------------------------------------

        else:

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        # --------------------------------------------------
        # Accumulate loss
        # --------------------------------------------------

        # Accumulate on-device and synchronize only once per epoch.
        batch_size = images.size(0)
        total_loss += loss.detach() * batch_size
        total += batch_size

    average_loss = (total_loss / total).item()

    return average_loss


# ============================================================
# Save training history
# ============================================================


def save_training_history(history: TrainingHistory, model_name: ModelName, modality: str) -> None:
    """
    Save training history as JSON.
    """

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = METRICS_DIR / f"{model_name}_{modality}_training.json"

    data = {
        "model_name": model_name,
        "modality": modality,

        "train_loss": history.train_loss,

        "epoch_times": history.epoch_times,
        "total_training_time": (
            history.total_training_time
        ),

        "best_loss": history.best_loss,
        "best_epoch": history.best_epoch,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print(f"Training history: {output_file}")


# ============================================================
# CUDA timing
# ============================================================


def synchronize_cuda(device: torch.device) -> None:
    """
    Synchronize CUDA before accurate timing.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_state_dict(model: BiometricEmbeddingModel) -> dict:
    """Return an uncompiled state dict when torch.compile is enabled."""
    original = getattr(model, "_orig_mod", None)
    if original is not None:
        return original.state_dict()
    return model.state_dict()


# ============================================================
# Train one model
# ============================================================


def train_model(
    model_name: ModelName,
    dataset: Dataset,
    modality: str,
    device: torch.device | None = None,
) -> tuple[BiometricEmbeddingModel, TrainingHistory]:
    """
    Train one biometric embedding model.

    The classifier operates only on the biometric classes
    belonging to the training subjects.

    Fingerprint:
        400 subjects × 8 fingers = 3200 classes

    Iris:
        800 subjects × 2 eyes = 1600 classes
    """

    if device is None:
        device = get_device()

    # --------------------------------------------------
    # Determine number of training classes
    # --------------------------------------------------

    if modality == "fingerprint":
        num_classes = FINGERPRINT_TRAIN_CLASSES

    elif modality == "iris":
        num_classes = IRIS_TRAIN_CLASSES

    else:

        raise ValueError(
            f"Unsupported modality: {modality}"
        )

    print("\n")
    print("=" * 60)
    print(
        f"TRAINING: {model_name} | "
        f"{modality}"
    )
    print("=" * 60)

    print(
        f"Training classes : {num_classes}"
    )

    print(
        f"Training samples : {len(cast(Sized, dataset))}\n"
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    dataloader = create_dataloader(dataset)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = BiometricEmbeddingModel(
        model_name=model_name,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)
    if USE_CHANNELS_LAST and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = cast(BiometricEmbeddingModel, torch.compile(model))

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    # --------------------------------------------------
    # Mixed precision scaler
    # --------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and device.type == "cuda"
        ),
    )

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history = TrainingHistory(
        train_loss=[],

        epoch_times=[],
        total_training_time=0.0,

        best_loss=float("inf"),
        best_epoch=0,
    )

    # --------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------

    model_dir = MODELS_DIR / modality

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = model_dir / f"{model_name}_best.pt"

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1,):

        synchronize_cuda(device)

        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        scheduler.step()

        synchronize_cuda(device)

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        # --------------------------------------------------
        # Record history
        # --------------------------------------------------

        history.train_loss.append(train_loss)
        history.epoch_times.append(epoch_time)

        # --------------------------------------------------
        # Save best/last checkpoint
        # --------------------------------------------------

        should_save = (
            (SAVE_BEST_ONLY and train_loss < history.best_loss)
            or
            (not SAVE_BEST_ONLY and epoch == NUM_EPOCHS)
        )

        if should_save:

            history.best_loss = train_loss
            history.best_epoch = epoch

            torch.save(
                {
                    "model_name": model_name,
                    "modality": modality,

                    "embedding_dim": EMBEDDING_DIM,
                    "num_classes": num_classes,

                    "epoch": epoch,
                    "loss": train_loss,

                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "num_epochs": NUM_EPOCHS,

                    "random_seed": RANDOM_SEED,
                    "use_amp": USE_AMP,

                    "model_state_dict": checkpoint_state_dict(model),

                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ) if SAVE_OPTIMIZER_STATE else None,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"CE Loss: {train_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    # --------------------------------------------------
    # Training complete
    # --------------------------------------------------

    synchronize_cuda(device)

    history.total_training_time = (time.perf_counter() - total_start)

    print("-" * 60)

    print(
        f"Best loss     : "
        f"{history.best_loss:.4f}"
    )

    print(
        f"Best epoch    : "
        f"{history.best_epoch}"
    )

    print(
        f"Total time    : "
        f"{history.total_training_time:.2f}s"
    )

    print(
        f"Checkpoint    : "
        f"{checkpoint_path}"
    )

    print("=" * 60)

    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    save_training_history(
        history=history,
        model_name=model_name,
        modality=modality,
    )

    return model, history


# ============================================================
# Train all models
# ============================================================


def train_all_models(datasets: dict, device: torch.device | None = None) -> dict:
    """
    Train all configured architectures for both modalities.

    Existing completed experiments are skipped if both their
    checkpoint and training history exist.
    """

    if device is None:
        device = get_device()

    model_names: list[ModelName] = list(MODEL_NAMES)
    modalities = MODALITIES

    results = {}

    for model_name in model_names:

        for modality in modalities:

            checkpoint_path = MODELS_DIR / modality / f"{model_name}_best.pt"
            history_path = METRICS_DIR / f"{model_name}_{modality}_training.json"

            # --------------------------------------------------
            # Skip completed experiment
            # --------------------------------------------------

            if (checkpoint_path.exists() and history_path.exists()):

                print("Status: COMPLETED — skipping")

                results[f"{model_name}_{modality}"] = {
                    "status": "skipped",
                    "checkpoint": checkpoint_path,
                    "history": history_path,
                }

                continue

            # --------------------------------------------------
            # Training dataset
            # --------------------------------------------------

            dataset = datasets[f"{modality}_train"]

            # --------------------------------------------------
            # Train
            # --------------------------------------------------

            model, history = train_model(
                model_name=model_name,
                dataset=dataset,
                modality=modality,
                device=device,
            )

            results[f"{model_name}_{modality}"] = {
                "status": "trained",
                "history": history,
                "checkpoint": checkpoint_path,
                "history_file": history_path,
            }

            # --------------------------------------------------
            # Release model memory
            # --------------------------------------------------

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results