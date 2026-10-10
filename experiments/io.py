from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import EMBEDDINGS_DIR, METRICS_DIR, RESULTS_DIR
from src.authentication import load_embeddings


def load_history(model_name: str, modality: str) -> dict:
    path = METRICS_DIR / f"{model_name}_{modality}_training.json"
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_embedding_artifact(modality: str, model_name: str, split: str = "eval", root: Path = EMBEDDINGS_DIR) -> dict:
    path = root / modality / model_name / f"{split}_embeddings.pt"
    return load_embeddings(path)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(value), file, indent=2, allow_nan=False)


def write_metrics_json(path: Path, records: Iterable[dict]) -> None:
    write_json(path, list(records))


def write_metrics_csv(path: Path, records: Iterable[dict]) -> None:
    rows = [_jsonable(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scores(path: Path, genuine: np.ndarray, impostor: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, genuine=np.asarray(genuine, dtype=np.float32), impostor=np.asarray(impostor, dtype=np.float32))


def experiment_dir(name: str) -> Path:
    path = RESULTS_DIR / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
