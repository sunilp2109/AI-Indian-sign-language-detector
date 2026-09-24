"""Project paths and dataset configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
ML_ROOT = PROJECT_ROOT / "ml"
DATASET_ROOT = ML_ROOT / "dataset"
RAW_ROOT = DATASET_ROOT / "raw"
PROCESSED_ROOT = DATASET_ROOT / "processed"
FRAMES_ROOT = PROCESSED_ROOT / "frames"
CONFIG_PATH = ML_ROOT / "config" / "dataset.json"
MODEL_CONFIG_PATH = ML_ROOT / "config" / "model.json"
MODELS_ROOT = ML_ROOT / "models"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def load_dataset_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if "sequence_length" not in config:
        raise ValueError("dataset.json must define sequence_length")
    return config


def load_model_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or MODEL_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_labels_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_dataset_config()
    path = Path(cfg.get("labels_path", "backend/config/labels.json"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
