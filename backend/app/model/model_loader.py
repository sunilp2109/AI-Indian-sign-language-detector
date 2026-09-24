"""Save and load ISL sequence checkpoints. Independent of FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.model.isl_model import ISLSequenceModel


CHECKPOINT_KEYS = ("model_state", "config", "label_map", "class_names")


def build_model(config: dict[str, Any]) -> ISLSequenceModel:
    return ISLSequenceModel(
        input_size=int(config["input_size"]),
        num_classes=int(config["num_classes"]),
        hidden_size=int(config.get("hidden_size", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.3)),
        rnn_type=str(config.get("rnn_type", "lstm")),
        bidirectional=bool(config.get("bidirectional", False)),
        sequence_length=config.get("sequence_length"),
    )


def save_checkpoint(
    path: Path,
    model: ISLSequenceModel,
    *,
    label_map: dict[str, int],
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": model.config_dict(),
        "label_map": label_map,
        "class_names": class_names,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    device: torch.device | str | None = None,
) -> tuple[ISLSequenceModel, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train with ml/scripts/train.py first."
        )
    map_location = device or "cpu"
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Invalid checkpoint (missing {missing}): {path}")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(map_location) if isinstance(map_location, str) else map_location)
    model.eval()
    return model, payload


def load_model(model_path: Path | None = None, device: str = "cpu") -> ISLSequenceModel:
    """Load a trained checkpoint. Does not invent a model if the file is missing."""
    if model_path is None:
        from app.config.settings import settings

        model_path = settings.resolved_model_path
    model, _payload = load_checkpoint(Path(model_path), device=device)
    return model
