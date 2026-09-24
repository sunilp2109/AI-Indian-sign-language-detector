"""PyTorch-backed sequence predictor. Imported only when MODEL_MODE=real."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.model.isl_model import ISLSequenceModel
from app.model.model_loader import load_checkpoint


def _infer_length(sequence: np.ndarray) -> int:
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        return 1
    occupied = np.any(np.abs(sequence) > 1e-6, axis=1)
    length = int(occupied.sum())
    return max(1, min(length, sequence.shape[0]))


class RealPredictor:
    """Thread-safe wrapper around a loaded ISLSequenceModel."""

    source = "real"

    def __init__(
        self,
        model: ISLSequenceModel,
        class_names: list[str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.class_names = list(class_names)
        self.device = torch.device(device)
        self.input_size = int(model.input_size)
        self._lock = threading.Lock()
        self.model.to(self.device)
        self.model.eval()

    def predict_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected (time, features), got {array.shape}")
        if array.shape[1] != self.input_size:
            raise ValueError(
                f"Expected feature_dim={self.input_size}, got {array.shape[1]}"
            )
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        lengths = torch.tensor([_infer_length(array)], dtype=torch.long)
        with self._lock:
            with torch.no_grad():
                probs = self.model.predict_proba(tensor, lengths)[0]
        index = int(torch.argmax(probs).item())
        confidence = float(probs[index].item())
        if probs.numel() >= 2:
            top2 = torch.topk(probs, k=2).values
            gap = float(top2[0] - top2[1])
            if gap < 0.10:
                confidence = min(confidence, 0.45 + gap)
        sign = self.class_names[index] if 0 <= index < len(self.class_names) else None
        return {"sign": sign, "confidence": confidence, "source": self.source}


def load_real_predictor(path: Path, device: str = "cpu") -> RealPredictor:
    model, payload = load_checkpoint(path, device=device)
    class_names = [str(name) for name in payload.get("class_names") or []]
    if len(class_names) != model.num_classes:
        raise ValueError(
            f"Checkpoint class_names ({len(class_names)}) != num_classes ({model.num_classes})"
        )
    return RealPredictor(model, class_names, device=device)
