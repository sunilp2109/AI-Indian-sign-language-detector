"""Fixed-length temporal windows for training and sliding-window inference."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from app.preprocessing.constants import FEATURE_DIM


def fit_sequence_length(
    frames: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Pad or uniformly resample to (sequence_length, feature_dim)."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected (frames, features), got {array.shape}")
    feature_dim = array.shape[1]
    target = int(sequence_length)
    if target < 1:
        raise ValueError("sequence_length must be >= 1")

    if array.shape[0] == 0:
        return np.zeros((target, feature_dim), dtype=np.float32)
    if array.shape[0] == target:
        return array.copy()
    if array.shape[0] < target:
        padded = np.zeros((target, feature_dim), dtype=np.float32)
        padded[: array.shape[0]] = array
        return padded

    indices = np.linspace(0, array.shape[0] - 1, num=target)
    indices = np.round(indices).astype(np.int64)
    return array[indices]


class SequenceBuffer:
    """Reusable frame buffer: frame → buffer → SEQUENCE_LENGTH window → model."""

    def __init__(
        self,
        sequence_length: int | None = None,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        if sequence_length is None:
            from app.config.settings import settings

            sequence_length = settings.sequence_length
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self._frames: Deque[np.ndarray] = deque(maxlen=self.sequence_length)

    def append(self, frame_features: np.ndarray | list[float]) -> None:
        vector = np.asarray(frame_features, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {vector.shape[0]}"
            )
        self._frames.append(vector)

    def is_ready(self) -> bool:
        return len(self._frames) >= self.sequence_length

    def get_sequence(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError(
                f"Buffer has {len(self._frames)}/{self.sequence_length} frames"
            )
        return np.stack(self._frames, axis=0)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
