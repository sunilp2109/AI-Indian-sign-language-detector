"""Load processed Phase 3 arrays into PyTorch datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.pipeline.paths import PROCESSED_ROOT


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
        *,
        augment: bool = False,
        noise_std: float = 0.02,
    ) -> None:
        if features.ndim != 3:
            raise ValueError(f"Expected X shape (N, T, F), got {features.shape}")
        if len(features) != len(labels) or len(features) != len(lengths):
            raise ValueError("X, y, and lengths must have the same N")
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.lengths = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
        self.augment = bool(augment)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.augment:
            noise = torch.randn_like(features) * self.noise_std
            if features.size(-1) >= 2:
                noise[..., -2:] = 0
            features = features + noise
            shift = int(torch.randint(-2, 3, (1,)).item())
            if shift:
                features = torch.roll(features, shifts=shift, dims=0)
        return features, self.labels[index], self.lengths[index]


def infer_lengths(features: np.ndarray) -> np.ndarray:
    mask = np.any(np.abs(features) > 1e-6, axis=2)
    lengths = mask.sum(axis=1).astype(np.int64)
    return np.clip(lengths, 1, features.shape[1])


def lengths_from_meta(meta: list[dict[str, Any]], sequence_length: int, fallback: np.ndarray) -> np.ndarray:
    if len(meta) != len(fallback):
        return fallback
    lengths = []
    for row, inferred in zip(meta, fallback, strict=True):
        if "num_frames" in row:
            lengths.append(int(min(max(int(row["num_frames"]), 1), sequence_length)))
        else:
            lengths.append(int(inferred))
    return np.asarray(lengths, dtype=np.int64)


def load_split_arrays(processed_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    split_dir = processed_root / split
    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"Missing {split} arrays under {split_dir}. Run create_sequences.py first."
        )
    features = np.load(x_path)
    labels = np.load(y_path)
    if features.shape[0] == 0:
        raise ValueError(
            f"{split} split is empty ({x_path}). Collect more samples before training/eval."
        )
    if features.ndim != 3:
        raise ValueError(f"{x_path} must have shape (N, T, F), got {features.shape}")
    meta: list[dict[str, Any]] = []
    meta_path = split_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inferred = infer_lengths(features)
    lengths = lengths_from_meta(meta, features.shape[1], inferred)
    return features, labels, lengths, meta


def load_label_artifacts(processed_root: Path = PROCESSED_ROOT) -> tuple[dict[str, int], list[str]]:
    map_path = processed_root / "label_map.json"
    names_path = processed_root / "class_names.json"
    if not map_path.is_file() or not names_path.is_file():
        raise FileNotFoundError(
            f"Missing label_map.json / class_names.json in {processed_root}. "
            "Run create_sequences.py first."
        )
    label_map = json.loads(map_path.read_text(encoding="utf-8"))
    class_names = json.loads(names_path.read_text(encoding="utf-8"))
    return label_map, class_names


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
) -> DataLoader:
    dataset = SequenceDataset(features, labels, lengths, augment=augment)
    return DataLoader(dataset, batch_size=max(1, batch_size), shuffle=shuffle)
