"""Read/write landmark takes without storing raw video."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.preprocessing.constants import LANDMARKS_PER_HAND
from app.preprocessing.features import hands_to_features
from ml.pipeline.paths import RAW_ROOT


def empty_hand_block(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, LANDMARKS_PER_HAND, 3), dtype=np.float32)


def take_dir(label: str, signer_id: str, raw_root: Path = RAW_ROOT) -> Path:
    return raw_root / label / signer_id


def next_take_stem(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("take_*.npz"))
    numbers = []
    for path in existing:
        suffix = path.stem.replace("take_", "")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"take_{max(numbers, default=0) + 1:04d}"


def detections_to_raw_frame(detections: dict[str, Any]) -> dict[str, Any]:
    left = detections.get("left")
    right = detections.get("right")
    return {
        "left": None if left is None else np.asarray(left, dtype=np.float32),
        "right": None if right is None else np.asarray(right, dtype=np.float32),
        "left_present": left is not None,
        "right_present": right is not None,
    }


def save_raw_take(
    *,
    label: str,
    signer_id: str,
    frames: list[dict[str, Any]],
    raw_root: Path = RAW_ROOT,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    if not frames:
        raise ValueError("Cannot save an empty take")

    directory = take_dir(label, signer_id, raw_root)
    stem = next_take_stem(directory)
    num_frames = len(frames)
    left = empty_hand_block(num_frames)
    right = empty_hand_block(num_frames)
    left_present = np.zeros(num_frames, dtype=np.uint8)
    right_present = np.zeros(num_frames, dtype=np.uint8)

    for index, frame in enumerate(frames):
        if frame.get("left") is not None:
            left[index] = np.asarray(frame["left"], dtype=np.float32)
            left_present[index] = 1
        if frame.get("right") is not None:
            right[index] = np.asarray(frame["right"], dtype=np.float32)
            right_present[index] = 1

    npz_path = directory / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        left=left,
        right=right,
        left_present=left_present,
        right_present=right_present,
    )

    meta = {
        "label": label,
        "signer_id": signer_id,
        "stem": stem,
        "num_frames": num_frames,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_saved": False,
        "source": "collect_data.py",
        "npz": npz_path.name,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = directory / f"{stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def load_raw_take(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    meta_path = npz_path.with_suffix(".json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "label": npz_path.parent.parent.name,
            "signer_id": npz_path.parent.name,
            "stem": npz_path.stem,
            "num_frames": int(arrays["left"].shape[0]),
            "video_saved": False,
        }
    return arrays, meta


def raw_take_to_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    num_frames = int(arrays["left"].shape[0])
    features = np.zeros((num_frames, hands_to_features(None, None).shape[0]), dtype=np.float32)
    left_present = arrays["left_present"].astype(bool)
    right_present = arrays["right_present"].astype(bool)
    for index in range(num_frames):
        left = arrays["left"][index] if left_present[index] else None
        right = arrays["right"][index] if right_present[index] else None
        features[index] = hands_to_features(left, right)
    return features


def iter_raw_takes(raw_root: Path = RAW_ROOT) -> list[Path]:
    if not raw_root.exists():
        return []
    return sorted(raw_root.glob("**/*.npz"))
