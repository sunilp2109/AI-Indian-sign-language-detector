"""Normalize a single hand so position and camera distance are less important."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.preprocessing.constants import (
    COORDS_PER_LANDMARK,
    LANDMARKS_PER_HAND,
    MIDDLE_MCP_INDEX,
    SCALE_EPSILON,
    WRIST_INDEX,
)


def missing_hand() -> np.ndarray:
    return np.zeros((LANDMARKS_PER_HAND, COORDS_PER_LANDMARK), dtype=np.float32)


def _coords_from_point(point: object) -> list[float] | None:
    try:
        if isinstance(point, dict):
            return [
                float(point.get("x") or 0.0),
                float(point.get("y") or 0.0),
                float(point.get("z") or 0.0),
            ]
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            z = float(point[2]) if len(point) > 2 else 0.0
            return [x, y, z]
    except (TypeError, ValueError):
        return None
    return None


def as_hand_array(points: Sequence | np.ndarray | None) -> np.ndarray | None:
    """Return (21, 3) float32 or None if the hand is missing/invalid."""
    if points is None:
        return None
    if isinstance(points, (list, tuple)) and points and isinstance(points[0], dict):
        coords = [_coords_from_point(point) or [0.0, 0.0, 0.0] for point in points]
        try:
            array = np.asarray(coords, dtype=np.float32)
        except (TypeError, ValueError):
            return None
    else:
        try:
            array = np.asarray(points, dtype=np.float32)
        except (TypeError, ValueError):
            return None
    if array.size == 0 or np.isnan(array).all():
        return None
    if array.ndim == 1:
        if array.size != LANDMARKS_PER_HAND * COORDS_PER_LANDMARK:
            return None
        array = array.reshape(LANDMARKS_PER_HAND, COORDS_PER_LANDMARK)
    if array.shape != (LANDMARKS_PER_HAND, COORDS_PER_LANDMARK):
        return None
    if not np.isfinite(array).any():
        return None
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def _scale_for_hand(centered: np.ndarray) -> float:
    """Use palm size when possible, otherwise the largest wrist-relative radius."""
    palm = float(np.linalg.norm(centered[MIDDLE_MCP_INDEX]))
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    return max(palm, radius, SCALE_EPSILON)


def normalize_hand(points: Sequence | np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Wrist-origin, scale-normalized hand.

    Returns (points, present). Missing hands become zeros and present=False.
    """
    array = as_hand_array(points)
    if array is None:
        return missing_hand(), False

    wrist = array[WRIST_INDEX]
    centered = array - wrist
    scale = _scale_for_hand(centered)
    normalized = (centered / scale).astype(np.float32)
    return normalized, True


def normalize_landmarks(landmarks: Sequence[float] | np.ndarray) -> list[float]:
    """Normalize one hand given as 21x3 or a flat length-63 vector."""
    normalized, _present = normalize_hand(landmarks)
    return normalized.reshape(-1).astype(np.float32).tolist()
