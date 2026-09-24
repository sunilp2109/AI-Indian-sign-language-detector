"""Build a fixed-length per-frame feature vector from left/right hands."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.preprocessing.constants import FEATURE_DIM, VALUES_PER_HAND
from app.preprocessing.normalize import normalize_hand


def hands_to_features(
    left: Any | None,
    right: Any | None,
) -> np.ndarray:
    """Return shape (FEATURE_DIM,) float32. Never raises on missing hands."""
    left_pts, left_present = normalize_hand(left)
    right_pts, right_present = normalize_hand(right)
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[0:VALUES_PER_HAND] = left_pts.reshape(-1)
    vector[VALUES_PER_HAND : VALUES_PER_HAND * 2] = right_pts.reshape(-1)
    vector[-2] = 1.0 if left_present else 0.0
    vector[-1] = 1.0 if right_present else 0.0
    return vector


def detections_to_features(detections: Mapping[str, Any] | None) -> np.ndarray:
    if not detections:
        return hands_to_features(None, None)
    return hands_to_features(detections.get("left"), detections.get("right"))
