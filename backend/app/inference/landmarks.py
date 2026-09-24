"""Turn WebSocket/HTTP landmark payloads into 128-d feature frames.

Invalid or missing hands become an empty (all-zero) frame instead of crashing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.preprocessing.constants import FEATURE_DIM
from app.preprocessing.features import hands_to_features


def _points_from_hand(hand: Any) -> Any:
    if hand is None:
        return None
    if isinstance(hand, dict):
        return hand.get("landmarks") or hand.get("points") or hand.get("coords")
    return hand


def parse_hands_payload(payload: Any) -> tuple[Any, Any]:
    """Return (left, right) landmark arrays from mixed client shapes."""
    if payload is None:
        return None, None

    if isinstance(payload, dict):
        if "hands" in payload:
            return parse_hands_payload(payload.get("hands"))
        left = payload.get("left") or payload.get("Left")
        right = payload.get("right") or payload.get("Right")
        if left is not None or right is not None:
            return _points_from_hand(left), _points_from_hand(right)

    if not isinstance(payload, list):
        return None, None

    left = right = None
    for hand in payload:
        if not isinstance(hand, dict):
            continue
        label = str(hand.get("label") or hand.get("handedness") or "").lower()
        points = _points_from_hand(hand)
        if label == "left":
            left = points
        elif label == "right":
            right = points
    return left, right


def features_from_landmarks(payload: Any) -> np.ndarray:
    try:
        left, right = parse_hands_payload(payload)
        return hands_to_features(left, right)
    except (TypeError, ValueError, AttributeError):
        return hands_to_features(None, None)


def parse_feature_vector(value: Any, feature_dim: int = FEATURE_DIM) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape[0] != feature_dim or not np.isfinite(vector).all():
        return None
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def hands_detected_from_features(vector: np.ndarray) -> list[str]:
    if vector.size < 2:
        return []
    detected: list[str] = []
    if float(vector[-2]) >= 0.5:
        detected.append("Left")
    if float(vector[-1]) >= 0.5:
        detected.append("Right")
    return detected
