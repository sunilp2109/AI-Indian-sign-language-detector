"""Confidence thresholding and temporal smoothing for live predictions."""

from __future__ import annotations

from collections import deque
from typing import Any

from app.config.settings import settings


def accept_prediction(confidence: float, threshold: float | None = None) -> bool:
    limit = settings.confidence_threshold if threshold is None else threshold
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return False
    return value >= float(limit)


class PredictionSmoother:
    """Majority vote over a rolling window, with confidence averaged per sign."""

    def __init__(self, window: int = 5, threshold: float | None = None) -> None:
        self.window = max(1, int(window))
        self.threshold = (
            settings.confidence_threshold if threshold is None else float(threshold)
        )
        self._votes: deque[tuple[str | None, float]] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._votes.clear()

    def push(self, sign: str | None, confidence: float) -> dict[str, Any]:
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if sign == "":
            sign = None
        self._votes.append((sign, score))

        counts: dict[str, int] = {}
        conf_sum: dict[str, float] = {}
        for label, value in self._votes:
            if label is None:
                continue
            counts[label] = counts.get(label, 0) + 1
            conf_sum[label] = conf_sum.get(label, 0.0) + value

        if not counts:
            return {
                "sign": None,
                "confidence": 0.0,
                "accepted": False,
                "votes": 0,
            }

        winner = max(counts, key=lambda key: (counts[key], conf_sum[key] / counts[key]))
        average = conf_sum[winner] / counts[winner]
        needed = (len(self._votes) + 1) // 2
        accepted = counts[winner] >= needed and accept_prediction(average, self.threshold)
        return {
            "sign": winner,
            "confidence": float(average),
            "accepted": bool(accepted),
            "votes": counts[winner],
        }
