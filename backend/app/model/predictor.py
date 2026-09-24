"""Run inference and return sign + confidence.

MODEL_MODE=mock uses MockPredictor (explicit demo, not trained accuracy).
MODEL_MODE=real uses RealPredictor and a Phase 4 checkpoint. Missing files
fail loudly instead of falling back to mock.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from app.preprocessing.constants import FEATURE_DIM


class Predictor(Protocol):
    source: str
    input_size: int

    def predict_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        """Return {"sign", "confidence", "source"} for a (T, F) window."""


def _hands_present(sequence: np.ndarray) -> bool:
    array = np.asarray(sequence, dtype=np.float32)
    if array.size == 0 or not np.any(np.abs(array) > 1e-6):
        return False
    if array.ndim == 2 and array.shape[1] >= 2:
        # Use the latest frame so empty/missing hands stop the demo immediately.
        return bool(np.any(array[-1, -2:] >= 0.5))
    return True


class MockPredictor:
    """Controlled demo predictor. Never claims trained ISL accuracy."""

    source = "mock"
    input_size = FEATURE_DIM

    def __init__(
        self,
        class_names: list[str],
        confidence: float = 0.90,
        hold: int = 8,
    ) -> None:
        names = [name for name in class_names if name]
        self.class_names = names or ["HELLO"]
        self.confidence = float(confidence)
        self.hold = max(1, int(hold))
        self._index = 0
        self._remaining = 0
        self._current: str | None = None

    def reset(self) -> None:
        self._remaining = 0
        self._current = None

    def predict_sequence(self, sequence: np.ndarray) -> dict[str, Any]:
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 2 or not _hands_present(array):
            self._remaining = 0
            self._current = None
            return {"sign": None, "confidence": 0.0, "source": self.source}

        if self._remaining <= 0 or self._current is None:
            self._current = self.class_names[self._index % len(self.class_names)]
            self._index += 1
            self._remaining = self.hold
        self._remaining -= 1
        return {
            "sign": self._current,
            "confidence": self.confidence,
            "source": self.source,
        }
