"""MediaPipe hand landmark detection for the Python dataset collector."""

from __future__ import annotations

from typing import Any

import numpy as np


class HandDetector:
    """Detect left and right hand landmarks from a BGR video frame."""

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "The mediapipe package is required for webcam landmark collection. "
                "Python 3.13 may not have a wheel — use 3.11 if install fails. "
                "pip install mediapipe"
            ) from exc

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )

    def detect(self, frame: Any) -> dict[str, np.ndarray | None]:
        """Return {"left": (21,3)|None, "right": (21,3)|None}. Never raises."""
        left = None
        right = None
        if frame is None:
            return {"left": left, "right": right}

        try:
            import cv2
        except ImportError:
            return {"left": left, "right": right}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return {"left": left, "right": right}

        handedness_list = result.multi_handedness or []
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, handedness_list):
            label = handedness.classification[0].label if handedness.classification else "Unknown"
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            if label.lower() == "left":
                left = points
            elif label.lower() == "right":
                right = points
            elif left is None:
                left = points
            else:
                right = points
        return {"left": left, "right": right}

    def close(self) -> None:
        self._hands.close()
