"""MediaPipe pose landmark detection. Designed for a later expansion of Phase 2."""

from typing import Any


class PoseDetector:
    """Detect body pose landmarks. Optional in the MVP; reserved for expansion."""

    def detect(self, frame: Any) -> dict[str, Any]:
        raise NotImplementedError("Pose detection is reserved for a later vision phase.")
