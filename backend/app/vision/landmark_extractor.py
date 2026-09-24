"""Turn detector outputs into a numerical feature vector."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.preprocessing.features import detections_to_features


class LandmarkExtractor:
    """Assemble left/right hands into the shared FEATURE_DIM vector.

    Missing hands, a single visible hand, and temporary dropouts become zeros
    plus presence flags. Never raises for empty detections.
    """

    def extract(self, detections: Mapping[str, Any] | None) -> np.ndarray:
        return detections_to_features(detections)
