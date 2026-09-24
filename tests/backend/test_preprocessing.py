"""Phase 3 preprocessing unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.config.labels import build_label_map, list_collect_label_ids
from app.preprocessing.constants import FEATURE_DIM, LANDMARKS_PER_HAND
from app.preprocessing.features import detections_to_features, hands_to_features
from app.preprocessing.normalize import normalize_hand
from app.preprocessing.sequence import SequenceBuffer, fit_sequence_length
from app.vision.landmark_extractor import LandmarkExtractor


def _hand(origin: float = 0.4) -> np.ndarray:
    points = np.zeros((LANDMARKS_PER_HAND, 3), dtype=np.float32)
    points[:, 0] = origin
    points[:, 1] = 0.5
    for index in range(1, LANDMARKS_PER_HAND):
        points[index, 1] = 0.5 - 0.02 * index
    return points


def test_missing_hand_is_zeros() -> None:
    points, present = normalize_hand(None)
    assert present is False
    assert points.shape == (21, 3)
    assert np.allclose(points, 0)


def test_incomplete_landmarks_are_missing() -> None:
    points, present = normalize_hand([{"x": 0.1, "y": 0.2, "z": 0.0}])
    assert present is False
    assert np.allclose(points, 0)
    vector = hands_to_features(None, [{"x": 1.0}])
    assert vector[-1] == 0
    assert np.allclose(vector, 0)


def test_normalize_hand_is_wrist_relative_and_scaled() -> None:
    points, present = normalize_hand(_hand(0.7))
    assert present is True
    assert np.allclose(points[0], 0)
    assert np.max(np.linalg.norm(points, axis=1)) == pytest.approx(1.0, abs=1e-5)


def test_feature_vector_handles_one_and_two_hands() -> None:
    none = hands_to_features(None, None)
    assert none.shape == (FEATURE_DIM,)
    assert none[-2] == 0 and none[-1] == 0

    right_only = hands_to_features(None, _hand(0.6))
    assert right_only[-2] == 0
    assert right_only[-1] == 1
    assert not np.allclose(right_only[63:126], 0)

    both = detections_to_features({"left": _hand(0.3), "right": _hand(0.7)})
    assert both[-2] == 1 and both[-1] == 1


def test_landmark_extractor_does_not_crash_on_empty() -> None:
    extractor = LandmarkExtractor()
    vector = extractor.extract({})
    assert vector.shape == (FEATURE_DIM,)
    assert np.allclose(vector, 0)


def test_sequence_buffer_sliding_window() -> None:
    buffer = SequenceBuffer(sequence_length=4, feature_dim=FEATURE_DIM)
    assert buffer.is_ready() is False
    with pytest.raises(ValueError):
        buffer.get_sequence()
    for _ in range(4):
        buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert buffer.is_ready()
    window = buffer.get_sequence()
    assert window.shape == (4, FEATURE_DIM)
    buffer.append(np.ones(FEATURE_DIM, dtype=np.float32))
    latest = buffer.get_sequence()
    assert latest.shape == (4, FEATURE_DIM)
    assert np.allclose(latest[-1], 1)


def test_fit_sequence_length_pad_and_resample() -> None:
    short = np.ones((5, FEATURE_DIM), dtype=np.float32)
    padded = fit_sequence_length(short, 10)
    assert padded.shape == (10, FEATURE_DIM)
    assert np.allclose(padded[:5], 1)
    assert np.allclose(padded[5:], 0)

    long = np.arange(50 * FEATURE_DIM, dtype=np.float32).reshape(50, FEATURE_DIM)
    resampled = fit_sequence_length(long, 10)
    assert resampled.shape == (10, FEATURE_DIM)


def test_collect_labels_are_the_phase3_subset() -> None:
    ids = list_collect_label_ids()
    assert ids == [
        "HELLO",
        "THANK_YOU",
        "HELP",
        "YES",
        "NO",
        "WATER",
        "DOCTOR",
        "HOSPITAL",
        "I",
        "YOU",
    ]
    mapping = build_label_map()
    assert mapping["HELLO"] == 0
    assert mapping["YOU"] == 9
