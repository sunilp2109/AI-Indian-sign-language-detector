"""Phase 5 inference tests: mock mode, real mode, WebSocket, invalid landmarks.

Random/synthetic checkpoints are not an ISL accuracy claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(ROOT))

from app.inference.service import (  # noqa: E402
    InferenceService,
    reset_inference_service,
)
from app.inference.session import InferenceSession  # noqa: E402
from app.language.confidence import PredictionSmoother, accept_prediction  # noqa: E402
from app.main import app  # noqa: E402
from app.model.predictor import MockPredictor  # noqa: E402
from app.preprocessing.constants import FEATURE_DIM  # noqa: E402
from app.preprocessing.features import hands_to_features  # noqa: E402

client = TestClient(app)


def _hand(offset: float = 0.1) -> list[dict[str, float]]:
    return [{"x": offset + i * 0.01, "y": 0.2, "z": 0.0} for i in range(21)]


def _occupied_sequence(frames: int = 30) -> list[list[float]]:
    array = np.zeros((frames, FEATURE_DIM), dtype=np.float32)
    array[:, 0] = 0.2
    array[:, -1] = 1.0
    return array.tolist()


def _empty_sequence(frames: int = 30) -> list[list[float]]:
    return np.zeros((frames, FEATURE_DIM), dtype=np.float32).tolist()


@pytest.fixture(autouse=True)
def restore_service() -> None:
    yield
    reset_inference_service(None)


def test_smoother_majority_and_threshold() -> None:
    smoother = PredictionSmoother(window=3, threshold=0.8)
    first = smoother.push("HELLO", 0.9)
    assert first["accepted"] is True
    smoother.push("YES", 0.9)
    third = smoother.push("YES", 0.91)
    assert third["sign"] == "YES"
    assert third["accepted"] is True
    low = PredictionSmoother(window=1, threshold=0.8).push("HELLO", 0.4)
    assert low["accepted"] is False
    assert accept_prediction(0.8, 0.8) is True


def test_mock_session_waits_for_window() -> None:
    session = InferenceSession(
        MockPredictor(["HELLO", "YES"], hold=1),
        sequence_length=4,
        feature_dim=FEATURE_DIM,
        threshold=0.8,
        smoothing_window=1,
        mode="mock",
    )
    vector = hands_to_features(None, _hand())
    for _ in range(3):
        status = session.push_features(vector)
        assert status["ready"] is False
        assert status["accepted"] is False
        assert status["sign"] is None
    ready = session.push_features(vector)
    assert ready["ready"] is True
    assert ready["source"] == "mock"
    assert ready["sign"] == "HELLO"
    assert ready["accepted"] is True
    assert ready["sentence"]
    assert ready["gloss"] == "HELLO"


def test_empty_frame_after_ready_is_not_accepted() -> None:
    session = InferenceSession(
        MockPredictor(["HELLO", "YES"], hold=99),
        sequence_length=3,
        feature_dim=FEATURE_DIM,
        threshold=0.8,
        smoothing_window=1,
        mode="mock",
    )
    vector = hands_to_features(None, _hand())
    for _ in range(3):
        session.push_features(vector)
    empty = session.push_features(hands_to_features(None, None))
    assert empty["accepted"] is False
    assert empty["sign"] is None
    assert empty["hands_detected"] == []


def test_mock_session_handles_missing_landmarks() -> None:
    session = InferenceSession(
        MockPredictor(["HELLO"], hold=1),
        sequence_length=3,
        smoothing_window=1,
        mode="mock",
    )
    empty = session.handle({"type": "landmarks", "hands": []})
    assert empty["type"] == "status"
    invalid = session.handle({"type": "landmarks", "hands": [{"label": "Right", "landmarks": [{"x": 1}]}]})
    assert invalid["type"] in {"status", "prediction"}
    garbage = session.handle("not-an-object")
    assert garbage["type"] == "error"
    reset = session.handle({"type": "reset"})
    assert reset["buffer_frames"] == 0


def test_predict_mock_sequence() -> None:
    reset_inference_service(
        InferenceService(
            mode="mock",
            predictor=MockPredictor(["HELLO"], hold=1),
            ready=True,
            sequence_length=30,
            threshold=0.8,
            smoothing_window=1,
        )
    )
    response = client.post("/predict", json={"sequence": _occupied_sequence()})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert body["mode"] == "mock"
    assert body["ready"] is True
    assert body["sign"]
    assert body["accepted"] is True
    assert body["sentence"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_mock_empty_sequence_not_accepted() -> None:
    reset_inference_service(
        InferenceService(
            mode="mock",
            predictor=MockPredictor(["HELLO"], hold=1),
            ready=True,
            sequence_length=30,
            threshold=0.8,
            smoothing_window=1,
        )
    )
    response = client.post("/predict", json={"sequence": _empty_sequence()})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert body["accepted"] is False
    assert body["sign"] is None


def test_mock_ignores_window_after_hands_leave() -> None:
    predictor = MockPredictor(["HELLO"], hold=1)
    occupied = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    occupied[:, -1] = 1.0
    occupied[-1, -1] = 0.0
    result = predictor.predict_sequence(occupied)
    assert result["sign"] is None
    assert result["confidence"] == 0.0


def test_websocket_mock_fills_buffer_then_predicts() -> None:
    reset_inference_service(
        InferenceService(
            mode="mock",
            predictor=MockPredictor(["HELLO"], hold=1),
            ready=True,
            sequence_length=5,
            threshold=0.8,
            smoothing_window=1,
        )
    )
    frame = {"type": "landmarks", "hands": [{"label": "Right", "landmarks": _hand()}]}
    with client.websocket_connect("/ws/translate") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello"
        assert hello["mode"] == "mock"
        last = None
        for _ in range(5):
            socket.send_json(frame)
            last = socket.receive_json()
        assert last["ready"] is True
        assert last["source"] == "mock"
        assert last["sign"] == "HELLO"
        assert last["accepted"] is True
        socket.send_text("not-json")
        error = socket.receive_json()
        assert error["type"] == "error"


def test_real_mode_missing_checkpoint_does_not_mock() -> None:
    service = InferenceService.real(Path("/no/such/best_model.pth"))
    assert service.mode == "real"
    assert service.ready is False
    assert service.predictor is None
    assert service.load_error
    session = service.new_session()
    result = session.handle({"type": "landmarks", "hands": [{"label": "Right", "landmarks": _hand()}]})
    assert result["type"] == "error"
    assert result["code"] == "model_unavailable"
    assert result["source"] != "mock" or "not loaded" in (result["message"] or "").lower()


def test_predict_real_mode_missing_file_is_503() -> None:
    reset_inference_service(InferenceService.real(Path("/no/such/best_model.pth")))
    response = client.post("/predict", json={"sequence": _occupied_sequence()})
    assert response.status_code == 503
    assert "mock" not in response.json()["detail"].lower() or "checkpoint" in response.json()["detail"].lower()


def test_real_predictor_and_session(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from app.model.isl_model import ISLSequenceModel
    from app.model.model_loader import save_checkpoint
    from app.model.real_predictor import load_real_predictor

    model = ISLSequenceModel(
        input_size=FEATURE_DIM,
        num_classes=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        sequence_length=8,
    )
    path = tmp_path / "best_model.pth"
    save_checkpoint(
        path,
        model,
        label_map={"HELLO": 0, "YES": 1},
        class_names=["HELLO", "YES"],
    )
    predictor = load_real_predictor(path, device="cpu")
    sequence = np.random.randn(8, FEATURE_DIM).astype(np.float32)
    sequence[:, -1] = 1.0
    raw = predictor.predict_sequence(sequence)
    assert raw["source"] == "real"
    assert raw["sign"] in {"HELLO", "YES"}
    assert 0.0 <= raw["confidence"] <= 1.0

    service = InferenceService(
        mode="real",
        predictor=predictor,
        ready=True,
        sequence_length=8,
        threshold=0.0,
        smoothing_window=1,
    )
    session = service.new_session(smoothing=False)
    last = None
    vector = hands_to_features(None, _hand(0.3))
    for _ in range(8):
        last = session.push_features(vector)
    assert last["ready"] is True
    assert last["source"] == "real"
    assert last["sign"] in {"HELLO", "YES"}
    assert last["gloss"] == last["sign"]

    reset_inference_service(service)
    response = client.post("/predict", json={"sequence": sequence.tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "real"
    assert body["mode"] == "real"
    assert body["sign"] in {"HELLO", "YES"}
    assert "accepted" in body
    _ = torch  # imported to skip cleanly when torch is missing
