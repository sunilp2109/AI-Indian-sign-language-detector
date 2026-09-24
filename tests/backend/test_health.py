"""Health, labels, and request-validation API tests."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "isl-bridge-api"
    assert payload["phase"] == 7
    assert payload["websocket"] == "/ws/translate"
    assert payload["model_mode"] in {"mock", "real"}
    assert "model_available" in payload
    assert "model_loaded" in payload


def test_root_ok() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["health"] == "/health"
    assert body["websocket"] == "/ws/translate"
    assert body["phase"] == 7


def test_labels_loaded_from_config() -> None:
    response = client.get("/labels")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 10
    ids = [item["id"] for item in payload["labels"]]
    assert "HELLO" in ids
    assert "DOCTOR" in ids


def test_predict_requires_payload() -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_wrong_feature_dim() -> None:
    response = client.post("/predict", json={"sequence": [[0.0, 1.0], [0.0, 1.0]]})
    assert response.status_code == 422


def test_sentence_requires_glosses() -> None:
    response = client.post("/sentence", json={"glosses": []})
    assert response.status_code == 422


def test_sentence_rejects_non_list_glosses() -> None:
    response = client.post("/sentence", json={"glosses": "HELLO"})
    assert response.status_code == 422
