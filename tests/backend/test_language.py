"""Phase 6 language layer. Independent of the ML model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(ROOT))

from app.language.engine import (  # noqa: E402
    RuleBasedLanguageEngine,
    create_language_engine,
    sentence_from_glosses,
)
from app.language.gloss_processor import process_glosses
from app.language.sentence_builder import build_sentence
from app.main import app  # noqa: E402
from app.inference.service import reset_inference_service  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def restore_service() -> None:
    yield
    reset_inference_service(None)


def test_collapse_sliding_window_repeats() -> None:
    assert process_glosses(["I", "I", "I", "NEED", "NEED", "DOCTOR", "DOCTOR"]) == [
        "I",
        "NEED",
        "DOCTOR",
    ]


def test_confidence_filtering() -> None:
    glosses = process_glosses(
        [
            {"sign": "I", "confidence": 0.95},
            {"sign": "NEED", "confidence": 0.40},
            {"sign": "DOCTOR", "confidence": 0.91},
        ],
        threshold=0.8,
    )
    assert glosses == ["I", "DOCTOR"]


def test_required_sentence_patterns() -> None:
    cases = {
        ("I", "NEED", "DOCTOR"): "I need a doctor.",
        ("I", "NEED", "WATER"): "I need water.",
        ("CALL", "DOCTOR"): "Please call a doctor.",
        ("WHERE", "HOSPITAL"): "Where is the hospital?",
        ("THANK_YOU",): "Thank you.",
        ("GOOD_MORNING",): "Good morning.",
        ("I", "NEED", "HELP"): "I need help.",
    }
    for glosses, expected in cases.items():
        assert build_sentence(glosses) == expected
        payload = sentence_from_glosses(list(glosses))
        assert payload["sentence"] == expected
        assert payload["language_backend"] == "rules"


def test_repeated_predictions_normalize_before_build() -> None:
    assert build_sentence(["I", "I", "NEED", "NEED", "DOCTOR"]) == "I need a doctor."


def test_pause_finalizes_sentence() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=1.0, repeat_hold=99)
    engine.observe("I", 0.9, now=0.0)
    engine.observe("NEED", 0.9, now=0.2)
    engine.observe("DOCTOR", 0.92, now=0.4)
    mid = engine.snapshot()
    assert mid.glosses == ["I", "NEED", "DOCTOR"]
    assert mid.finalized is False
    done = engine.observe(None, 0.0, now=1.5)
    assert done.finalized is True
    assert done.finalize_reason == "pause"
    assert done.sentence == "I need a doctor."
    assert done.glosses == []


def test_repeat_after_gap_finalizes() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=5.0, repeat_hold=99)
    engine.observe("THANK_YOU", 0.9, now=0.0)
    engine.observe(None, 0.1, now=0.2)
    done = engine.observe("THANK_YOU", 0.9, now=0.3)
    assert done.finalized is True
    assert done.finalize_reason == "repeat"
    assert done.sentence == "Thank you."


def test_sliding_window_does_not_append_same_sign() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=5.0, repeat_hold=99)
    for index in range(8):
        state = engine.observe("HELLO", 0.9, now=index * 0.05)
    assert state.glosses == ["HELLO"]
    assert state.finalized is False


def test_explicit_finalize_and_clear() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=5.0, repeat_hold=99)
    engine.observe("I", 0.9, now=0.0)
    engine.observe("NEED", 0.9, now=0.1)
    engine.observe("HELP", 0.9, now=0.2)
    done = engine.finalize("explicit")
    assert done.finalized is True
    assert done.sentence == "I need help."
    cleared = engine.clear()
    assert cleared.glosses == []
    assert cleared.sentence is None
    assert cleared.finalized is False


def test_low_confidence_is_ignored() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=5.0, repeat_hold=99)
    state = engine.observe("DOCTOR", 0.2, now=0.0)
    assert state.glosses == []
    assert state.sentence is None


def test_hold_on_maximal_pattern_counts_as_repeat() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=5.0, repeat_hold=3)
    engine.observe("GOOD_MORNING", 0.9, now=0.0)
    engine.observe("GOOD_MORNING", 0.9, now=0.05)
    done = engine.observe("GOOD_MORNING", 0.9, now=0.1)
    assert done.finalized is True
    assert done.finalize_reason == "repeat"
    assert done.sentence == "Good morning."


def test_create_language_engine_is_the_swap_point() -> None:
    engine = create_language_engine("rules", threshold=0.8)
    assert engine.name == "rules"
    try:
        create_language_engine("llm")
        raise AssertionError("LLM backend should not be silently invented")
    except ValueError as exc:
        assert "LANGUAGE_BACKEND" in str(exc)


def test_sentence_endpoint_does_not_need_a_model() -> None:
    response = client.post("/sentence", json={"glosses": ["I", "NEED", "DOCTOR"]})
    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "I need a doctor."
    assert body["glosses"] == ["I", "NEED", "DOCTOR"]
    assert body["language_backend"] == "rules"


def test_websocket_finalize_and_clear() -> None:
    from app.inference.service import InferenceService, reset_inference_service
    from app.model.predictor import MockPredictor

    reset_inference_service(
        InferenceService(
            mode="mock",
            predictor=MockPredictor(["HELLO"], hold=100),
            ready=True,
            sequence_length=3,
            threshold=0.8,
            smoothing_window=1,
        )
    )
    try:
        frame = {
            "type": "landmarks",
            "hands": [{"label": "Right", "landmarks": [{"x": 0.1 * i, "y": 0.2, "z": 0.0} for i in range(21)]}],
        }
        with client.websocket_connect("/ws/translate") as socket:
            socket.receive_json()
            last = None
            for tick in range(3):
                payload = dict(frame)
                payload["now"] = tick * 0.05
                socket.send_json(payload)
                last = socket.receive_json()
            assert last["glosses"] == ["HELLO"]
            socket.send_json({"type": "finalize"})
            finalized = socket.receive_json()
            assert finalized["finalized"] is True
            assert finalized["sentence"] == "Hello."
            socket.send_json({"type": "clear"})
            cleared = socket.receive_json()
            assert cleared["glosses"] == []
            assert cleared["sentence"] is None
    finally:
        reset_inference_service(None)


def test_long_draft_finalizes_before_runaway() -> None:
    engine = RuleBasedLanguageEngine(threshold=0.8, pause_seconds=99.0, repeat_hold=99)
    labels = ["HELLO", "YES", "NO", "WATER", "HELP", "I", "YOU", "DOCTOR"]
    last = None
    for index, label in enumerate(labels):
        last = engine.observe(label, 0.95, now=float(index))
    assert last is not None
    assert last.finalized is True
    assert last.finalize_reason == "repeat"
    assert last.last_glosses == labels
