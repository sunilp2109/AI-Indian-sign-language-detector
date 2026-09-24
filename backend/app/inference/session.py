"""Per-connection temporal buffer + smoothing + predictor.

FastAPI routes must not call the model directly; they use InferenceSession.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.language.confidence import PredictionSmoother, accept_prediction
from app.language.engine import LanguageEngine, create_language_engine
from app.preprocessing.constants import FEATURE_DIM
from app.preprocessing.sequence import SequenceBuffer, fit_sequence_length

from app.inference.landmarks import (
    features_from_landmarks,
    hands_detected_from_features,
    parse_feature_vector,
)


def _empty_prediction(
    *,
    message: str,
    mode: str,
    buffer_frames: int,
    sequence_length: int,
    ready: bool = False,
    kind: str = "status",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": kind,
        "sign": None,
        "confidence": 0.0,
        "accepted": False,
        "sentence": None,
        "gloss": None,
        "source": "mock" if mode == "mock" else "real",
        "mode": mode,
        "ready": ready,
        "buffer_frames": buffer_frames,
        "sequence_length": sequence_length,
        "hands_detected": [],
        "message": message,
        "glosses": [],
        "draft_sentence": None,
        "last_sentence": None,
        "finalized": False,
        "finalize_reason": None,
        "language_backend": "rules",
    }
    if extra:
        payload.update(extra)
    return payload


class InferenceSession:
    def __init__(
        self,
        predictor: Any | None,
        *,
        sequence_length: int,
        feature_dim: int = FEATURE_DIM,
        threshold: float = 0.80,
        smoothing_window: int = 5,
        mode: str = "mock",
        ready: bool = True,
        load_error: str | None = None,
        language: LanguageEngine | None = None,
        pause_seconds: float | None = None,
        repeat_hold: int | None = None,
    ) -> None:
        self.predictor = predictor
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.threshold = float(threshold)
        self.mode = mode
        self.service_ready = ready
        self.load_error = load_error
        self.buffer = SequenceBuffer(self.sequence_length, feature_dim=self.feature_dim)
        self.smoother = PredictionSmoother(window=smoothing_window, threshold=threshold)
        self.language = language or create_language_engine(
            threshold=threshold,
            pause_seconds=pause_seconds,
            repeat_hold=repeat_hold,
        )

    def reset(self) -> dict[str, Any]:
        self.buffer.clear()
        self.smoother.reset()
        reset = getattr(self.predictor, "reset", None)
        if callable(reset):
            reset()
        self.language.clear()
        return self._with_language(
            _empty_prediction(
                message="Buffer cleared",
                mode=self.mode,
                buffer_frames=0,
                sequence_length=self.sequence_length,
            ),
            tick=False,
        )

    def clear_language(self) -> dict[str, Any]:
        self.language.clear()
        return self._with_language(
            _empty_prediction(
                message="Gloss sequence cleared",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                ready=self.buffer.is_ready(),
                kind="language",
            ),
            tick=False,
        )

    def finalize_language(self) -> dict[str, Any]:
        self.language.finalize("explicit")
        return self._with_language(
            _empty_prediction(
                message="Sentence finalized",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                ready=self.buffer.is_ready(),
                kind="language",
            ),
            tick=False,
        )

    def _with_language(
        self,
        payload: dict[str, Any],
        *,
        sign: str | None = None,
        confidence: float = 0.0,
        accepted: bool = False,
        now: float | None = None,
        tick: bool = True,
    ) -> dict[str, Any]:
        if tick:
            state = self.language.observe(
                sign,
                confidence,
                accepted=accepted,
                now=now,
            )
        else:
            state = self.language.snapshot()
        payload.update(state.to_dict())
        return payload

    def handle(self, message: Any) -> dict[str, Any]:
        try:
            return self._dispatch(message)
        except Exception as exc:
            return _empty_prediction(
                message=f"Invalid landmark payload: {exc}",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )

    def _dispatch(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            return _empty_prediction(
                message="Invalid message: expected a JSON object",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )

        msg_type = str(message.get("type") or "landmarks").strip().lower()
        now = message.get("now")
        try:
            now = float(now) if now is not None else None
        except (TypeError, ValueError):
            now = None
        if msg_type in {"reset"}:
            return self.reset()
        if msg_type in {"clear"}:
            return self.clear_language()
        if msg_type in {"finalize", "done"}:
            return self.finalize_language()

        if not self.service_ready or self.predictor is None:
            return _empty_prediction(
                message=self.load_error
                or "Model is not loaded. Train a checkpoint or set MODEL_MODE=mock.",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
                extra={"code": "model_unavailable"},
            )

        if msg_type == "sequence":
            sequence = message.get("sequence")
            if sequence is None:
                return _empty_prediction(
                    message="Missing sequence array",
                    mode=self.mode,
                    buffer_frames=len(self.buffer),
                    sequence_length=self.sequence_length,
                    kind="error",
                )
            return self.predict_sequence(sequence, smooth=False, now=now)

        vector: np.ndarray | None = None
        if msg_type in {"features", "feature"}:
            vector = parse_feature_vector(
                message.get("vector") or message.get("features"),
                feature_dim=self.feature_dim,
            )
            if vector is None:
                return _empty_prediction(
                    message="Invalid feature vector",
                    mode=self.mode,
                    buffer_frames=len(self.buffer),
                    sequence_length=self.sequence_length,
                    kind="error",
                )
        elif msg_type in {"landmarks", "frame", "hands"}:
            vector = features_from_landmarks(message)
        else:
            return _empty_prediction(
                message=f"Unknown message type: {msg_type}",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )

        return self.push_features(vector, now=now)

    def push_features(self, vector: np.ndarray, *, now: float | None = None) -> dict[str, Any]:
        if not self.service_ready or self.predictor is None:
            return _empty_prediction(
                message=self.load_error or "Model is not loaded",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
                extra={"code": "model_unavailable"},
            )
        try:
            self.buffer.append(vector)
        except ValueError as exc:
            return _empty_prediction(
                message=str(exc),
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )

        detected = hands_detected_from_features(np.asarray(vector, dtype=np.float32))
        if not self.buffer.is_ready():
            return self._with_language(
                _empty_prediction(
                    message=(
                        f"Filling temporal buffer ({len(self.buffer)}/{self.sequence_length})"
                    ),
                    mode=self.mode,
                    buffer_frames=len(self.buffer),
                    sequence_length=self.sequence_length,
                    extra={"hands_detected": detected},
                ),
                sign=None,
                confidence=0.0,
                accepted=False,
                now=now,
            )
        return self._predict_window(detected, now=now)

    def predict_sequence(
        self,
        frames: Any,
        *,
        smooth: bool = False,
        now: float | None = None,
    ) -> dict[str, Any]:
        if not self.service_ready or self.predictor is None:
            return _empty_prediction(
                message=self.load_error or "Model is not loaded",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
                extra={"code": "model_unavailable"},
            )
        try:
            array = np.asarray(frames, dtype=np.float32)
        except (TypeError, ValueError):
            return _empty_prediction(
                message="sequence must be a numeric (frames, features) array",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )
        if array.ndim != 2:
            return _empty_prediction(
                message=f"sequence must be 2-D, got shape {array.shape}",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )
        if array.shape[1] != self.feature_dim:
            return _empty_prediction(
                message=f"Expected feature_dim={self.feature_dim}, got {array.shape[1]}",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
            )
        window = fit_sequence_length(array, self.sequence_length)
        return self._emit(
            window,
            detected=hands_detected_from_features(window[-1]),
            smooth=smooth,
            now=now,
        )

    def _predict_window(self, detected: list[str], now: float | None = None) -> dict[str, Any]:
        window = self.buffer.get_sequence()
        return self._emit(window, detected=detected, smooth=True, now=now)

    def _emit(
        self,
        window: np.ndarray,
        *,
        detected: list[str],
        smooth: bool,
        now: float | None = None,
    ) -> dict[str, Any]:
        try:
            raw = self.predictor.predict_sequence(window)
        except Exception as exc:
            return _empty_prediction(
                message=f"Inference error: {exc}",
                mode=self.mode,
                buffer_frames=len(self.buffer),
                sequence_length=self.sequence_length,
                kind="error",
                extra={"hands_detected": detected},
            )
        sign = raw.get("sign")
        confidence = float(raw.get("confidence") or 0.0)
        if smooth:
            smoothed = self.smoother.push(sign, confidence)
            sign = smoothed["sign"]
            confidence = float(smoothed["confidence"])
            accepted = bool(smoothed["accepted"])
        else:
            accepted = accept_prediction(confidence, self.threshold)

        if not detected:
            sign = None
            confidence = 0.0
            accepted = False

        payload = {
            "type": "prediction",
            "sign": sign,
            "confidence": round(confidence, 4),
            "accepted": accepted,
            "gloss": sign,
            "source": raw.get("source") or ("mock" if self.mode == "mock" else "real"),
            "mode": self.mode,
            "ready": True,
            "buffer_frames": len(self.buffer) if len(self.buffer) else int(window.shape[0]),
            "sequence_length": self.sequence_length,
            "hands_detected": detected,
            "message": None if accepted else (
                "Uncertain — please sign again." if sign else "No sign recognized."
            ),
        }
        return self._with_language(
            payload,
            sign=sign if accepted else None,
            confidence=confidence,
            accepted=accepted,
            now=now,
        )
