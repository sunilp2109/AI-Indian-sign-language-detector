"""Load mock or real predictors once. API code talks to this service only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.labels import list_collect_label_ids
from app.config.settings import settings
from app.inference.session import InferenceSession
from app.model.predictor import MockPredictor
from app.preprocessing.constants import FEATURE_DIM


class InferenceService:
    def __init__(
        self,
        *,
        mode: str,
        predictor: Any | None,
        ready: bool,
        load_error: str | None = None,
        sequence_length: int | None = None,
        threshold: float | None = None,
        smoothing_window: int | None = None,
        feature_dim: int | None = None,
    ) -> None:
        self.mode = mode
        self.predictor = predictor
        self.ready = ready
        self.load_error = load_error
        self.sequence_length = int(
            settings.sequence_length if sequence_length is None else sequence_length
        )
        self.threshold = float(
            settings.confidence_threshold if threshold is None else threshold
        )
        self.smoothing_window = int(
            settings.smoothing_window if smoothing_window is None else smoothing_window
        )
        if feature_dim is not None:
            self.feature_dim = int(feature_dim)
        elif predictor is not None and getattr(predictor, "input_size", None):
            self.feature_dim = int(predictor.input_size)
        else:
            self.feature_dim = FEATURE_DIM

    @property
    def source(self) -> str:
        if self.predictor is not None:
            return getattr(self.predictor, "source", self.mode)
        return self.mode

    def status(self) -> dict[str, Any]:
        return {
            "model_mode": self.mode,
            "model_loaded": self.ready,
            "model_available": settings.model_file_exists,
            "model_source": self.source if self.ready else None,
            "load_error": self.load_error,
            "sequence_length": self.sequence_length,
            "confidence_threshold": self.threshold,
            "smoothing_window": self.smoothing_window,
        }

    def new_session(self, *, smoothing: bool = True) -> InferenceSession:
        window = self.smoothing_window if smoothing else 1
        return InferenceSession(
            self.predictor,
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim,
            threshold=self.threshold,
            smoothing_window=window,
            mode=self.mode,
            ready=self.ready,
            load_error=self.load_error,
        )

    def one_shot(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.new_session(smoothing=False)
        if payload.get("sequence") is not None:
            return session.predict_sequence(payload["sequence"], smooth=False)
        if payload.get("hands_frames"):
            last = session.handle({"type": "reset"})
            for frame in payload["hands_frames"]:
                if isinstance(frame, dict) and "type" in frame:
                    last = session.handle(frame)
                else:
                    last = session.handle({"type": "landmarks", "hands": frame})
            return last
        if payload.get("hands") is not None:
            return session.handle({"type": "landmarks", "hands": payload["hands"]})
        return session.handle({"type": "error"})

    @classmethod
    def from_settings(cls) -> InferenceService:
        mode = str(settings.model_mode or "mock").strip().lower()
        if mode == "mock":
            names = list_collect_label_ids()
            return cls(mode="mock", predictor=MockPredictor(names), ready=True)
        if mode != "real":
            return cls(
                mode=mode,
                predictor=None,
                ready=False,
                load_error=f"Unknown MODEL_MODE={mode!r}. Use 'mock' or 'real'.",
            )
        return cls.real(settings.resolved_model_path, device=settings.inference_device)

    @classmethod
    def real(cls, path: Path | str, device: str | None = None) -> InferenceService:
        try:
            from app.model.real_predictor import load_real_predictor

            predictor = load_real_predictor(
                Path(path),
                device=device or settings.inference_device,
            )
        except FileNotFoundError as exc:
            return cls(mode="real", predictor=None, ready=False, load_error=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface load failures to /health
            return cls(
                mode="real",
                predictor=None,
                ready=False,
                load_error=f"Failed to load checkpoint: {exc}",
            )
        return cls(mode="real", predictor=predictor, ready=True)


_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService.from_settings()
    return _service


def reset_inference_service(service: InferenceService | None = None) -> None:
    global _service
    _service = service
