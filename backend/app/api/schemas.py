"""Request/response models for inference endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: list[list[float]] | None = Field(
        default=None,
        description="One temporal window as (frames, features). Preferred for POST /predict.",
    )
    hands: list[dict[str, Any]] | None = Field(
        default=None,
        description="Single MediaPipe-style frame. Buffer will not be full until SEQUENCE_LENGTH frames.",
    )
    hands_frames: list[Any] | None = Field(
        default=None,
        description="Ordered landmark frames used to fill a SequenceBuffer.",
    )


class SentenceRequest(BaseModel):
    glosses: list[Any] = Field(
        default_factory=list,
        description='Recognized gloss ids, e.g. ["I", "NEED", "DOCTOR"].',
    )


class PredictResponse(BaseModel):
    type: str = "prediction"
    sign: str | None = None
    confidence: float = 0.0
    accepted: bool = False
    sentence: str | None = None
    gloss: str | None = None
    glosses: list[str] = Field(default_factory=list)
    draft_sentence: str | None = None
    last_sentence: str | None = None
    finalized: bool = False
    finalize_reason: str | None = None
    language_backend: str | None = None
    source: str = "mock"
    mode: str = "mock"
    ready: bool = False
    buffer_frames: int = 0
    sequence_length: int = 30
    hands_detected: list[str] = Field(default_factory=list)
    message: str | None = None
    code: str | None = None
