"""HTTP routes for ISL Bridge."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException

from app.api.schemas import PredictRequest, SentenceRequest
from app.config.labels import load_label_catalog
from app.config.settings import settings
from app.inference.service import get_inference_service
from app.language.engine import sentence_from_glosses

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness plus inference-mode status for the local MVP."""
    service = get_inference_service()
    status = service.status()
    return {
        "status": "ok",
        "service": "isl-bridge-api",
        "version": "0.1.0",
        "phase": 7,
        "websocket": "/ws/translate",
        "model_mode": status["model_mode"],
        "model_available": status["model_available"],
        "model_loaded": status["model_loaded"],
        "model_source": status["model_source"],
        "load_error": status["load_error"],
        "confidence_threshold": status["confidence_threshold"],
        "sequence_length": status["sequence_length"],
        "smoothing_window": status["smoothing_window"],
        "language_backend": settings.language_backend,
        "pause_seconds": settings.pause_seconds,
    }


@router.get("/labels")
def get_labels() -> dict:
    """Return the configurable ISL vocabulary."""
    try:
        catalog = load_label_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="Unable to load labels.json") from exc

    return {
        "version": catalog.get("version"),
        "language": catalog.get("language", "ISL"),
        "count": len(catalog["labels"]),
        "labels": catalog["labels"],
    }


@router.post("/predict")
async def predict(body: PredictRequest) -> dict:
    """Run one-shot inference on a sequence or landmark frames."""
    if body.sequence is None and body.hands is None and body.hands_frames is None:
        raise HTTPException(
            status_code=422,
            detail="Provide sequence, hands, or hands_frames.",
        )

    service = get_inference_service()
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=service.load_error
            or "Model is not loaded. Train a checkpoint or set MODEL_MODE=mock.",
        )

    payload = body.model_dump()
    result = await asyncio.to_thread(service.one_shot, payload)
    if result.get("code") == "model_unavailable":
        raise HTTPException(status_code=503, detail=result.get("message"))
    if result.get("type") == "error" and result.get("sign") is None and not result.get("ready"):
        message = result.get("message") or "Inference failed"
        if "Unknown message" in message or "must be" in message or "Invalid" in message or "Expected" in message:
            raise HTTPException(status_code=422, detail=message)
    return result


@router.post("/sentence")
def sentence(body: SentenceRequest) -> dict:
    """Rule-based gloss sequence → sentence. Independent of the ML model."""
    if not body.glosses:
        raise HTTPException(status_code=422, detail="Provide a glosses array.")
    return sentence_from_glosses(body.glosses)
