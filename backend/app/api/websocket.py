"""WebSocket real-time translation: landmarks → SequenceBuffer → model."""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.inference.service import get_inference_service


async def _safe_send(websocket: WebSocket, payload: dict) -> bool:
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


def register_websocket_routes(app: FastAPI) -> None:
    @app.websocket("/ws/translate")
    async def translate(websocket: WebSocket) -> None:
        await websocket.accept()
        service = get_inference_service()
        session = service.new_session()
        hello = {
            "type": "hello",
            "mode": service.mode,
            "source": service.source if service.ready else None,
            "ready": service.ready,
            "sequence_length": service.sequence_length,
            "confidence_threshold": service.threshold,
            "smoothing_window": service.smoothing_window,
            "load_error": service.load_error,
            "sign": None,
            "confidence": 0.0,
            "accepted": False,
            "sentence": None,
            "gloss": None,
            "message": (
                None
                if service.ready
                else (service.load_error or "Model is not loaded")
            ),
        }
        hello.update(session.language.snapshot().to_dict())
        if not await _safe_send(websocket, hello):
            return

        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    ok = await _safe_send(
                        websocket,
                        {
                            "type": "error",
                            "sign": None,
                            "confidence": 0.0,
                            "accepted": False,
                            "sentence": None,
                            "gloss": None,
                            "source": service.source if service.ready else service.mode,
                            "mode": service.mode,
                            "ready": False,
                            "buffer_frames": 0,
                            "sequence_length": service.sequence_length,
                            "hands_detected": [],
                            "message": "Invalid JSON",
                        },
                    )
                    if not ok:
                        break
                    continue

                try:
                    result = await asyncio.to_thread(session.handle, data)
                except Exception as exc:
                    result = {
                        "type": "error",
                        "sign": None,
                        "confidence": 0.0,
                        "accepted": False,
                        "sentence": None,
                        "gloss": None,
                        "source": service.source if service.ready else service.mode,
                        "mode": service.mode,
                        "ready": False,
                        "buffer_frames": 0,
                        "sequence_length": service.sequence_length,
                        "hands_detected": [],
                        "message": f"Inference error: {exc}",
                    }
                if not await _safe_send(websocket, result):
                    break
        except WebSocketDisconnect:
            return
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close()
                except Exception:
                    return
