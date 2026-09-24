# API

Base URL (local): `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

Inference runs **off the event loop** (`asyncio.to_thread`). The FastAPI layer does not call PyTorch directly; it uses `app.inference`.

## `GET /health`

Example:

```json
{
  "status": "ok",
  "service": "isl-bridge-api",
  "version": "0.1.0",
  "phase": 7,
  "websocket": "/ws/translate",
  "model_mode": "mock",
  "model_available": false,
  "model_loaded": true,
  "model_source": "mock",
  "load_error": null,
  "confidence_threshold": 0.8,
  "sequence_length": 30,
  "smoothing_window": 5,
  "language_backend": "rules",
  "pause_seconds": 1.2
}
```

`model_mode` is the configured mode. `model_loaded` is whether that mode actually has a predictor. In `MODEL_MODE=real`, a missing checkpoint sets `model_loaded=false` and `load_error` — it does **not** silently switch to mock.

## `GET /labels`

Returns the vocabulary from `backend/config/labels.json`.

## `POST /predict`

One-shot inference. Preferred body: a full temporal window of shape `(frames, 128)`, typically 30 frames:

```json
{ "sequence": [[0.0, 0.0, 0.0]] }
```

Each inner array must have **128** floats. A window with the wrong feature size returns **422**. Shorter windows are padded; longer windows are resampled to `SEQUENCE_LENGTH`.

Optional:

- `hands` — one MediaPipe-style frame (buffer will not be ready until `SEQUENCE_LENGTH` frames)
- `hands_frames` — ordered frames used to fill a `SequenceBuffer`

Response:

```json
{
  "type": "prediction",
  "sign": "HELLO",
  "confidence": 0.9,
  "accepted": true,
  "sentence": "Hello.",
  "gloss": "HELLO",
  "source": "mock",
  "mode": "mock",
  "ready": true,
  "buffer_frames": 30,
  "sequence_length": 30,
  "hands_detected": ["Right"],
  "message": null
}
```

| Field | Meaning |
| --- | --- |
| `sign` | Predicted label id, or `null` |
| `confidence` | Softmax (real) or demo score (mock) |
| `accepted` | Confidence ≥ threshold (and smoother majority on the WebSocket path) |
| `sentence` | Current draft or last finalized sentence |
| `gloss` | Current sign id |
| `glosses` | In-progress gloss sequence (repeats collapsed) |
| `finalized` | True after pause, repeat confirmation, or explicit Finalize |
| `source` | `mock` or `real` |

`MODEL_MODE=real` with no checkpoint returns **503**. Empty/missing hands return `accepted=false`, not a crash.

POST `/predict` does not apply the rolling smoother (window=1). Live WebSocket does.

## WebSocket `/ws/translate`

Browser MediaPipe sends compact landmarks. Video never leaves the device.

**Server hello** (immediately after accept):

```json
{
  "type": "hello",
  "mode": "mock",
  "ready": true,
  "sequence_length": 30,
  "confidence_threshold": 0.8,
  "smoothing_window": 5
}
```

**Client → server**

```json
{ "type": "landmarks", "hands": [{ "label": "Right", "landmarks": [{ "x": 0.1, "y": 0.2, "z": 0.0 }] }] }
```

Also accepted: `{ "type": "features", "vector": [ /* 128 floats */ ] }`, `{ "type": "sequence", "sequence": [[...]] }`, `{ "type": "reset" }`.

Invalid JSON or missing landmarks do not close the socket. A dropped client is ignored (`WebSocketDisconnect`).

**Server → client** while the buffer fills:

```json
{
  "type": "status",
  "ready": false,
  "buffer_frames": 12,
  "sequence_length": 30,
  "sign": null,
  "accepted": false,
  "message": "Filling temporal buffer (12/30)"
}
```

Once `SEQUENCE_LENGTH` frames are present, each new frame slides the window, runs the predictor, applies majority-vote smoothing, then the language layer.

Client language actions: `{ "type": "finalize" }`, `{ "type": "clear" }`, `{ "type": "reset" }`.

## `POST /sentence`

Gloss sequence → sentence. Does **not** load the ML model.

```json
{ "glosses": ["I", "NEED", "DOCTOR"] }
```

```json
{
  "glosses": ["I", "NEED", "DOCTOR"],
  "sentence": "I need a doctor.",
  "language_backend": "rules"
}
```

Patterns live in `data/sentence_patterns.json`. Consecutive duplicate glosses are collapsed. Low-confidence items are dropped when a confidence field is present.

Finalization (live path):

- **pause** — no accepted sign for `PAUSE_SECONDS` (default 1.2)
- **repeat** — same sign again after a gap, or a maximal pattern held for `REPEAT_HOLD` windows
- **explicit** — `{ "type": "finalize" }` or the Finalize button

