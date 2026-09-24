# Architecture

ISL Bridge is a local, modular prototype that translates a **limited ISL vocabulary** from a webcam into text and speech.

## Pipeline (Phase 7)

```
Frontend camera
  → MediaPipe hand landmarks (browser)
  → compact landmarks over WebSocket
  → FastAPI /ws/translate
  → preprocessing (normalize → 128-d frame)
  → SequenceBuffer (sliding SEQUENCE_LENGTH window)
  → PyTorch LSTM/GRU  OR  MockPredictor
  → confidence threshold + temporal smoothing
  → language layer (collapse repeats → gloss sequence → patterns)
  → {sign, glosses, sentence, finalized}
  → WebSocket
  → Frontend (history + Speak on click)
```

`POST /sentence` runs only the language layer: `["I","NEED","DOCTOR"]` → `"I need a doctor."`

## Current status (Phase 7 + QA)

Implemented:

- Translator UI, live browser MediaPipe overlay, no-hands and low-confidence warnings
- Dataset pipeline and shared preprocessing
- Configurable LSTM/GRU classifier, train/eval/predict scripts
- Live WebSocket `/ws/translate` and `POST /predict`
- Mock vs real model modes (no silent fallback)
- Rule-based language layer with configurable patterns and pause/repeat/explicit finalization
- Translation history of finalized sentences
- Browser SpeechSynthesis only from the Speak button
- QA: draft vs Speak split, invalid-input 422s, incomplete landmarks treated as missing

Not implemented yet:

- Broader tests and extra polish (Phase 8, optional)
- Pose / face landmarks
- A trained ISL checkpoint (do not claim accuracy)

There is no claimed recognition accuracy until `evaluate.py` runs on a real test split. This is **not** a complete ISL language translator.

## Communication decision

**Live demo:** MediaPipe in the browser. Video is not recorded and is not sent to the backend.

**Dataset collection:** separate Python script, requires `--i-am-recording`, writes `.npz` landmark takes (no video file).

**Live inference:** compact landmarks (or 128-d vectors) over WebSocket. Torch runs in a worker thread so the event loop stays free.

## MODEL_MODE

| Mode | Behavior |
| --- | --- |
| `mock` | `MockPredictor`. UI shows **Demo Model**. Application runs without a checkpoint. |
| `real` | Load `MODEL_PATH` once at startup. Missing/invalid files set `load_error` and return 503 / WS errors. Never pretends to be mock. |

## Module boundaries

| Concern | Location |
| --- | --- |
| Camera (demo) | `frontend/src/hooks/useCamera.js` |
| Landmarks (demo) | `frontend/src/services/handLandmarks.js` |
| Landmark stream | `frontend/src/hooks/useTranslateStream.js` |
| Landmarks (dataset) | `backend/app/vision/hand_detector.py` |
| Feature vector | `backend/app/preprocessing/features.py` |
| Normalization | `backend/app/preprocessing/normalize.py` |
| Sequences | `backend/app/preprocessing/sequence.py` |
| Dataset IO / splits | `ml/pipeline/` |
| Predictors | `backend/app/model/predictor.py`, `real_predictor.py` |
| Session / buffer / smoother | `backend/app/inference/` |
| Language (gloss → sentence) | `backend/app/language/` (`create_language_engine` is the NLP/LLM swap point) |
| Sentence patterns | `data/sentence_patterns.json` |
| HTTP / WebSocket | `backend/app/api/` (thin wrappers) |
