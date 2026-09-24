# ISL Bridge

Real-time Indian Sign Language translator prototype.

ISL Bridge uses a laptop or phone webcam to recognize a **focused, configurable ISL vocabulary** and turn accepted signs into text and speech. It is **not** a complete ISL language translator.

Current milestone: **Phase 7 + simple-sign training**. Live camera, landmarks, 10-word trained model (kinematic templates), sentences, history, Speak-on-click. Webcam ISL accuracy is not claimed as a public benchmark.

## 1. Project overview

The MVP recognizes a **focused, configurable vocabulary** (currently 22 labels in `backend/config/labels.json`; 10 are flagged for collection/training). It is **not** a complete ISL language translator.

Target user flow:

1. Grant camera access.
2. Sign within the supported vocabulary.
3. See the current sign, confidence, and sentence.
4. Speak a **finalized** sentence on demand.

## 2. Architecture

```
Camera → frames → landmarks → WebSocket → FastAPI
      → normalize → SequenceBuffer → LSTM/GRU (or mock)
      → confidence gate → gloss/sentence → frontend
```

Frontend: React + Vite  
Backend: FastAPI  
ML: PyTorch LSTM/GRU  
Speech: browser `SpeechSynthesis` (Speak button only)

See [docs/architecture.md](docs/architecture.md).

**Realtime decision:** MediaPipe hand landmarks run in the browser. Video is not sent to the backend and is not recorded. Compact landmarks go over WebSocket `/ws/translate`.

## 3. Features (Phase 7)

Working:

- Translator page with live camera, MediaPipe overlay, and no-hands warning
- Current sign, confidence percent + bar, low-confidence warning
- Gloss sequences, configurable sentence patterns, pause/repeat/explicit finalization
- Translation history of **finalized** sentences
- Speak uses browser SpeechSynthesis only when you click Speak
- Camera, connection, and model status
- Configurable vocabulary and local train/eval pipeline

Not a full ISL translator. Optional Phase 8 is extra polish and broader tests.

## 4. Tech stack

- React 18, Vite 6, JavaScript
- FastAPI, Uvicorn, Pydantic Settings
- OpenCV, MediaPipe Tasks Vision (browser) + optional Python MediaPipe for collection
- PyTorch, scikit-learn, NumPy, Pandas, JSON

No Docker, Redis, auth, or cloud services in the MVP.

## 5. Installation

Requirements: Python 3.11+ (3.13 works for Phase 1), Node.js 20+.

```powershell
cd "C:\Users\Sunil2109\Desktop\ai based indian sign language detector\isl-bridge"
```

## 6. Backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Tests:

```powershell
cd ..
.\backend\.venv\Scripts\python.exe -m pytest tests -q
```

## 7. Frontend setup

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) (or the next free port Vite prints, often 5174).

Keep the backend running so status badges and vocabulary can load.

Frontend tests:

```powershell
npm test
```

## 8. Dataset preparation

See [docs/dataset.md](docs/dataset.md) and `ml/dataset/README.md`.

Labels with `"collect": true` in `backend/config/labels.json` are the Phase 3 training set (10 signs). Sequence length and splits live in `ml/config/dataset.json`.

```powershell
# 1) Record landmarks only (not video). Requires Python mediapipe + a webcam.
.\backend\.venv\Scripts\python.exe ml/scripts/collect_data.py --list-labels
.\backend\.venv\Scripts\python.exe ml/scripts/collect_data.py --i-am-recording --label HELLO --signer signer_01

# 2) Normalize takes → (frames, 128) arrays
.\backend\.venv\Scripts\python.exe ml/scripts/extract_landmarks.py

# 3) Fixed-length windows + train/validation/test
.\backend\.venv\Scripts\python.exe ml/scripts/create_sequences.py
.\backend\.venv\Scripts\python.exe ml/scripts/summarize_dataset.py
```

Inference never writes to `ml/dataset/`. There is no public ISL corpus in this repo.

A **simple-sign kinematic set** can be built for the 10 `collect: true` words:

```powershell
.\backend\.venv\Scripts\python.exe ml/scripts/build_simple_sign_dataset.py
```

That set is signer-split synthetic MediaPipe-like motion, **not** filmed ISL. Quote `evaluate.py --split test` numbers only as local template-set metrics.

## 9. Model training

See [docs/model.md](docs/model.md).

Install PyTorch (CPU is enough):

```powershell
.\backend\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\backend\.venv\Scripts\python.exe ml/scripts/build_simple_sign_dataset.py
.\backend\.venv\Scripts\python.exe ml/scripts/train.py
.\backend\.venv\Scripts\python.exe ml/scripts/evaluate.py --split test
.\backend\.venv\Scripts\python.exe ml/scripts/predict.py --split test --index 0
```

`train.py` refuses to run without processed arrays. Training history is not test accuracy. Use `evaluate.py` on the held-out test split.

The live app uses `MODEL_MODE=real` only after `backend/models/best_model.pth` exists. That model recognizes the 10 simple trained words when you match the on-screen motion hints. It is **not** a full ISL recognizer.

## 10. Running the complete demo

1. Start FastAPI from `backend/` on port 8000.
2. Start Vite from `frontend/` (`npm run dev`).
3. Open the printed local URL (often `http://localhost:5174`).
4. Click **Start camera** and allow permission.
5. Sign within the supported list. Predictions start after 30 frames.
6. Pause, hold a complete phrase, or press **Finalize**.
7. Press **Speak** only when you want the **finalized** sentence read aloud.

Keep `MODEL_MODE=real` in `backend/.env` when `backend/models/best_model.pth` exists. Use `mock` if the checkpoint is missing.

```powershell
# After train.py has written a checkpoint
MODEL_MODE=real
MODEL_PATH=models/best_model.pth

# Demo without a trained checkpoint
# MODEL_MODE=mock
```

`MODEL_MODE=real` with a missing file does **not** fall back to mock. `/health` reports `load_error`, and `/predict` returns 503.

After `evaluate.py` has been run on **test**, you may quote those local numbers. Until then, do not claim accuracy. Mock mode is labeled **Demo Model** in the UI.

## 11. API documentation

See [docs/api.md](docs/api.md) and `/docs` on the running server.

| Method | Path | Phase 7 |
| --- | --- | --- |
| GET | `/health` | Working |
| GET | `/labels` | Working |
| POST | `/predict` | Working |
| POST | `/sentence` | Working |
| WS | `/ws/translate` | Working |

## 12. Troubleshooting

**Backend will not start.** Run uvicorn from the `backend/` folder so `app.main:app` is importable.

**Frontend shows “Backend unavailable.”** Start FastAPI on port 8000. Check `VITE_API_URL` in `frontend/.env`.

**Camera does nothing / permission denied.** Click **Start camera** and allow the browser prompt. Use `http://localhost` (not a file:// URL). If another app is using the webcam, close it. Click **Stop camera** to release the device.

**No skeleton on the hands.** Wait until the HUD shows an FPS number (MediaPipe model download can take a few seconds). Show a hand clearly, with decent light. Left is teal, right is saffron, labeled at the wrist.

**`/mediapipe/wasm` 404.** From `frontend/`, run `node scripts/copy-mediapipe-wasm.mjs` (also runs on `npm install`).

**Labels missing.** Confirm `backend/config/labels.json` exists and `LABELS_PATH` in `.env` points to it.

**Python `mediapipe` install fails.** Browser MediaPipe is enough for the live demo. Collection (`collect_data.py`) needs the Python package; use Python 3.11 if 3.13 has no wheel. Extraction and splitting do not need MediaPipe.

**`train.py` says the split is empty.** Collect landmark takes, then run extract + create_sequences. Do not invent samples.

**PyTorch missing.** `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

**WebSocket disconnected / no predictions.** Confirm FastAPI is on port 8000 and `VITE_WEBSOCKET_URL` points at `ws://127.0.0.1:8000/ws/translate`. Keep the camera live; the stream reconnects automatically.

**`MODEL_MODE=real` says the model is missing.** Train first (`ml/scripts/train.py`) so `backend/models/best_model.pth` exists, or switch `MODEL_MODE=mock`.

## 13. Future improvements

- Pose and face landmarks
- Broader tests and polish (Phase 8)
- Optional later NLP/LLM sentence layer (swap `create_language_engine`)
