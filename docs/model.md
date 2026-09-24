# Model

The classifier is a bidirectional LSTM over 30×128 landmark windows. **Webcam ISL accuracy is not claimed** as a public benchmark. Local `evaluate.py --split test` numbers apply only to the split you evaluate (currently a signer-held-out **kinematic template** set for 10 simple words).

## Architecture

```
Input: (batch, SEQUENCE_LENGTH, FEATURE_DIM)
  → LSTM or GRU (configurable, default bidirectional LSTM)
  → last real hidden state (packed lengths, so end-padding is ignored)
  → Dropout
  → Linear
  → logits

Inference applies softmax to logits.
Training uses weighted CrossEntropyLoss with optional label smoothing.
```

Defaults (`ml/config/model.json`): bidirectional LSTM, hidden_size=128, num_layers=2, dropout=0.35, label_smoothing=0.05, cosine LR, grad clip 1.0.

**Configurable (do not hardcode in the model body):**

| Value | Source |
| --- | --- |
| sequence_length | `X.npy` shape[1] / `dataset.json` |
| input_size (feature dim) | `X.npy` shape[2] (128 from Phase 3) |
| num_classes | `label_map.json` |
| rnn_type, hidden_size, layers, dropout, bidirectional | `ml/config/model.json` |

## Simple-sign training set

`ml/scripts/build_simple_sign_dataset.py` writes signer-split windows for:

HELLO, THANK_YOU, HELP, YES, NO, WATER, DOCTOR, HOSPITAL, I, YOU

These are MediaPipe-shaped motion templates with noise, **not** a filmed ISL corpus. Features still go through `hands_to_features` (wrist-normalize + 128-d layout) so they match live inference.

On a held-out synthetic-signer test split, a local run reported about **99% accuracy / 0.99 macro-F1**, with HOSPITAL occasionally confused with NO (both use an extended index finger). That number is **not** webcam ISL accuracy.

## Expected dataset structure

Produced by `create_sequences.py` or `build_simple_sign_dataset.py`:

```
ml/dataset/processed/
  train/X.npy          (N, T, F)
  train/y.npy          (N,) integer class index
  train/meta.json      includes num_frames for packing
  validation/...
  test/...
  label_map.json
  class_names.json
```

Empty splits: `train.py` / `evaluate.py` exit with an error instead of inventing samples.

## How to train

```powershell
# CPU wheel is enough for the laptop MVP
.\backend\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.\backend\.venv\Scripts\python.exe ml/scripts/build_simple_sign_dataset.py
.\backend\.venv\Scripts\python.exe ml/scripts/train.py
```

Writes:

- `ml/models/best_model.pth` — best **validation macro-F1** (early stopping patience 8)
- `ml/models/training_history.json`
- `ml/models/training_summary.json`
- copy at `backend/models/best_model.pth`

If validation is empty, pass `--allow-no-val` (the run cannot pick a best-val checkpoint).

Training metrics are **not** test accuracy.

## How to evaluate

```powershell
.\backend\.venv\Scripts\python.exe ml/scripts/evaluate.py --split test
```

Reports accuracy, precision, recall, F1 (macro and weighted), and a confusion matrix. Saved under `ml/models/eval_test.json` and `confusion_matrix_test.txt`.

Quote only these held-out numbers, and only after a real test split exists.

Single sequence:

```powershell
.\backend\.venv\Scripts\python.exe ml/scripts/predict.py --split test --index 0
.\backend\.venv\Scripts\python.exe ml/scripts/predict.py --sequence path\to\window.npy
```

Softmax confidence is not an evaluation score.

## Modes

| `MODEL_MODE` | Meaning |
| --- | --- |
| `mock` | `MockPredictor` (Phase 5). UI shows **Demo Model**. No checkpoint required. |
| `real` | Load `MODEL_PATH` at startup. UI shows **Trained Model** only if the file loads. Missing files error; they do not fall back to mock. |

Live path: landmarks → `/ws/translate` → `SequenceBuffer` → predictor → threshold + smoother → `{sign, confidence, accepted, sentence, gloss}`.

Torch inference runs in `asyncio.to_thread`. Predictors live in `backend/app/model/` and `backend/app/inference/`, not in the route handlers.
