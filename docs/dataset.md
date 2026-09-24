# Dataset

Collection is **explicit and opt-in**. Inference never writes webcam data to disk.

Phase 3 stores **landmark sequences**, not video files.

## Directory layout

```
ml/config/dataset.json          # sequence length, splits, min/max frames
backend/config/labels.json     # vocabulary; collect=true marks training signs

ml/dataset/raw/
  {LABEL}/{signer_id}/take_0001.npz
  {LABEL}/{signer_id}/take_0001.json

ml/dataset/processed/
  frames/{LABEL}/{signer_id}/take_0001.npy   # (num_frames, 128)
  train/X.npy   y.npy   meta.json
  validation/X.npy   y.npy   meta.json
  test/X.npy   y.npy   meta.json
  label_map.json
  class_names.json
  summary.json
  summary.txt
```

Raw `.npz` keys: `left`, `right` (F×21×3), `left_present`, `right_present`.
No `.mp4` is written.

## How to collect samples

This is **DATASET RECORDING**, not the translator demo.

```powershell
cd isl-bridge
.\backend\.venv\Scripts\python.exe ml/scripts/collect_data.py --list-labels

.\backend\.venv\Scripts\python.exe ml/scripts/collect_data.py `
  --i-am-recording --label HELLO --signer signer_01
```

Keys: **SPACE** start/stop a take (3s countdown), **Q** quit.

Python MediaPipe is required for this script (`pip install mediapipe`). If your Python is 3.13 and the wheel is missing, use 3.11 for collection only. Extraction and splitting do not need MediaPipe.

## How to extract landmarks

```powershell
.\backend\.venv\Scripts\python.exe ml/scripts/extract_landmarks.py
```

Each take becomes a `(frames, 128)` float32 array: per-hand wrist-origin, scale-normalized xyz, plus left/right presence flags. Missing hands are zeros.

## How to create sequences

```powershell
.\backend\.venv\Scripts\python.exe ml/scripts/create_sequences.py
.\backend\.venv\Scripts\python.exe ml/scripts/summarize_dataset.py
```

Takes are resampled or zero-padded to `sequence_length` (default 30).

Splits:

- **2+ signer IDs:** each person is in exactly one of train / validation / test.
- **One signer (typical solo demo):** sample-level split and `leakage_risk: true` in the summary. Use `--strict-signer` to refuse that fallback.

## Label encoding

Training classes are `labels.json` entries with `"collect": true`, in file order:

| index | id |
| --- | --- |
| 0 | HELLO |
| 1 | THANK_YOU |
| 2 | HELP |
| 3 | YES |
| 4 | NO |
| 5 | WATER |
| 6 | DOCTOR |
| 7 | HOSPITAL |
| 8 | I |
| 9 | YOU |

`y.npy` stores those integers. `label_map.json` is the string→index map. **Append** new collect labels at the end of `labels.json`; do not reorder existing ones.

## Feature vector (reusable at inference)

`backend/app/preprocessing/` is shared with later live inference:

- `normalize.py` — wrist-relative, scale-invariant one hand
- `features.py` — 128-d frame vector
- `sequence.py` — `SequenceBuffer` sliding window + `fit_sequence_length`

Do not invent a dataset or claim accuracy. This pipeline only prepares arrays for Phase 4 training.
