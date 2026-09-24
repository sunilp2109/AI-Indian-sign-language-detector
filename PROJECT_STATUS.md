# Project status

Last updated: 2026-09-21  
Current milestone: **Phase 7 + simple-sign training**

ISL Bridge is a **limited-vocabulary** Indian Sign Language prototype. It is **not** a full ISL translator.

## DONE

### Phase 1–7
- As previously implemented (camera, MediaPipe, dataset pipeline, LSTM, live WS, language layer, translator UI, QA)

### Simple-word training
- Kinematic MediaPipe-like dataset for HELLO, THANK_YOU, HELP, YES, NO, WATER, DOCTOR, HOSPITAL, I, YOU
- Signer-held-out splits (6 train / 2 val / 2 test synthetic signers)
- Bidirectional LSTM, class-weighted loss, label smoothing, cosine LR, train-time noise
- Local test-split result: **accuracy ≈ 0.992, macro-F1 ≈ 0.992** on those templates (HOSPITAL vs NO is the remaining confusion)
- Live app set to `MODEL_MODE=real` when `backend/models/best_model.pth` exists
- UI lists how to perform the 10 trained motions

Those test metrics are **not** a public ISL webcam benchmark.

## IN PROGRESS

- None.

## TODO

- Collect real multi-signer webcam takes and retrain before quoting live accuracy
- Add `NEED` / `CALL` to the collect subset if those phrases should come from the camera
- Pose / face landmarks (`PoseDetector` remains an intentional stub)

## KNOWN LIMITATIONS

- **Not a full ISL translator.** 22 runtime labels; the trained model covers the 10 `collect: true` simple words
- Training data is **synthetic kinematic templates**, not a filmed ISL corpus. Live recognition works when you match the on-screen motion hints
- Wrist-normalized features drop absolute hand position; signs must differ in finger pose or wrist-relative motion
- `NEED` / `CALL` are language-pattern labels only; the real model cannot emit them
- Sentence output is rule-based, not ISL grammar
- Pose and face landmarks are not enabled
- `MODEL_MODE=real` with a missing checkpoint errors instead of falling back to mock
- Speech uses browser `SpeechSynthesis`; Speak is never automatic
