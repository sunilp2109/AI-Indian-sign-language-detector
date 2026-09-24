Dataset folders for ISL Bridge.

raw/
  Landmark takes from collect_data.py only. No video files.
  Layout: {LABEL}/{signer_id}/take_XXXX.npz + take_XXXX.json

processed/
  Normalized frame arrays and train/validation/test sequences
  produced by extract_landmarks.py, create_sequences.py, or
  `ml/scripts/build_simple_sign_dataset.py` (kinematic templates for the
  10 simple collect-true words — not a filmed ISL corpus).

Collection is opt-in (`--i-am-recording`). Inference never writes here.

Splits are signer-aware when multiple signer IDs exist so the same person
does not appear in more than one split.

See docs/dataset.md. Do not commit captured takes or claim accuracy.
