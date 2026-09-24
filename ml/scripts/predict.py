"""Run a single-sequence prediction from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one ISL sequence.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sequence", type=Path, help="Path to a (T, F) or (1, T, F) .npy file")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--index", type=int, default=0, help="Sample index when using --split")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_sequence(path: Path):
    import numpy as np

    array = np.load(path)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected (T, F) or (N, T, F), got {array.shape}")
    return array.astype("float32")


def main() -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and NumPy are required.\n"
            "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    from app.model.model_loader import load_checkpoint
    from app.preprocessing.sequence import fit_sequence_length
    from ml.pipeline.paths import MODELS_ROOT, PROCESSED_ROOT
    from ml.pipeline.torch_data import infer_lengths, load_split_arrays

    args = parse_args()
    checkpoint = args.checkpoint or (MODELS_ROOT / "best_model.pth")
    try:
        model, payload = load_checkpoint(checkpoint, device=args.device)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    config = payload["config"]
    class_names = payload["class_names"]
    expected_t = config.get("sequence_length")
    expected_f = int(config["input_size"])

    if args.sequence:
        features = _load_sequence(args.sequence)
        if features.shape[0] != 1:
            features = features[:1]
        if features.shape[-1] != expected_f:
            raise SystemExit(
                f"Sequence feature_dim={features.shape[-1]} does not match model input_size={expected_f}."
            )
        if expected_t and features.shape[1] != expected_t:
            fitted = fit_sequence_length(features[0], int(expected_t))
            features = fitted[None, ...]
        lengths = infer_lengths(features)
    elif args.split:
        processed = args.processed_dir or PROCESSED_ROOT
        try:
            split_x, split_y, split_len, _meta = load_split_arrays(processed, args.split)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if args.index < 0 or args.index >= len(split_x):
            raise SystemExit(f"index {args.index} out of range for {args.split} (n={len(split_x)})")
        features = split_x[args.index : args.index + 1]
        lengths = split_len[args.index : args.index + 1]
        true_index = int(split_y[args.index])
    else:
        raise SystemExit("Provide --sequence path.npy  or  --split test --index 0")

    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    length_tensor = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(tensor, length_tensor)[0].cpu().numpy()
    pred_index = int(np.argmax(probs))
    result = {
        "sign": class_names[pred_index],
        "class_index": pred_index,
        "confidence": float(probs[pred_index]),
        "probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names) if i < len(probs)
        },
        "checkpoint": str(checkpoint),
        "note": "Single-sequence prediction. Confidence is a softmax score, not an evaluated accuracy.",
    }
    if args.split:
        true_name = class_names[true_index] if true_index < len(class_names) else str(true_index)
        result["true_sign"] = true_name
        result["correct"] = true_name == result["sign"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
