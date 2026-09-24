"""Evaluate a trained checkpoint on a held-out split.

Reports accuracy, precision, recall, F1, and a confusion matrix.
Does not invent numbers if the split or checkpoint is missing.
"""

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
    parser = argparse.ArgumentParser(description="Evaluate ISL sequence classifier.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required.\n"
            "Install a CPU wheel:\n"
            "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    from app.model.model_loader import load_checkpoint
    from ml.pipeline.engine import evaluate_model, select_device
    from ml.pipeline.paths import MODELS_ROOT, PROCESSED_ROOT
    from ml.pipeline.torch_data import load_label_artifacts, load_split_arrays, make_loader

    args = parse_args()
    processed = args.processed_dir or PROCESSED_ROOT
    output_dir = args.output_dir or MODELS_ROOT
    checkpoint = args.checkpoint or (output_dir / "best_model.pth")

    try:
        features, labels, lengths, _meta = load_split_arrays(processed, args.split)
        label_map, class_names = load_label_artifacts(processed)
        model, payload = load_checkpoint(checkpoint, device="cpu")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    ckpt_classes = payload["config"]["num_classes"]
    if ckpt_classes != len(class_names):
        raise SystemExit(
            f"Checkpoint num_classes={ckpt_classes} does not match "
            f"label_map size {len(class_names)}."
        )

    device = select_device(args.device)
    model.to(device)
    loader = make_loader(features, labels, lengths, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_model(model, loader, class_names=class_names, device=device)
    report = {
        "split": args.split,
        "checkpoint": str(checkpoint),
        "n_samples": metrics["n_samples"],
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": class_names,
        "label_map": label_map,
        "note": metrics["note"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"eval_{args.split}.json"
    txt_path = output_dir / f"confusion_matrix_{args.split}.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(metrics["confusion_matrix_text"], encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "per_class"}, indent=2))
    print("\nConfusion matrix:\n")
    print(metrics["confusion_matrix_text"])
    print(f"Wrote {json_path}")
    print("These numbers are only for this local split. They are not a published ISL benchmark.")


if __name__ == "__main__":
    main()
