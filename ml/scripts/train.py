"""Train the LSTM/GRU ISL classifier on Phase 3 processed arrays.

Saves best_model.pth by validation macro-F1 when a validation split exists.
Does not claim test accuracy — run evaluate.py on the held-out test split.
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
    parser = argparse.ArgumentParser(description="Train ISL sequence classifier.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument(
        "--allow-no-val",
        action="store_true",
        help="Train even if the validation split is empty (cannot select a best-val checkpoint)",
    )
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

    from app.model.isl_model import ISLSequenceModel
    from ml.pipeline.engine import select_device, set_seed, train_model
    from ml.pipeline.paths import MODELS_ROOT, PROCESSED_ROOT, load_model_config
    from ml.pipeline.torch_data import load_label_artifacts, load_split_arrays, make_loader
    import numpy as np

    args = parse_args()
    processed = args.processed_dir or PROCESSED_ROOT
    output_dir = args.output_dir or MODELS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    hyper = load_model_config()

    try:
        features, labels, lengths, _meta = load_split_arrays(processed, "train")
        label_map, class_names = load_label_artifacts(processed)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    val_loader = None
    try:
        val_x, val_y, val_len, _ = load_split_arrays(processed, "validation")
        val_loader = make_loader(
            val_x,
            val_y,
            val_len,
            batch_size=int(args.batch_size or hyper.get("batch_size", 16)),
            shuffle=False,
        )
    except (OSError, ValueError) as exc:
        if not args.allow_no_val:
            raise SystemExit(
                f"Validation split unavailable ({exc}). "
                "Collect more signers/samples, or pass --allow-no-val."
            ) from exc
        print(f"Warning: {exc}  Training without validation.")

    input_size = int(features.shape[2])
    sequence_length = int(features.shape[1])
    num_classes = len(class_names)
    if num_classes < 2:
        raise SystemExit("Need at least 2 classes in label_map.json")
    if int(labels.max()) >= num_classes or int(labels.min()) < 0:
        raise SystemExit("Train labels are outside label_map indices.")

    set_seed(int(hyper.get("seed", 42)))
    device = select_device(args.device)
    model = ISLSequenceModel(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=int(hyper.get("hidden_size", 128)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
        rnn_type=str(hyper.get("rnn_type", "lstm")),
        bidirectional=bool(hyper.get("bidirectional", False)),
        sequence_length=sequence_length,
    ).to(device)

    batch_size = int(args.batch_size or hyper.get("batch_size", 16))
    train_loader = make_loader(
        features, labels, lengths, batch_size=batch_size, shuffle=True, augment=True
    )

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    print(
        f"Device={device}  N_train={len(features)}  "
        f"shape=({sequence_length}, {input_size})  classes={num_classes}  "
        f"rnn={hyper.get('rnn_type', 'lstm')}"
    )
    result = train_model(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        label_map=label_map,
        epochs=int(args.epochs or hyper.get("epochs", 40)),
        learning_rate=float(hyper.get("learning_rate", 0.001)),
        weight_decay=float(hyper.get("weight_decay", 0.0001)),
        patience=int(hyper.get("early_stopping_patience", 8)),
        device=device,
        checkpoint_path=output_dir / "best_model.pth",
        history_path=output_dir / "training_history.json",
        label_smoothing=float(hyper.get("label_smoothing", 0.0)),
        grad_clip=float(hyper.get("grad_clip", 0.0)),
        class_weights=weights,
        use_cosine=bool(hyper.get("use_cosine", False)),
    )
    summary_path = output_dir / "training_summary.json"
    slim = {key: value for key, value in result.items() if key != "history"}
    summary_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    backend_copy = ROOT / "backend" / "models" / "best_model.pth"
    if Path(result["checkpoint"]).resolve() != backend_copy.resolve():
        import shutil

        backend_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result["checkpoint"], backend_copy)
        slim["backend_copy"] = str(backend_copy)
        summary_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(json.dumps(slim, indent=2))
    print("Training finished. Do not quote accuracy until evaluate.py runs on the test split.")


if __name__ == "__main__":
    main()
