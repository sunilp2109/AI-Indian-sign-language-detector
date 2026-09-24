"""Build a signer-split dataset for the 10 simple collect-true signs.

Uses kinematic MediaPipe-style templates plus noise — not a public ISL corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize simple-sign training windows.")
    parser.add_argument("--signers", type=int, default=10)
    parser.add_argument("--takes", type=int, default=12, help="Takes per sign per signer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    from app.preprocessing.constants import FEATURE_DIM
    from ml.pipeline.paths import PROCESSED_ROOT, load_dataset_config
    from ml.pipeline.simple_signs import build_simple_sign_samples, label_map_for_simple_signs
    from ml.pipeline.splits import SPLIT_NAMES, split_samples
    from ml.pipeline.summary import build_summary, write_summary

    args = parse_args()
    processed = args.processed_dir or PROCESSED_ROOT
    cfg = load_dataset_config()
    sequence_length = int(cfg["sequence_length"])
    samples = build_simple_sign_samples(
        signers=args.signers,
        takes_per_sign=args.takes,
        sequence_length=sequence_length,
        seed=args.seed,
    )
    grouped, split_info = split_samples(samples, cfg)
    label_map = label_map_for_simple_signs()
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    split_stats: dict[str, object] = {}
    for split_name in SPLIT_NAMES:
        items = grouped[split_name]
        split_dir = processed / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if not items:
            np.save(split_dir / "X.npy", np.zeros((0, sequence_length, FEATURE_DIM), dtype=np.float32))
            np.save(split_dir / "y.npy", np.zeros((0,), dtype=np.int64))
            (split_dir / "meta.json").write_text("[]", encoding="utf-8")
            split_stats[split_name] = {"samples": 0, "per_label": {}, "signers": []}
            continue
        x = np.stack([item["features"] for item in items], axis=0).astype(np.float32)
        y = np.asarray([label_map[item["label"]] for item in items], dtype=np.int64)
        meta = [
            {
                "label": item["label"],
                "class_index": int(label_map[item["label"]]),
                "signer_id": item["signer_id"],
                "stem": item["stem"],
                "num_frames": int(item["num_frames"]),
                "source": "simple_sign_kinematics",
            }
            for item in items
        ]
        np.save(split_dir / "X.npy", x)
        np.save(split_dir / "y.npy", y)
        (split_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        per_label: dict[str, int] = {}
        for row in meta:
            per_label[row["label"]] = per_label.get(row["label"], 0) + 1
        split_stats[split_name] = {
            "samples": int(len(items)),
            "shape": list(x.shape),
            "per_label": per_label,
            "signers": sorted({str(row["signer_id"]) for row in meta}),
        }

    processed.mkdir(parents=True, exist_ok=True)
    (processed / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (processed / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    source = {
        "kind": "synthetic_kinematic_templates",
        "description": (
            "Signer-split MediaPipe-like trajectories for HELLO, THANK_YOU, HELP, "
            "YES, NO, WATER, DOCTOR, HOSPITAL, I, YOU. Not a public ISL corpus. "
            "Live accuracy depends on performing the documented simple motions."
        ),
        "signers": args.signers,
        "takes_per_sign": args.takes,
        "seed": args.seed,
    }
    (processed / "data_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    result = {
        "sequence_length": sequence_length,
        "feature_dim": FEATURE_DIM,
        "label_map": label_map,
        "class_names": class_names,
        "splits": split_stats,
        "split_info": split_info,
        "num_classes": len(label_map),
    }
    summary = build_summary(result, grouped)
    summary["data_source"] = source
    write_summary(summary, processed)
    print(json.dumps({"processed": str(processed), "splits": split_stats, "source": source["kind"]}, indent=2))


if __name__ == "__main__":
    main()
