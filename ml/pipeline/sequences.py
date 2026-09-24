"""Extract normalized frame features and assemble fixed-length sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.preprocessing.constants import FEATURE_DIM
from app.preprocessing.sequence import fit_sequence_length
from ml.pipeline.labels import build_label_map, collect_label_ids, encode_label
from ml.pipeline.paths import FRAMES_ROOT, PROCESSED_ROOT, RAW_ROOT, load_dataset_config, resolve_labels_path
from ml.pipeline.raw_io import iter_raw_takes, load_raw_take, raw_take_to_features
from ml.pipeline.splits import SPLIT_NAMES, split_samples


def extract_all_takes(
    *,
    raw_root: Path = RAW_ROOT,
    frames_root: Path = FRAMES_ROOT,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_dataset_config()
    min_frames = int(cfg.get("min_frames", 10))
    label_ids = collect_label_ids(resolve_labels_path(cfg))
    manifest: list[dict[str, Any]] = []

    for npz_path in iter_raw_takes(raw_root):
        arrays, meta = load_raw_take(npz_path)
        label = str(meta.get("label") or npz_path.parent.parent.name)
        if label not in label_ids:
            continue
        features = raw_take_to_features(arrays)
        if features.shape[0] < min_frames:
            continue
        signer_id = str(meta.get("signer_id") or npz_path.parent.name)
        stem = str(meta.get("stem") or npz_path.stem)
        out_dir = frames_root / label / signer_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{stem}.npy"
        np.save(npy_path, features.astype(np.float32))
        record = {
            "label": label,
            "signer_id": signer_id,
            "stem": stem,
            "num_frames": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "path": str(npy_path.relative_to(frames_root)),
            "source_npz": str(npz_path),
        }
        npy_path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        manifest.append(record)

    frames_root.mkdir(parents=True, exist_ok=True)
    (frames_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_frame_samples(frames_root: Path = FRAMES_ROOT) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for npy_path in sorted(frames_root.glob("**/*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if meta_path.is_file():
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            record = {
                "label": npy_path.parent.parent.name,
                "signer_id": npy_path.parent.name,
                "stem": npy_path.stem,
                "path": str(npy_path.relative_to(frames_root)),
            }
        record["npy_path"] = npy_path
        samples.append(record)
    return samples


def create_split_arrays(
    *,
    frames_root: Path = FRAMES_ROOT,
    processed_root: Path = PROCESSED_ROOT,
    config: dict[str, Any] | None = None,
    strict_signer: bool = False,
) -> dict[str, Any]:
    cfg = config or load_dataset_config()
    sequence_length = int(cfg["sequence_length"])
    label_ids = collect_label_ids(resolve_labels_path(cfg))
    label_map = build_label_map(label_ids)
    samples = [sample for sample in _load_frame_samples(frames_root) if sample["label"] in label_map]
    grouped, split_info = split_samples(samples, cfg, strict_signer=strict_signer)

    split_stats: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        items = grouped[split_name]
        split_dir = processed_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if not items:
            empty_x = np.zeros((0, sequence_length, FEATURE_DIM), dtype=np.float32)
            empty_y = np.zeros((0,), dtype=np.int64)
            np.save(split_dir / "X.npy", empty_x)
            np.save(split_dir / "y.npy", empty_y)
            (split_dir / "meta.json").write_text("[]", encoding="utf-8")
            split_stats[split_name] = {"samples": 0, "per_label": {}, "signers": []}
            continue

        sequences = []
        labels = []
        meta_rows = []
        for item in items:
            frames = np.load(item["npy_path"])
            sequence = fit_sequence_length(frames, sequence_length)
            sequences.append(sequence)
            labels.append(encode_label(item["label"], label_map))
            meta_rows.append(
                {
                    "label": item["label"],
                    "class_index": encode_label(item["label"], label_map),
                    "signer_id": item.get("signer_id"),
                    "stem": item.get("stem"),
                    "num_frames": int(frames.shape[0]),
                }
            )
        x = np.stack(sequences, axis=0).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        np.save(split_dir / "X.npy", x)
        np.save(split_dir / "y.npy", y)
        (split_dir / "meta.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")
        per_label: dict[str, int] = {}
        for row in meta_rows:
            per_label[row["label"]] = per_label.get(row["label"], 0) + 1
        split_stats[split_name] = {
            "samples": int(len(items)),
            "shape": list(x.shape),
            "per_label": per_label,
            "signers": sorted({str(row.get("signer_id")) for row in meta_rows}),
        }

    processed_root.mkdir(parents=True, exist_ok=True)
    (processed_root / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    (processed_root / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    return {
        "sequence_length": sequence_length,
        "feature_dim": FEATURE_DIM,
        "label_map": label_map,
        "class_names": class_names,
        "splits": split_stats,
        "split_info": split_info,
        "num_classes": len(label_map),
    }
