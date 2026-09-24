"""Dataset pipeline tests. Uses synthetic landmarks, not a claimed ISL corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preprocessing.constants import FEATURE_DIM, LANDMARKS_PER_HAND
from ml.pipeline.labels import build_label_map, collect_label_ids, encode_label
from ml.pipeline.paths import load_dataset_config, resolve_labels_path
from ml.pipeline.raw_io import save_raw_take
from ml.pipeline.sequences import create_split_arrays, extract_all_takes
from ml.pipeline.splits import signer_overlap, split_samples
from ml.pipeline.summary import build_summary, write_summary


def _hand(origin: float) -> np.ndarray:
    points = np.zeros((LANDMARKS_PER_HAND, 3), dtype=np.float32)
    for index in range(LANDMARKS_PER_HAND):
        points[index] = [origin, 0.5 - 0.01 * index, 0.0]
    return points


def _take_frames(num_frames: int, *, left: bool, right: bool) -> list[dict]:
    frames = []
    for _ in range(num_frames):
        frames.append(
            {
                "left": _hand(0.3) if left else None,
                "right": _hand(0.7) if right else None,
            }
        )
    return frames


def test_label_encoding_uses_config_order() -> None:
    config = load_dataset_config()
    ids = collect_label_ids(resolve_labels_path(config))
    mapping = build_label_map(ids)
    assert encode_label("HELLO", mapping) == 0
    assert encode_label("YOU", mapping) == 9
    with pytest.raises(KeyError):
        encode_label("NOT_A_SIGN", mapping)


def test_signer_split_has_no_overlap() -> None:
    samples = []
    for signer in ("a", "b", "c", "d"):
        for label in ("HELLO", "HELP"):
            samples.append({"label": label, "signer_id": signer, "stem": f"{signer}_{label}"})
    grouped, info = split_samples(samples, {"seed": 1, "split": {"train": 0.5, "validation": 0.25, "test": 0.25}})
    assert info["strategy"] == "signer"
    assert info["leakage_risk"] is False
    assert signer_overlap(grouped) == set()
    assigned = {sample["signer_id"] for items in grouped.values() for sample in items}
    assert assigned == {"a", "b", "c", "d"}


def test_single_signer_sets_leakage_warning() -> None:
    samples = [{"label": "HELLO", "signer_id": "only", "stem": f"t{i}"} for i in range(12)]
    grouped, info = split_samples(samples, {"seed": 0, "split": {"train": 0.7, "validation": 0.15, "test": 0.15}})
    assert info["leakage_risk"] is True
    assert sum(len(v) for v in grouped.values()) == 12
    with pytest.raises(ValueError):
        split_samples(samples, {"seed": 0}, strict_signer=True)


def test_extract_and_create_sequences(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    frames_root = tmp_path / "frames"
    processed_root = tmp_path / "processed"
    signers = ("signer_a", "signer_b", "signer_c")
    labels = ("HELLO", "WATER", "DOCTOR")
    for signer in signers:
        for label in labels:
            save_raw_take(
                label=label,
                signer_id=signer,
                frames=_take_frames(18, left=True, right=label != "I"),
                raw_root=raw_root,
            )

    manifest = extract_all_takes(raw_root=raw_root, frames_root=frames_root)
    assert len(manifest) == 9
    sample = np.load(frames_root / manifest[0]["path"])
    assert sample.shape[1] == FEATURE_DIM
    assert sample.dtype == np.float32

    result = create_split_arrays(
        frames_root=frames_root,
        processed_root=processed_root,
    )
    summary = build_summary(result)
    write_summary(summary, processed_root=processed_root)
    assert (processed_root / "summary.json").is_file()
    assert (processed_root / "label_map.json").is_file()
    train_x = np.load(processed_root / "train" / "X.npy")
    assert train_x.ndim == 3
    assert train_x.shape[1] == load_dataset_config()["sequence_length"]
    assert train_x.shape[2] == FEATURE_DIM
    assert summary["strategy"] == "signer"
    assert summary["leakage_risk"] is False
    assert (processed_root / "summary.txt").is_file()


def test_collect_script_requires_opt_in() -> None:
    script = ROOT / "ml" / "scripts" / "collect_data.py"
    refused = subprocess.run(
        [sys.executable, str(script), "--label", "HELLO"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "DATASET RECORDING" in (refused.stderr + refused.stdout)

    listed = subprocess.run(
        [sys.executable, str(script), "--list-labels"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0
    assert "HELLO" in listed.stdout
    assert "GOOD_MORNING" not in listed.stdout
