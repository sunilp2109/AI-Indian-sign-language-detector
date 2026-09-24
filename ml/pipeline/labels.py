"""Load configurable sign labels for the dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_label_catalog(labels_path: Path) -> dict[str, Any]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with labels_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if "labels" not in catalog or not isinstance(catalog["labels"], list):
        raise ValueError("labels.json must contain a 'labels' array")
    return catalog


def collect_label_ids(labels_path: Path) -> list[str]:
    items = load_label_catalog(labels_path)["labels"]
    ids = [item["id"] for item in items]
    flagged = [item["id"] for item in items if item.get("collect")]
    return flagged or ids


def build_label_map(label_ids: list[str]) -> dict[str, int]:
    if not label_ids:
        raise ValueError("No labels configured")
    duplicates = {item for item in label_ids if label_ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"Duplicate label ids: {sorted(duplicates)}")
    return {label_id: index for index, label_id in enumerate(label_ids)}


def class_names(label_map: dict[str, int]) -> list[str]:
    return [name for name, _index in sorted(label_map.items(), key=lambda item: item[1])]


def encode_label(label_id: str, label_map: dict[str, int]) -> int:
    if label_id not in label_map:
        raise KeyError(f"Unknown label '{label_id}'. Add it to labels.json (collect=true).")
    return label_map[label_id]
