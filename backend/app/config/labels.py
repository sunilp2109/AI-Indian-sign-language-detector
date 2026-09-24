"""Load sign labels from the configured JSON file. Do not hardcode vocabulary."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.config.settings import settings


@lru_cache(maxsize=1)
def load_label_catalog() -> dict[str, Any]:
    path = settings.resolved_labels_path
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if "labels" not in catalog or not isinstance(catalog["labels"], list):
        raise ValueError("labels.json must contain a 'labels' array")
    return catalog


def list_label_ids() -> list[str]:
    return [item["id"] for item in load_label_catalog()["labels"]]


def list_collect_label_ids() -> list[str]:
    """Labels marked collect=true, or the full catalog if none are flagged."""
    items = load_label_catalog()["labels"]
    flagged = [item["id"] for item in items if item.get("collect")]
    return flagged or list_label_ids()


def build_label_map(label_ids: list[str] | None = None) -> dict[str, int]:
    ids = label_ids if label_ids is not None else list_collect_label_ids()
    return {label_id: index for index, label_id in enumerate(ids)}


def get_label(sign_id: str) -> dict[str, Any] | None:
    for item in load_label_catalog()["labels"]:
        if item.get("id") == sign_id:
            return item
    return None
