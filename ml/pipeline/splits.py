"""Train / validation / test assignment with optional signer-aware grouping."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


SPLIT_NAMES = ("train", "validation", "test")


def _ratios(config: dict[str, Any]) -> tuple[float, float, float]:
    split = config.get("split") or {}
    train = float(split.get("train", 0.70))
    validation = float(split.get("validation", 0.15))
    test = float(split.get("test", 0.15))
    total = train + validation + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number")
    return train / total, validation / total, test / total


def _assign_signers(signers: list[str], config: dict[str, Any]) -> dict[str, str]:
    train_r, val_r, test_r = _ratios(config)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    ordered = np.array(sorted(signers), dtype=object)
    rng.shuffle(ordered)
    n = len(ordered)
    assignment: dict[str, str] = {}

    if n == 1:
        assignment[str(ordered[0])] = "train"
        return assignment
    if n == 2:
        assignment[str(ordered[0])] = "train"
        assignment[str(ordered[1])] = "test"
        return assignment

    n_test = max(1, int(round(n * test_r)))
    n_val = max(1, int(round(n * val_r)))
    if n_test + n_val >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = 1
        n_val = max(1, n - n_train - n_test)

    chunks = [
        ("train", ordered[:n_train]),
        ("validation", ordered[n_train : n_train + n_val]),
        ("test", ordered[n_train + n_val :]),
    ]
    for split_name, group in chunks:
        for signer in group:
            assignment[str(signer)] = split_name
    return assignment


def split_samples(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    strict_signer: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return {split: samples} and split metadata.

    If multiple real signer IDs exist, each signer is in exactly one split.
    If only one signer (typical solo hackathon collection), falls back to a
    sample-level split and sets leakage_risk=true unless strict_signer=True.
    """
    info: dict[str, Any] = {
        "strategy": "signer",
        "leakage_risk": False,
        "warnings": [],
        "signer_assignment": {},
    }
    if not samples:
        return {name: [] for name in SPLIT_NAMES}, info

    real_signers = sorted(
        {
            str(sample.get("signer_id") or "")
            for sample in samples
            if str(sample.get("signer_id") or "").strip()
            and not str(sample.get("signer_id")).startswith("__")
        }
    )

    use_signer = len(real_signers) >= 2
    if not use_signer:
        message = (
            "Only one signer (or missing signer IDs). Using sample-level split. "
            "Validation/test may leak signer identity."
        )
        if strict_signer:
            raise ValueError(
                message + " Refusing because --strict-signer is set. Collect from more people."
            )
        info["strategy"] = "sample"
        info["leakage_risk"] = True
        info["warnings"].append(message)
        return _sample_split(samples, config, info), info

    assignment = _assign_signers(real_signers, config)
    info["signer_assignment"] = assignment
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        signer = str(sample.get("signer_id") or "")
        split_name = assignment.get(signer)
        if split_name is None:
            info["warnings"].append(f"Unassigned signer '{signer}' sent to train")
            split_name = "train"
        grouped[split_name].append(sample)
    return grouped, info


def _sample_split(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    train_r, val_r, test_r = _ratios(config)
    seed = int(config.get("seed", 42))
    labels = [sample["label"] for sample in samples]
    indices = np.arange(len(samples))
    stratify = labels if min(labels.count(label) for label in set(labels)) >= 2 else None
    if stratify is None:
        info["warnings"].append("Not enough samples per class to stratify the sample-level split.")

    try:
        rest, test_idx = train_test_split(
            indices,
            test_size=test_r,
            random_state=seed,
            stratify=stratify,
        )
        rest_labels = [labels[i] for i in rest]
        rest_stratify = (
            rest_labels if min(rest_labels.count(label) for label in set(rest_labels)) >= 2 else None
        )
        val_ratio_of_rest = val_r / (train_r + val_r) if train_r + val_r > 0 else 0.5
        train_idx, val_idx = train_test_split(
            rest,
            test_size=val_ratio_of_rest,
            random_state=seed,
            stratify=rest_stratify,
        )
    except ValueError as exc:
        info["warnings"].append(f"Falling back to an unstratified shuffle: {exc}")
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(round(n * test_r))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_r))) if n >= 3 else 0
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test : n_test + n_val]
        train_idx = shuffled[n_test + n_val :]

    grouped = {name: [] for name in SPLIT_NAMES}
    for index in train_idx:
        grouped["train"].append(samples[int(index)])
    for index in val_idx:
        grouped["validation"].append(samples[int(index)])
    for index in test_idx:
        grouped["test"].append(samples[int(index)])
    return grouped


def signer_overlap(grouped: dict[str, list[dict[str, Any]]]) -> set[tuple[str, str]]:
    """Pairs of splits that share a real signer id. Empty means no leakage."""
    per_split: dict[str, set[str]] = {}
    for name, items in grouped.items():
        per_split[name] = {
            str(sample.get("signer_id"))
            for sample in items
            if str(sample.get("signer_id") or "").strip()
            and not str(sample.get("signer_id")).startswith("__")
        }
    overlaps: set[tuple[str, str]] = set()
    names = list(per_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = per_split[left] & per_split[right]
            if shared:
                overlaps.add((left, right))
    return overlaps
