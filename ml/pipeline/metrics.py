"""Classification metrics. Values are only meaningful on a real labelled split."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)
    labels = list(range(n_classes))
    if y_true.size == 0:
        return {
            "n_samples": 0,
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "f1_weighted": None,
            "per_class": {},
            "confusion_matrix": [],
            "note": "No samples. This is not an accuracy result.",
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "note": "Metrics computed on the provided labelled split only. Not a published benchmark.",
    }


def format_confusion_matrix(matrix: list[list[int]], class_names: list[str]) -> str:
    names = [name[:10] for name in class_names]
    width = max(10, *(len(name) for name in names), 4)
    header = " ".join(name.rjust(width) for name in ["pred->"] + names)
    lines = [header]
    for true_name, row in zip(names, matrix, strict=False):
        cells = " ".join(str(int(value)).rjust(width) for value in row)
        lines.append(f"{true_name.rjust(width)} {cells}")
    return "\n".join(lines) + "\n"
