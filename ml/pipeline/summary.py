"""Human-readable dataset summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.pipeline.paths import PROCESSED_ROOT
from ml.pipeline.splits import signer_overlap, SPLIT_NAMES


def build_summary(result: dict[str, Any], grouped_for_overlap: dict[str, list] | None = None) -> dict[str, Any]:
    overlap = set()
    if grouped_for_overlap:
        overlap = signer_overlap(grouped_for_overlap)
    split_info = result.get("split_info") or {}
    summary = {
        "sequence_length": result["sequence_length"],
        "feature_dim": result["feature_dim"],
        "num_classes": result["num_classes"],
        "label_map": result["label_map"],
        "class_names": result["class_names"],
        "encoding": (
            "y.npy stores integer class indices. label_map.json maps gloss IDs "
            "(from labels.json collect=true, file order) to those indices. "
            "Do not reorder collect labels or existing indices will change."
        ),
        "splits": result["splits"],
        "strategy": split_info.get("strategy"),
        "leakage_risk": bool(split_info.get("leakage_risk")),
        "signer_assignment": split_info.get("signer_assignment") or {},
        "warnings": list(split_info.get("warnings") or []),
        "signer_overlap_splits": sorted(overlap),
    }
    return summary


def write_summary(summary: dict[str, Any], processed_root: Path = PROCESSED_ROOT) -> Path:
    processed_root.mkdir(parents=True, exist_ok=True)
    json_path = processed_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "ISL Bridge dataset summary",
        f"sequence_length={summary['sequence_length']}  feature_dim={summary['feature_dim']}",
        f"classes={summary['num_classes']}  strategy={summary.get('strategy')}  leakage_risk={summary.get('leakage_risk')}",
        "",
        "Label encoding:",
    ]
    for name in summary.get("class_names") or []:
        lines.append(f"  {summary['label_map'][name]:2d}  {name}")
    lines.append("")
    for split_name in SPLIT_NAMES:
        stats = (summary.get("splits") or {}).get(split_name) or {}
        lines.append(
            f"{split_name}: {stats.get('samples', 0)} samples  signers={stats.get('signers', [])}"
        )
        for label, count in sorted((stats.get("per_label") or {}).items()):
            lines.append(f"    {label}: {count}")
    if summary.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for warning in summary["warnings"]:
            lines.append(f"  - {warning}")
    text_path = processed_root / "summary.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2)
