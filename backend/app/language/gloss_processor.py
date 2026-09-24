"""Normalize raw model labels into a stable gloss sequence.

This module does not import the ML model. It only filters, normalizes, and
collapses repeats so sliding-window predictions do not append the same sign.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.language.confidence import accept_prediction


def normalize_gloss(sign: str | None) -> str | None:
    if sign is None:
        return None
    text = str(sign).strip()
    if not text:
        return None
    return text.replace(" ", "_").replace("-", "_").upper()


def _iter_predictions(
    predictions: Sequence[Any] | None,
) -> Iterable[tuple[str | None, float | None]]:
    if not predictions:
        return
    for item in predictions:
        if item is None:
            continue
        if isinstance(item, str):
            yield item, None
            continue
        if isinstance(item, dict):
            yield item.get("sign") or item.get("gloss") or item.get("id"), item.get(
                "confidence"
            )
            continue
        yield str(item), None


def process_glosses(
    predictions: Sequence[Any] | None,
    *,
    threshold: float | None = None,
    collapse_repeats: bool = True,
) -> list[str]:
    """Return a de-duplicated gloss sequence.

    Low-confidence items are dropped when a confidence value is present.
    Consecutive identical signs (typical of a sliding window) collapse to one.
    """
    glosses: list[str] = []
    for raw_sign, confidence in _iter_predictions(predictions):
        sign = normalize_gloss(raw_sign)
        if not sign:
            continue
        if confidence is not None and not accept_prediction(confidence, threshold):
            continue
        if collapse_repeats and glosses and glosses[-1] == sign:
            continue
        glosses.append(sign)
    return glosses
