"""Load configurable gloss-to-sentence patterns. Independent of the ML model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config.settings import BACKEND_ROOT, settings

DEFAULT_PATTERNS_PATH = BACKEND_ROOT.parent / "data" / "sentence_patterns.json"


def patterns_path() -> Path:
    configured = getattr(settings, "resolved_sentence_patterns_path", None)
    if configured is not None:
        return Path(configured)
    return DEFAULT_PATTERNS_PATH


@lru_cache(maxsize=4)
def load_pattern_catalog(path: str | None = None) -> dict[str, Any]:
    file_path = Path(path) if path else patterns_path()
    if not file_path.is_file():
        return {"version": 1, "backend": "rules", "fallback": "join_with_spaces", "patterns": {}}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "backend": "rules", "fallback": "join_with_spaces", "patterns": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "backend": "rules", "fallback": "join_with_spaces", "patterns": {}}
    patterns = payload.get("patterns")
    if not isinstance(patterns, dict):
        payload = dict(payload)
        payload["patterns"] = {}
    return payload


def load_sentence_patterns(path: str | None = None) -> dict[str, str]:
    catalog = load_pattern_catalog(path)
    patterns = catalog.get("patterns") or {}
    return {str(key): str(value) for key, value in patterns.items()}


def pattern_key(glosses: list[str]) -> str:
    return "|".join(glosses)


def is_exact_pattern(glosses: list[str], patterns: dict[str, str] | None = None) -> bool:
    table = patterns if patterns is not None else load_sentence_patterns()
    return bool(glosses) and pattern_key(glosses) in table


def has_longer_pattern(glosses: list[str], patterns: dict[str, str] | None = None) -> bool:
    if not glosses:
        return False
    table = patterns if patterns is not None else load_sentence_patterns()
    prefix = pattern_key(glosses) + "|"
    return any(key.startswith(prefix) for key in table)


def is_maximal_pattern(glosses: list[str], patterns: dict[str, str] | None = None) -> bool:
    table = patterns if patterns is not None else load_sentence_patterns()
    return is_exact_pattern(glosses, table) and not has_longer_pattern(glosses, table)


def clear_pattern_cache() -> None:
    load_pattern_catalog.cache_clear()
