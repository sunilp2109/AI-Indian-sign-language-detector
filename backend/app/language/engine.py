"""Language engines: gloss sequence → sentence.

RuleBasedLanguageEngine is the MVP. create_language_engine() is the swap point
for a future NLP/LLM backend. This package must not import the ML model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Protocol, Sequence

from app.config.settings import settings
from app.language.confidence import accept_prediction
from app.language.gloss_processor import normalize_gloss, process_glosses
from app.language.patterns import is_maximal_pattern, load_sentence_patterns
from app.language.sentence_builder import build_sentence


@dataclass
class LanguageState:
    glosses: list[str] = field(default_factory=list)
    last_glosses: list[str] = field(default_factory=list)
    draft_sentence: str = ""
    sentence: str | None = None
    last_sentence: str | None = None
    finalized: bool = False
    finalize_reason: str | None = None
    language_backend: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "glosses": list(self.glosses),
            "last_glosses": list(self.last_glosses),
            "draft_sentence": self.draft_sentence or None,
            "sentence": self.sentence,
            "last_sentence": self.last_sentence,
            "finalized": self.finalized,
            "finalize_reason": self.finalize_reason,
            "language_backend": self.language_backend,
        }


class LanguageEngine(Protocol):
    """Replaceable language layer (rules today, NLP/LLM later)."""

    name: str

    def observe(
        self,
        sign: str | None,
        confidence: float,
        *,
        accepted: bool | None = None,
        hands_detected: Sequence[str] | None = None,
        now: float | None = None,
    ) -> LanguageState: ...

    def finalize(self, reason: str = "explicit") -> LanguageState: ...

    def clear(self) -> LanguageState: ...

    def snapshot(self) -> LanguageState: ...


class RuleBasedLanguageEngine:
    """Collapse sliding-window repeats, then finalize on pause / repeat / explicit."""

    name = "rules"

    def __init__(
        self,
        *,
        threshold: float | None = None,
        pause_seconds: float | None = None,
        repeat_hold: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.threshold = (
            settings.confidence_threshold if threshold is None else float(threshold)
        )
        self.pause_seconds = (
            settings.pause_seconds if pause_seconds is None else float(pause_seconds)
        )
        self.repeat_hold = (
            settings.repeat_hold if repeat_hold is None else int(repeat_hold)
        )
        self._clock = clock or monotonic
        self._glosses: list[str] = []
        self._last_committed: str | None = None
        self._last_accepted_at: float | None = None
        self._saw_gap = False
        self._hold_count = 0
        self._finalized = False
        self._finalize_reason: str | None = None
        self._last_sentence: str | None = None
        self._last_finalized_glosses: list[str] = []

    def snapshot(self) -> LanguageState:
        draft = build_sentence(self._glosses) if self._glosses else ""
        display = draft or self._last_sentence
        return LanguageState(
            glosses=list(self._glosses),
            last_glosses=list(self._last_finalized_glosses),
            draft_sentence=draft,
            sentence=display,
            last_sentence=self._last_sentence,
            finalized=self._finalized,
            finalize_reason=self._finalize_reason,
            language_backend=self.name,
        )

    def clear(self) -> LanguageState:
        self._glosses = []
        self._last_committed = None
        self._last_accepted_at = None
        self._saw_gap = False
        self._hold_count = 0
        self._finalized = False
        self._finalize_reason = None
        self._last_sentence = None
        self._last_finalized_glosses = []
        return self.snapshot()

    def finalize(self, reason: str = "explicit") -> LanguageState:
        if not self._glosses:
            return self.snapshot()
        text = build_sentence(self._glosses)
        self._last_sentence = text or None
        self._last_finalized_glosses = list(self._glosses)
        self._glosses = []
        self._last_committed = None
        self._saw_gap = False
        self._hold_count = 0
        self._finalized = True
        self._finalize_reason = reason
        return self.snapshot()

    def observe(
        self,
        sign: str | None,
        confidence: float,
        *,
        accepted: bool | None = None,
        hands_detected: Sequence[str] | None = None,
        now: float | None = None,
    ) -> LanguageState:
        del hands_detected  # reserved for pose/face pauses later
        timestamp = self._clock() if now is None else float(now)
        gloss = normalize_gloss(sign)
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        is_accepted = (
            bool(accepted)
            if accepted is not None
            else bool(gloss) and accept_prediction(score, self.threshold)
        )

        if not is_accepted or not gloss:
            if self._glosses:
                self._saw_gap = True
                if (
                    self._last_accepted_at is not None
                    and (timestamp - self._last_accepted_at) >= self.pause_seconds
                ):
                    return self.finalize("pause")
            return self.snapshot()

        self._last_accepted_at = timestamp
        self._finalized = False
        self._finalize_reason = None

        if not self._glosses:
            self._glosses = [gloss]
            self._last_committed = gloss
            self._saw_gap = False
            self._hold_count = 1
            return self.snapshot()

        if gloss == self._last_committed:
            self._hold_count += 1
            if self._saw_gap:
                return self.finalize("repeat")
            if (
                self.repeat_hold > 0
                and self._hold_count >= self.repeat_hold
                and is_maximal_pattern(self._glosses)
            ):
                return self.finalize("repeat")
            return self.snapshot()

        self._saw_gap = False
        self._hold_count = 1
        self._last_committed = gloss
        self._glosses.append(gloss)
        if len(self._glosses) >= 8:
            return self.finalize("repeat")
        return self.snapshot()


def create_language_engine(
    backend: str | None = None,
    **kwargs: Any,
) -> LanguageEngine:
    name = (backend or settings.language_backend or "rules").strip().lower()
    if name in {"rules", "rule", "rule_based"}:
        return RuleBasedLanguageEngine(**kwargs)
    raise ValueError(
        f"Unknown LANGUAGE_BACKEND={name!r}. "
        "The MVP ships a rule-based engine; plug in an NLP/LLM backend here later."
    )


def sentence_from_glosses(glosses: Sequence[Any], **kwargs: Any) -> dict[str, Any]:
    """Stateless helper for POST /sentence and tests. Does not use live pause/repeat."""
    del kwargs
    labels = process_glosses(glosses)
    return {
        "glosses": labels,
        "sentence": build_sentence(labels) or None,
        "language_backend": "rules",
        "pattern_count": len(load_sentence_patterns()),
    }
