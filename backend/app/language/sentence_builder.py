"""Rule-based sentence builder. Swap via LanguageEngine for an NLP/LLM later."""

from __future__ import annotations

from typing import Sequence

from app.config.labels import get_label
from app.language.gloss_processor import process_glosses
from app.language.patterns import load_sentence_patterns, pattern_key


def spoken_word(sign: str | None) -> str:
    if not sign:
        return ""
    item = get_label(sign)
    if item:
        spoken = str(item.get("spoken") or item.get("display") or "").strip()
        if spoken:
            return spoken.rstrip(".?!")
    return sign.replace("_", " ").strip().title()


def utterance_for_sign(sign: str | None) -> str | None:
    if not sign:
        return None
    patterns = load_sentence_patterns()
    if sign in patterns:
        return patterns[sign]
    word = spoken_word(sign)
    if not word:
        return None
    if word[-1] not in ".?!":
        return f"{word}."
    return word


def _fallback_chunk(glosses: Sequence[str]) -> str:
    words = [spoken_word(sign) for sign in glosses]
    words = [word for word in words if word]
    if not words:
        return ""
    text = " ".join(words)
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if text[-1] not in ".?!":
        text += "."
    return text


def build_sentence(glosses: Sequence[str]) -> str:
    """Map a gloss sequence to a sentence using configurable patterns."""
    labels = process_glosses(list(glosses))
    if not labels:
        return ""
    patterns = load_sentence_patterns()
    chunks: list[str] = []
    index = 0
    while index < len(labels):
        matched_at: int | None = None
        matched_text = ""
        for end in range(len(labels), index, -1):
            key = pattern_key(labels[index:end])
            if key in patterns:
                matched_at = end
                matched_text = patterns[key].strip()
                break
        if matched_at is not None:
            chunks.append(matched_text)
            index = matched_at
            continue
        run: list[str] = [labels[index]]
        index += 1
        while index < len(labels):
            extends = False
            for end in range(len(labels), index, -1):
                if pattern_key(labels[index:end]) in patterns:
                    extends = True
                    break
            if extends:
                break
            run.append(labels[index])
            index += 1
        chunk = _fallback_chunk(run)
        if chunk:
            chunks.append(chunk)
    return " ".join(chunks).strip()
