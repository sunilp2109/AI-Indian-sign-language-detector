"""Language layer: glosses → sentences. Independent of the ML model."""

from app.language.confidence import PredictionSmoother, accept_prediction
from app.language.engine import (
    LanguageEngine,
    LanguageState,
    RuleBasedLanguageEngine,
    create_language_engine,
    sentence_from_glosses,
)
from app.language.gloss_processor import process_glosses
from app.language.sentence_builder import build_sentence, utterance_for_sign

__all__ = [
    "LanguageEngine",
    "LanguageState",
    "PredictionSmoother",
    "RuleBasedLanguageEngine",
    "accept_prediction",
    "build_sentence",
    "create_language_engine",
    "process_glosses",
    "sentence_from_glosses",
    "utterance_for_sign",
]
