"""Centralized runtime configuration.

Values come from environment variables (or backend/.env). Do not hardcode
model paths, thresholds, or sequence length in feature modules.
Reload this module after changing MODEL_MODE.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_mode: str = "mock"
    model_path: str = "models/best_model.pth"
    confidence_threshold: float = 0.80
    sequence_length: int = 30
    smoothing_window: int = 5
    inference_device: str = "cpu"
    labels_path: str = "config/labels.json"
    language_backend: str = "rules"
    sentence_patterns_path: str = ""
    pause_seconds: float = 1.2
    repeat_hold: int = 12
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174,"
        "http://localhost:5175,http://127.0.0.1:5175"
    )

    @property
    def backend_root(self) -> Path:
        return BACKEND_ROOT

    @property
    def resolved_labels_path(self) -> Path:
        path = Path(self.labels_path)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def resolved_sentence_patterns_path(self) -> Path:
        if self.sentence_patterns_path:
            path = Path(self.sentence_patterns_path)
            if not path.is_absolute():
                path = BACKEND_ROOT / path
            return path
        return BACKEND_ROOT.parent / "data" / "sentence_patterns.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_file_exists(self) -> bool:
        return self.resolved_model_path.is_file()


settings = Settings()
