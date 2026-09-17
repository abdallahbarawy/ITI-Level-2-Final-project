import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    ollama_timeout: float = Field(default=180, gt=0)
    retrieval_k: int = Field(default=3, ge=1, le=20)
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    @field_validator("vector_store_path")
    @classmethod
    def resolve_store_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    def load_store_config(self) -> dict:
        path = self.vector_store_path / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        required = {"embedding_model", "collection_name", "embedding_dim", "ollama_model"}
        if not required.issubset(config):
            raise ValueError("Vector store config is missing required export settings")
        return config


@lru_cache
def get_settings() -> Settings:
    return Settings()
