"""Load and parse configuration from YAML."""

import logging
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    file_path: str
    openai_api_key: SecretStr
    semaphore: int = 10
    database_url: str = "postgresql://postgres:mysecretpassword@0.0.0.0:5432/legal_ai"
    embeding_model: str = "bge-m3"
    generation_model: str = "qwen2.5:7b"
    reranking_model: str = "BAAI/bge-reranker-v2-m3"
    ollama_host: str = "http://127.0.0.1:11434"
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")


settings = Settings()
