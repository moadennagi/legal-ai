"""Load and parse configuration from YAML."""

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    file_path: str
    semaphore: int = 10
    database_url: str = "postgresql://postgres:mysecretpassword@0.0.0.0:5432/legal_ai"
    embeding_model: str = "bge-m3"
    generation_model: str = "qwen2.5:7b"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_host: str = "http://172.21.112.1:11434"
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
