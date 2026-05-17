"""Load and parse configuration from .env / environment variables."""

import logging
from pathlib import Path
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Storage
    file_path: str = "./data/pdfs"
    semaphore: int = 10
    database_url: str = "postgresql://postgres:CHANGEME@localhost:5432/legal_ai"

    # Local LLM (Ollama)
    ollama_host: str = "http://127.0.0.1:11434"
    embeding_model: str = "bge-m3"
    generation_model: str = "qwen2.5:7b"
    reranking_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM provider routing
    llm_provider: Literal["ollama", "together", "groq", "openai", "openrouter"] = "ollama"

    # Cloud providers (OpenAI-compatible)
    openai_api_key: SecretStr = SecretStr("")
    together_api_key: SecretStr = SecretStr("")
    together_base_url: str = "https://api.together.xyz/v1"
    together_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-70b-versatile"
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "mistralai/mistral-7b-instruct"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8501"

    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")


settings = Settings()
