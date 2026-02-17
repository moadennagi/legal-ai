"""Load and parse configuration from YAML."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    file_path: str
    semaphore: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
