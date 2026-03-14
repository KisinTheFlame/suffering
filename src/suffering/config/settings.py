"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    default_data_provider: str = "yfinance"
    default_start_date: str = "2020-01-01"
    default_end_date: str | None = None
    default_symbols: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    default_label_horizon_days: int = 5
    default_dataset_name: str = "panel_5d"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    @field_validator("default_symbols", mode="before")
    @classmethod
    def parse_default_symbols(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
