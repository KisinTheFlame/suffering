"""Local CSV storage helpers for raw daily market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data.models import DAILY_PRICE_COLUMNS, DATE_COLUMN, normalize_symbol


class DailyDataStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.raw_daily_dir = self.data_dir / "raw" / "daily"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "DailyDataStorage":
        resolved_settings = settings or get_settings()
        return cls(data_dir=resolved_settings.data_dir)

    def path_for_symbol(self, symbol: str) -> Path:
        normalized_symbol = normalize_symbol(symbol)
        return self.raw_daily_dir / f"{normalized_symbol}.csv"

    def exists(self, symbol: str) -> bool:
        return self.path_for_symbol(symbol).exists()

    def write_daily_data(self, symbol: str, frame: pd.DataFrame) -> Path:
        path = self.path_for_symbol(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = frame.loc[:, DAILY_PRICE_COLUMNS].copy()
        output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN], utc=True).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def read_daily_data(self, symbol: str) -> pd.DataFrame:
        path = self.path_for_symbol(symbol)
        if not path.exists():
            raise FileNotFoundError(f"Cached data not found for symbol: {normalize_symbol(symbol)}")

        frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
        return frame.loc[:, DAILY_PRICE_COLUMNS]
