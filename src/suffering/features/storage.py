"""Local CSV storage helpers for daily feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data.models import DATE_COLUMN, normalize_symbol
from suffering.features.definitions import FEATURE_OUTPUT_COLUMNS


class DailyFeatureStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.feature_daily_dir = self.data_dir / "features" / "daily"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "DailyFeatureStorage":
        resolved_settings = settings or get_settings()
        return cls(data_dir=resolved_settings.data_dir)

    def path_for_symbol(self, symbol: str) -> Path:
        normalized_symbol = normalize_symbol(symbol)
        return self.feature_daily_dir / f"{normalized_symbol}.csv"

    def exists(self, symbol: str) -> bool:
        return self.path_for_symbol(symbol).exists()

    def write_daily_features(self, symbol: str, frame: pd.DataFrame) -> Path:
        path = self.path_for_symbol(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = frame.loc[:, FEATURE_OUTPUT_COLUMNS].copy()
        output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN]).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def read_daily_features(self, symbol: str) -> pd.DataFrame:
        path = self.path_for_symbol(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"Cached features not found for symbol: {normalize_symbol(symbol)}"
            )

        frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
        return frame.loc[:, FEATURE_OUTPUT_COLUMNS]
