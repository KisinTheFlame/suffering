"""Local CSV storage helpers for labels and panel datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data.models import DATE_COLUMN, normalize_symbol
from suffering.ranking.labels import LABEL_OUTPUT_COLUMNS


class RankingStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.label_daily_dir = self.data_dir / "labels" / "daily"
        self.dataset_daily_dir = self.data_dir / "datasets" / "daily"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RankingStorage":
        resolved_settings = settings or get_settings()
        return cls(data_dir=resolved_settings.data_dir)

    def label_path_for_symbol(self, symbol: str) -> Path:
        return self.label_daily_dir / f"{normalize_symbol(symbol)}.csv"

    def dataset_path(self, dataset_name: str) -> Path:
        return self.dataset_daily_dir / f"{dataset_name}.csv"

    def label_exists(self, symbol: str) -> bool:
        return self.label_path_for_symbol(symbol).exists()

    def dataset_exists(self, dataset_name: str) -> bool:
        return self.dataset_path(dataset_name).exists()

    def write_daily_labels(self, symbol: str, frame: pd.DataFrame) -> Path:
        path = self.label_path_for_symbol(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = frame.loc[:, LABEL_OUTPUT_COLUMNS].copy()
        output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN]).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def read_daily_labels(self, symbol: str) -> pd.DataFrame:
        path = self.label_path_for_symbol(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"Cached labels not found for symbol: {normalize_symbol(symbol)}"
            )

        frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
        return frame.loc[:, LABEL_OUTPUT_COLUMNS]

    def write_daily_dataset(self, dataset_name: str, frame: pd.DataFrame) -> Path:
        path = self.dataset_path(dataset_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = frame.copy()
        if DATE_COLUMN in output.columns:
            output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN]).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def read_daily_dataset(self, dataset_name: str) -> pd.DataFrame:
        path = self.dataset_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {dataset_name}")

        return pd.read_csv(path, parse_dates=[DATE_COLUMN])
