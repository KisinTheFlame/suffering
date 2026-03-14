"""High-level service API for label generation and panel dataset assembly."""

from __future__ import annotations

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data import DailyDataService, build_data_service
from suffering.data.models import normalize_symbol
from suffering.data.universe import resolve_symbols
from suffering.features import FeatureService, build_feature_service
from suffering.ranking.labels import build_daily_labels
from suffering.ranking.panel import build_daily_panel_dataset
from suffering.ranking.storage import RankingStorage


class RankingService:
    def __init__(
        self,
        storage: RankingStorage,
        data_service: DailyDataService,
        feature_service: FeatureService,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.data_service = data_service
        self.feature_service = feature_service
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RankingService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=RankingStorage.from_settings(resolved_settings),
            data_service=build_data_service(settings=resolved_settings),
            feature_service=build_feature_service(settings=resolved_settings),
            settings=resolved_settings,
        )

    def build_label_for_symbol(self, symbol: str, write_cache: bool = True) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        if not self.data_service.storage.exists(normalized_symbol):
            raise FileNotFoundError(
                f"raw daily data not found for {normalized_symbol}. "
                f"Run `suffering data-fetch {normalized_symbol}` first."
            )

        raw_frame = self.data_service.storage.read_daily_data(normalized_symbol)
        label_frame = build_daily_labels(raw_frame)
        if write_cache:
            self.storage.write_daily_labels(normalized_symbol, label_frame)
        return label_frame

    def build_labels_for_symbols(
        self,
        symbols: list[str] | None = None,
        write_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        resolved_symbols = resolve_symbols(symbols=symbols, settings=self.settings)
        return {
            symbol: self.build_label_for_symbol(symbol=symbol, write_cache=write_cache)
            for symbol in resolved_symbols
        }

    def read_labels(self, symbol: str) -> pd.DataFrame:
        return self.storage.read_daily_labels(symbol)

    def build_panel_dataset(
        self,
        symbols: list[str] | None = None,
        dataset_name: str | None = None,
        write_cache: bool = True,
    ) -> pd.DataFrame:
        resolved_symbols = resolve_symbols(symbols=symbols, settings=self.settings)
        missing_features = [
            symbol for symbol in resolved_symbols if not self.feature_service.storage.exists(symbol)
        ]
        if missing_features:
            missing_text = ", ".join(missing_features)
            raise FileNotFoundError(
                f"cached features not found for symbol(s): {missing_text}. "
                f"Run `suffering feature-build {missing_text}` first."
            )

        missing_labels = [
            symbol for symbol in resolved_symbols if not self.storage.label_exists(symbol)
        ]
        if missing_labels:
            missing_text = ", ".join(missing_labels)
            raise FileNotFoundError(
                f"cached labels not found for symbol(s): {missing_text}. "
                f"Run `suffering label-build {missing_text}` first."
            )

        feature_frames = [self.feature_service.read_features(symbol) for symbol in resolved_symbols]
        label_frames = [self.storage.read_daily_labels(symbol) for symbol in resolved_symbols]
        dataset_frame = build_daily_panel_dataset(
            feature_frames=feature_frames,
            label_frames=label_frames,
        )

        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        if write_cache:
            self.storage.write_daily_dataset(resolved_dataset_name, dataset_frame)
        return dataset_frame

    def read_panel_dataset(self, dataset_name: str | None = None) -> pd.DataFrame:
        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        return self.storage.read_daily_dataset(resolved_dataset_name)


def build_ranking_service(settings: Settings | None = None) -> RankingService:
    return RankingService.from_settings(settings=settings)
