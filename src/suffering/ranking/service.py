"""High-level service API for label generation and panel dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data import DailyDataService, build_data_service
from suffering.data.models import normalize_symbol
from suffering.data.universe import resolve_symbols
from suffering.features import FeatureService, build_feature_service
from suffering.ranking.labels import build_daily_labels
from suffering.ranking.panel import build_daily_panel_dataset
from suffering.ranking.storage import RankingStorage


@dataclass(frozen=True)
class LabelBuildResult:
    frame: pd.DataFrame
    action: str
    cached_rows: int


@dataclass(frozen=True)
class DatasetBuildResult:
    frame: pd.DataFrame
    action: str
    cached_rows: int


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

    def update_label_for_symbol(
        self,
        symbol: str,
        write_cache: bool = True,
        refresh: bool = False,
    ) -> LabelBuildResult:
        normalized_symbol = normalize_symbol(symbol)
        raw_path = self.data_service.storage.path_for_symbol(normalized_symbol)
        label_path = self.storage.label_path_for_symbol(normalized_symbol)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"raw daily data not found for {normalized_symbol}. "
                f"Run `suffering data-fetch {normalized_symbol}` first."
            )

        if (
            not refresh
            and label_path.exists()
            and label_path.stat().st_mtime_ns >= raw_path.stat().st_mtime_ns
        ):
            cached_frame = self.storage.read_daily_labels(normalized_symbol)
            return LabelBuildResult(
                frame=cached_frame,
                action="cache_hit",
                cached_rows=int(len(cached_frame)),
            )

        label_frame = self.build_label_for_symbol(symbol=normalized_symbol, write_cache=write_cache)
        return LabelBuildResult(
            frame=label_frame,
            action="rebuilt",
            cached_rows=int(len(label_frame)),
        )

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

    def update_panel_dataset(
        self,
        symbols: list[str] | None = None,
        dataset_name: str | None = None,
        write_cache: bool = True,
        refresh: bool = False,
    ) -> DatasetBuildResult:
        resolved_symbols = resolve_symbols(symbols=symbols, settings=self.settings)
        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        dataset_path = self.storage.dataset_path(resolved_dataset_name)

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

        input_paths = [
            self.feature_service.storage.path_for_symbol(symbol) for symbol in resolved_symbols
        ] + [self.storage.label_path_for_symbol(symbol) for symbol in resolved_symbols]
        newest_input_mtime_ns = max(path.stat().st_mtime_ns for path in input_paths)

        if (
            not refresh
            and dataset_path.exists()
            and dataset_path.stat().st_mtime_ns >= newest_input_mtime_ns
        ):
            cached_frame = self.storage.read_daily_dataset(resolved_dataset_name)
            return DatasetBuildResult(
                frame=cached_frame,
                action="cache_hit",
                cached_rows=int(len(cached_frame)),
            )

        dataset_frame = self.build_panel_dataset(
            symbols=resolved_symbols,
            dataset_name=resolved_dataset_name,
            write_cache=write_cache,
        )
        return DatasetBuildResult(
            frame=dataset_frame,
            action="rebuilt",
            cached_rows=int(len(dataset_frame)),
        )

    def read_panel_dataset(self, dataset_name: str | None = None) -> pd.DataFrame:
        resolved_dataset_name = dataset_name or self.settings.default_dataset_name
        return self.storage.read_daily_dataset(resolved_dataset_name)


def build_ranking_service(settings: Settings | None = None) -> RankingService:
    return RankingService.from_settings(settings=settings)
