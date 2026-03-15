"""High-level service API for daily feature access."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data import DailyDataService, build_data_service
from suffering.data.models import normalize_symbol
from suffering.data.universe import resolve_symbols
from suffering.features.storage import DailyFeatureStorage
from suffering.features.transforms import build_daily_features


@dataclass(frozen=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    action: str
    cached_rows: int


class FeatureService:
    def __init__(
        self,
        storage: DailyFeatureStorage,
        data_service: DailyDataService,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.data_service = data_service
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "FeatureService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=DailyFeatureStorage.from_settings(resolved_settings),
            data_service=build_data_service(settings=resolved_settings),
            settings=resolved_settings,
        )

    def build_features_for_symbol(self, symbol: str, write_cache: bool = True) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        if not self.data_service.storage.exists(normalized_symbol):
            raise FileNotFoundError(
                f"raw daily data not found for {normalized_symbol}. "
                f"Run `suffering data-fetch {normalized_symbol}` first."
            )

        raw_frame = self.data_service.storage.read_daily_data(normalized_symbol)
        feature_frame = build_daily_features(raw_frame)
        if write_cache:
            self.storage.write_daily_features(normalized_symbol, feature_frame)
        return feature_frame

    def update_features_for_symbol(
        self,
        symbol: str,
        write_cache: bool = True,
        refresh: bool = False,
    ) -> FeatureBuildResult:
        normalized_symbol = normalize_symbol(symbol)
        raw_path = self.data_service.storage.path_for_symbol(normalized_symbol)
        feature_path = self.storage.path_for_symbol(normalized_symbol)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"raw daily data not found for {normalized_symbol}. "
                f"Run `suffering data-fetch {normalized_symbol}` first."
            )

        if (
            not refresh
            and feature_path.exists()
            and feature_path.stat().st_mtime_ns >= raw_path.stat().st_mtime_ns
        ):
            cached_frame = self.storage.read_daily_features(normalized_symbol)
            return FeatureBuildResult(
                frame=cached_frame,
                action="cache_hit",
                cached_rows=int(len(cached_frame)),
            )

        feature_frame = self.build_features_for_symbol(
            symbol=normalized_symbol,
            write_cache=write_cache,
        )
        return FeatureBuildResult(
            frame=feature_frame,
            action="rebuilt",
            cached_rows=int(len(feature_frame)),
        )

    def build_features_for_symbols(
        self,
        symbols: list[str] | None = None,
        write_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        resolved_symbols = resolve_symbols(symbols=symbols, settings=self.settings)
        return {
            symbol: self.build_features_for_symbol(symbol=symbol, write_cache=write_cache)
            for symbol in resolved_symbols
        }

    def read_features(self, symbol: str) -> pd.DataFrame:
        return self.storage.read_daily_features(symbol)


def build_feature_service(settings: Settings | None = None) -> FeatureService:
    return FeatureService.from_settings(settings=settings)
