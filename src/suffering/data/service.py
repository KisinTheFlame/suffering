"""High-level service API for raw daily data access."""

from __future__ import annotations

import pandas as pd

from suffering.config.settings import Settings, get_settings
from suffering.data.providers import YFinanceDailyProvider
from suffering.data.storage import DailyDataStorage
from suffering.data.universe import resolve_symbols


class DailyDataService:
    def __init__(
        self,
        storage: DailyDataStorage,
        provider: YFinanceDailyProvider,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "DailyDataService":
        resolved_settings = settings or get_settings()
        provider_name = resolved_settings.default_data_provider.lower()
        if provider_name != "yfinance":
            raise ValueError(
                f"Unsupported data provider: {resolved_settings.default_data_provider}"
            )

        return cls(
            storage=DailyDataStorage.from_settings(resolved_settings),
            provider=YFinanceDailyProvider(),
            settings=resolved_settings,
        )

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        resolved_start_date = start_date or self.settings.default_start_date
        resolved_end_date = end_date or self.settings.default_end_date
        frame = self.provider.fetch_daily_data(
            symbol=symbol,
            start_date=resolved_start_date,
            end_date=resolved_end_date,
        )
        self.storage.write_daily_data(symbol=symbol, frame=frame)
        return frame

    def get_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if use_cache and self.storage.exists(symbol):
            frame = self.storage.read_daily_data(symbol)
        else:
            frame = self.fetch_daily_data(symbol=symbol, start_date=start_date, end_date=end_date)

        if start_date:
            frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
        return frame.reset_index(drop=True)

    def get_universe_daily_data(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        resolved_symbols = resolve_symbols(symbols=symbols, settings=self.settings)
        return {
            symbol: self.get_daily_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )
            for symbol in resolved_symbols
        }


def build_data_service(settings: Settings | None = None) -> DailyDataService:
    return DailyDataService.from_settings(settings=settings)
