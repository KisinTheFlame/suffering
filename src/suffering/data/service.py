"""High-level service API for raw daily data access."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import BDay

from suffering.config.settings import Settings, get_settings
from suffering.data.models import DAILY_PRICE_COLUMNS, DATE_COLUMN
from suffering.data.providers import YFinanceDailyProvider
from suffering.data.storage import DailyDataStorage
from suffering.data.universe import resolve_symbols


@dataclass(frozen=True)
class DailyDataUpdateResult:
    frame: pd.DataFrame
    action: str
    fetched_rows: int
    cached_rows: int


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

    def update_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        refresh: bool = False,
    ) -> DailyDataUpdateResult:
        resolved_start_date = start_date or self.settings.default_start_date
        resolved_end_date = end_date or self.settings.default_end_date

        if refresh or not self.storage.exists(symbol):
            frame = self.fetch_daily_data(
                symbol=symbol,
                start_date=resolved_start_date,
                end_date=resolved_end_date,
            )
            return DailyDataUpdateResult(
                frame=frame,
                action="full_refresh",
                fetched_rows=int(len(frame)),
                cached_rows=int(len(frame)),
            )

        cached_frame = self.storage.read_daily_data(symbol)
        merged_frame = cached_frame.copy()
        fetched_rows = 0

        resolved_start_ts = pd.Timestamp(resolved_start_date).normalize()
        resolved_end_ts = pd.Timestamp(resolved_end_date).normalize() if resolved_end_date else None
        cached_start_ts = pd.to_datetime(cached_frame[DATE_COLUMN]).min().normalize()
        cached_end_ts = pd.to_datetime(cached_frame[DATE_COLUMN]).max().normalize()

        fetch_ranges: list[tuple[str, str | None]] = []
        if resolved_start_ts < cached_start_ts:
            fetch_ranges.append(
                (
                    resolved_start_ts.strftime("%Y-%m-%d"),
                    (cached_start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
            )

        should_try_append = False
        if resolved_end_ts is not None and resolved_end_ts > cached_end_ts:
            should_try_append = True
        elif resolved_end_ts is None and cached_end_ts < self._latest_expected_market_date():
            should_try_append = True

        if should_try_append:
            fetch_ranges.append(
                (
                    (cached_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    resolved_end_date,
                )
            )

        for fetch_start_date, fetch_end_date in fetch_ranges:
            fetched_frame = self._fetch_range_or_empty(
                symbol=symbol,
                start_date=fetch_start_date,
                end_date=fetch_end_date,
            )
            if fetched_frame.empty:
                continue
            fetched_rows += int(len(fetched_frame))
            merged_frame = self._merge_daily_frames(merged_frame, fetched_frame)

        if fetched_rows > 0:
            self.storage.write_daily_data(symbol=symbol, frame=merged_frame)
            filtered_frame = self._filter_frame_by_dates(
                merged_frame,
                start_date=start_date,
                end_date=end_date,
            )
            return DailyDataUpdateResult(
                frame=filtered_frame,
                action="incremental_update",
                fetched_rows=fetched_rows,
                cached_rows=int(len(merged_frame)),
            )

        filtered_cached_frame = self._filter_frame_by_dates(
            cached_frame,
            start_date=start_date,
            end_date=end_date,
        )
        return DailyDataUpdateResult(
            frame=filtered_cached_frame,
            action="cache_hit",
            fetched_rows=0,
            cached_rows=int(len(cached_frame)),
        )

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
        return self._filter_frame_by_dates(frame, start_date=start_date, end_date=end_date)

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

    def _fetch_range_or_empty(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None,
    ) -> pd.DataFrame:
        try:
            return self.provider.fetch_daily_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except ValueError as exc:
            if "No daily data returned for symbol" not in str(exc):
                raise
            return _empty_daily_frame()

    def _filter_frame_by_dates(
        self,
        frame: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        output = frame.copy()
        if start_date:
            output = output.loc[output[DATE_COLUMN] >= pd.Timestamp(start_date)]
        if end_date:
            output = output.loc[output[DATE_COLUMN] <= pd.Timestamp(end_date)]
        return output.reset_index(drop=True)

    def _merge_daily_frames(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
    ) -> pd.DataFrame:
        merged = pd.concat([left, right], ignore_index=True)
        merged[DATE_COLUMN] = pd.to_datetime(merged[DATE_COLUMN]).dt.tz_localize(None)
        return (
            merged.sort_values(DATE_COLUMN, kind="stable")
            .drop_duplicates(subset=[DATE_COLUMN], keep="last")
            .reset_index(drop=True)
            .loc[:, DAILY_PRICE_COLUMNS]
        )

    def _latest_expected_market_date(self) -> pd.Timestamp:
        return (pd.Timestamp.today().normalize() - BDay(1)).normalize()


def build_data_service(settings: Settings | None = None) -> DailyDataService:
    return DailyDataService.from_settings(settings=settings)
def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_PRICE_COLUMNS)
