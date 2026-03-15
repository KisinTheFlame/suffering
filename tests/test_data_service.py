from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings
from suffering.data.service import DailyDataService
from suffering.data.storage import DailyDataStorage


class RecordingProvider:
    def __init__(self, frames: dict[tuple[str, str | None], pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, str | None]] = []

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        key = (start_date or "", end_date)
        self.calls.append(key)
        frame = self.frames.get(key)
        if frame is None:
            raise ValueError(f"No daily data returned for symbol: {symbol.upper()}")
        return frame.copy()


class SymbolAwareProvider:
    def __init__(
        self,
        frames: dict[tuple[str, str | None, str | None], pd.DataFrame],
        failures_before_success: dict[str, int] | None = None,
    ) -> None:
        self.frames = frames
        self.failures_before_success = failures_before_success or {}
        self.calls: list[tuple[str, str | None, str | None]] = []

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        normalized_symbol = symbol.upper()
        self.calls.append((normalized_symbol, start_date, end_date))
        remaining_failures = self.failures_before_success.get(normalized_symbol, 0)
        if remaining_failures > 0:
            self.failures_before_success[normalized_symbol] = remaining_failures - 1
            raise ValueError(f"temporary failure for {normalized_symbol}")

        key = (normalized_symbol, start_date, end_date)
        frame = self.frames.get(key)
        if frame is None:
            raise ValueError(f"No daily data returned for symbol: {normalized_symbol}")
        return frame.copy()


def build_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [100.0 + index for index in range(len(dates))],
            "high": [101.0 + index for index in range(len(dates))],
            "low": [99.0 + index for index in range(len(dates))],
            "close": [100.5 + index for index in range(len(dates))],
            "adj_close": [100.3 + index for index in range(len(dates))],
            "volume": [1000 + index for index in range(len(dates))],
            "symbol": ["AAPL"] * len(dates),
        }
    )


def test_update_daily_data_returns_cache_hit_when_requested_range_is_already_cached(
    tmp_path: Path,
) -> None:
    storage = DailyDataStorage(data_dir=tmp_path)
    cached_frame = build_frame(["2024-01-02", "2024-01-03"])
    storage.write_daily_data("AAPL", cached_frame)

    service = DailyDataService(
        storage=storage,
        provider=RecordingProvider({}),
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-01"),
    )

    result = service.update_daily_data("AAPL", start_date="2024-01-02", end_date="2024-01-03")

    assert result.action == "cache_hit"
    assert result.fetched_rows == 0
    assert result.cached_rows == 2
    assert result.frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]


def test_update_daily_data_appends_new_rows_after_cached_tail(tmp_path: Path) -> None:
    storage = DailyDataStorage(data_dir=tmp_path)
    cached_frame = build_frame(["2024-01-02", "2024-01-03"])
    storage.write_daily_data("AAPL", cached_frame)

    incremental_frame = build_frame(["2024-01-04", "2024-01-05"])
    provider = RecordingProvider({("2024-01-04", "2024-01-05"): incremental_frame})
    service = DailyDataService(
        storage=storage,
        provider=provider,
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-02"),
    )

    result = service.update_daily_data("AAPL", end_date="2024-01-05")

    assert provider.calls == [("2024-01-04", "2024-01-05")]
    assert result.action == "incremental_update"
    assert result.fetched_rows == 2
    assert result.cached_rows == 4
    assert result.frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_fetch_daily_data_clips_future_end_date_to_latest_expected_market_date(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider({("2024-01-02", "2024-01-05"): build_frame(["2024-01-02"])})
    service = DailyDataService(
        storage=DailyDataStorage(data_dir=tmp_path),
        provider=provider,
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-02"),
    )
    service._latest_expected_market_date = lambda: pd.Timestamp("2024-01-05")  # type: ignore[method-assign]

    result = service.fetch_daily_data("AAPL", start_date="2024-01-02", end_date="2025-12-31")

    assert provider.calls == [("2024-01-02", "2024-01-05")]
    assert len(result) == 1


def test_update_daily_data_does_not_fetch_future_tail_beyond_latest_expected_market_date(
    tmp_path: Path,
) -> None:
    storage = DailyDataStorage(data_dir=tmp_path)
    cached_frame = build_frame(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    storage.write_daily_data("AAPL", cached_frame)
    provider = RecordingProvider({})
    service = DailyDataService(
        storage=storage,
        provider=provider,
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-02"),
    )
    service._latest_expected_market_date = lambda: pd.Timestamp("2024-01-05")  # type: ignore[method-assign]

    result = service.update_daily_data("AAPL", end_date="2025-12-31")

    assert provider.calls == []
    assert result.action == "cache_hit"
    assert result.cached_rows == 4


def test_update_many_daily_data_returns_results_in_input_order(tmp_path: Path) -> None:
    provider = SymbolAwareProvider(
        {
            ("MSFT", "2024-01-02", "2024-01-05"): build_frame(["2024-01-02"]),
            ("AAPL", "2024-01-02", "2024-01-05"): build_frame(["2024-01-03"]),
        }
    )
    service = DailyDataService(
        storage=DailyDataStorage(data_dir=tmp_path),
        provider=provider,
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-02"),
    )

    results = service.update_many_daily_data(
        ["MSFT", "AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-05",
        max_workers=2,
    )

    assert [item.symbol for item in results] == ["MSFT", "AAPL"]
    assert [item.result.action for item in results if item.result is not None] == [
        "full_refresh",
        "full_refresh",
    ]


def test_update_many_daily_data_retries_failed_symbols(tmp_path: Path) -> None:
    provider = SymbolAwareProvider(
        {("AAPL", "2024-01-02", "2024-01-05"): build_frame(["2024-01-02"])},
        failures_before_success={"AAPL": 1},
    )
    service = DailyDataService(
        storage=DailyDataStorage(data_dir=tmp_path),
        provider=provider,
        settings=Settings(data_dir=tmp_path, default_start_date="2024-01-02"),
    )

    results = service.update_many_daily_data(
        ["AAPL"],
        start_date="2024-01-02",
        end_date="2024-01-05",
        retries=2,
    )

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].result is not None
    assert results[0].result.action == "full_refresh"
    assert provider.calls == [
        ("AAPL", "2024-01-02", "2024-01-05"),
        ("AAPL", "2024-01-02", "2024-01-05"),
    ]
