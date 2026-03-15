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
