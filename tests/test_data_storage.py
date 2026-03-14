from pathlib import Path

import pandas as pd

from suffering.data.storage import DailyDataStorage


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.5],
            "close": [101.5, 102.5],
            "adj_close": [101.2, 102.2],
            "volume": [1000, 1200],
            "symbol": ["AAPL", "AAPL"],
        }
    )


def test_storage_can_write_and_read_daily_data(tmp_path: Path) -> None:
    storage = DailyDataStorage(data_dir=tmp_path)
    frame = build_sample_frame()

    path = storage.write_daily_data("aapl", frame)
    loaded = storage.read_daily_data("AAPL")

    assert path == tmp_path / "raw" / "daily" / "AAPL.csv"
    assert loaded.equals(frame)
