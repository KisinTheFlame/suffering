from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from suffering.features.storage import DailyFeatureStorage


def build_sample_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "return_1d": [None, 0.01],
            "return_5d": [None, None],
            "return_10d": [None, None],
            "return_20d": [None, None],
            "return_60d": [None, None],
            "volatility_5d": [None, None],
            "volatility_20d": [None, None],
            "volatility_60d": [None, None],
            "sma_5_ratio": [None, None],
            "sma_20_ratio": [None, None],
            "sma_60_ratio": [None, None],
            "intraday_range_1d": [0.02, 0.018],
            "open_to_close_return_1d": [0.005, 0.004],
            "gap_return_1d": [None, 0.003],
            "volume_change_1d": [None, 0.2],
            "avg_volume_5d": [None, None],
            "avg_volume_20d": [None, None],
            "avg_dollar_volume_20d": [None, None],
        }
    )


def test_storage_can_write_and_read_daily_features(tmp_path: Path) -> None:
    storage = DailyFeatureStorage(data_dir=tmp_path)
    frame = build_sample_feature_frame()

    path = storage.write_daily_features("aapl", frame)
    loaded = storage.read_daily_features("AAPL")
    expected = frame.where(pd.notna(frame), np.nan)

    assert path == tmp_path / "features" / "daily" / "AAPL.csv"
    assert_frame_equal(loaded, expected, check_dtype=False)
