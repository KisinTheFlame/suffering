from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from suffering.ranking.storage import RankingStorage


def build_sample_label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "future_return_5d": [0.05, None],
        }
    )


def build_sample_dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["AAPL", "MSFT"],
            "feature_alpha": [1.0, 2.0],
            "future_return_5d": [0.05, -0.02],
            "relevance_5d_5q": [1, 0],
        }
    )


def test_storage_can_write_and_read_daily_labels(tmp_path: Path) -> None:
    storage = RankingStorage(data_dir=tmp_path)
    frame = build_sample_label_frame()

    path = storage.write_daily_labels("aapl", frame)
    loaded = storage.read_daily_labels("AAPL")
    expected = frame.where(pd.notna(frame), np.nan)

    assert path == tmp_path / "labels" / "daily" / "AAPL.csv"
    assert_frame_equal(loaded, expected, check_dtype=False)


def test_storage_can_write_and_read_daily_dataset(tmp_path: Path) -> None:
    storage = RankingStorage(data_dir=tmp_path)
    frame = build_sample_dataset_frame()

    path = storage.write_daily_dataset("panel_5d", frame)
    loaded = storage.read_daily_dataset("panel_5d")

    assert path == tmp_path / "datasets" / "daily" / "panel_5d.csv"
    assert_frame_equal(loaded, frame, check_dtype=False)
