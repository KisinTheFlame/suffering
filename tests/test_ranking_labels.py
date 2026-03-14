import math

import pandas as pd

from suffering.ranking.labels import (
    FUTURE_RETURN_5D_COLUMN,
    LABEL_OUTPUT_COLUMNS,
    build_daily_labels,
)


def build_sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=8, freq="B"),
            "open": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "high": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
            "low": [9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0],
            "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "adj_close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "volume": [1_000, 1_100, 1_200, 1_300, 1_400, 1_500, 1_600, 1_700],
            "symbol": ["aapl"] * 8,
        }
    )


def test_build_daily_labels_uses_open_t_plus_1_and_close_t_plus_5() -> None:
    raw_frame = build_sample_raw_frame().iloc[[2, 0, 1, 4, 3, 7, 6, 5]].reset_index(drop=True)

    label_frame = build_daily_labels(raw_frame)

    assert list(label_frame.columns) == LABEL_OUTPUT_COLUMNS
    assert label_frame["symbol"].tolist() == ["AAPL"] * 8
    assert label_frame["date"].tolist() == sorted(label_frame["date"].tolist())

    assert math.isclose(
        label_frame.loc[0, FUTURE_RETURN_5D_COLUMN],
        150.0 / 20.0 - 1.0,
    )
    assert math.isclose(
        label_frame.loc[1, FUTURE_RETURN_5D_COLUMN],
        160.0 / 30.0 - 1.0,
    )
    assert math.isclose(
        label_frame.loc[2, FUTURE_RETURN_5D_COLUMN],
        170.0 / 40.0 - 1.0,
    )


def test_build_daily_labels_leaves_tail_rows_without_future_data_as_nan() -> None:
    label_frame = build_daily_labels(build_sample_raw_frame())

    assert label_frame[FUTURE_RETURN_5D_COLUMN].notna().sum() == 3
    assert label_frame[FUTURE_RETURN_5D_COLUMN].tail(5).isna().all()
