import math

import pandas as pd

from suffering.features.definitions import FEATURE_OUTPUT_COLUMNS
from suffering.features.transforms import build_daily_features


def build_sample_daily_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-02", periods=70, freq="B")

    for index, date in enumerate(dates):
        close = 100.0 + index
        rows.append(
            {
                "date": date,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "adj_close": close - 0.2,
                "volume": 1_000 + index * 10,
                "symbol": "aapl",
            }
        )

    return pd.DataFrame(rows[::-1])


def test_build_daily_features_has_expected_columns_and_sorting() -> None:
    frame = build_sample_daily_frame()

    feature_frame = build_daily_features(frame)

    assert list(feature_frame.columns) == FEATURE_OUTPUT_COLUMNS
    assert feature_frame["symbol"].tolist() == ["AAPL"] * len(feature_frame)
    assert feature_frame["date"].is_monotonic_increasing


def test_build_daily_features_computes_key_metrics_without_leakage() -> None:
    frame = build_sample_daily_frame()

    feature_frame = build_daily_features(frame)
    row_19 = feature_frame.iloc[19]
    closes = feature_frame["return_1d"]

    expected_return_1d = 119.0 / 118.0 - 1.0
    expected_gap_return = 118.5 / 118.0 - 1.0
    expected_intraday_range = (120.0 - 118.0) / 119.0
    expected_avg_dollar_volume_20d = sum(
        (100.0 + index) * (1_000 + index * 10) for index in range(20)
    ) / 20.0

    assert math.isclose(feature_frame.iloc[19]["return_1d"], expected_return_1d)
    assert math.isclose(feature_frame.iloc[19]["gap_return_1d"], expected_gap_return)
    assert math.isclose(feature_frame.iloc[19]["intraday_range_1d"], expected_intraday_range)
    assert math.isclose(row_19["avg_dollar_volume_20d"], expected_avg_dollar_volume_20d)
    assert pd.isna(feature_frame.iloc[0]["return_1d"])
    assert pd.isna(feature_frame.iloc[3]["avg_volume_5d"])
    assert pd.isna(feature_frame.iloc[18]["return_20d"])
    assert pd.isna(feature_frame.iloc[58]["volatility_60d"])
    assert not pd.isna(closes.iloc[1])
