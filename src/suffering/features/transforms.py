"""Pure transforms for daily feature engineering."""

from __future__ import annotations

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.features.definitions import (
    FEATURE_OUTPUT_COLUMNS,
    REQUIRED_DAILY_FEATURE_INPUT_COLUMNS,
)


def build_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal daily feature table for a single symbol."""
    missing_columns = [
        column for column in REQUIRED_DAILY_FEATURE_INPUT_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing expected columns for feature generation: {missing_text}")

    if frame.empty:
        return pd.DataFrame(columns=FEATURE_OUTPUT_COLUMNS)

    output = frame.loc[:, REQUIRED_DAILY_FEATURE_INPUT_COLUMNS].copy()
    output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    output[SYMBOL_COLUMN] = output[SYMBOL_COLUMN].astype(str).str.strip().str.upper()
    output = output.sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable").reset_index(drop=True)

    unique_symbols = output[SYMBOL_COLUMN].dropna().unique()
    if len(unique_symbols) != 1:
        raise ValueError("Daily feature generation expects exactly one symbol per input frame.")

    close = output["close"].astype("float64")
    open_price = output["open"].astype("float64")
    high = output["high"].astype("float64")
    low = output["low"].astype("float64")
    volume = output["volume"].astype("float64")

    prev_close = close.shift(1)
    daily_return = close.pct_change()

    output["return_1d"] = daily_return
    output["return_5d"] = close.pct_change(5)
    output["return_10d"] = close.pct_change(10)
    output["return_20d"] = close.pct_change(20)
    output["return_60d"] = close.pct_change(60)

    # Rolling windows only use the current row and prior rows, which avoids future leakage.
    output["volatility_5d"] = daily_return.rolling(window=5, min_periods=5).std()
    output["volatility_20d"] = daily_return.rolling(window=20, min_periods=20).std()
    output["volatility_60d"] = daily_return.rolling(window=60, min_periods=60).std()

    output["sma_5_ratio"] = close / close.rolling(window=5, min_periods=5).mean() - 1.0
    output["sma_20_ratio"] = close / close.rolling(window=20, min_periods=20).mean() - 1.0
    output["sma_60_ratio"] = close / close.rolling(window=60, min_periods=60).mean() - 1.0

    safe_close = close.where(close != 0.0)
    safe_open = open_price.where(open_price != 0.0)
    safe_prev_close = prev_close.where(prev_close != 0.0)

    output["intraday_range_1d"] = (high - low) / safe_close
    output["open_to_close_return_1d"] = close / safe_open - 1.0
    output["gap_return_1d"] = open_price / safe_prev_close - 1.0

    output["volume_change_1d"] = volume.pct_change()
    output["avg_volume_5d"] = volume.rolling(window=5, min_periods=5).mean()
    output["avg_volume_20d"] = volume.rolling(window=20, min_periods=20).mean()
    output["avg_dollar_volume_20d"] = (close * volume).rolling(window=20, min_periods=20).mean()

    return (
        output.loc[:, FEATURE_OUTPUT_COLUMNS]
        .sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable")
        .reset_index(drop=True)
    )
