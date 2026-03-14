"""Shared constants for the minimal daily feature layer."""

from __future__ import annotations

from suffering.data.models import DAILY_PRICE_COLUMNS, DATE_COLUMN, SYMBOL_COLUMN

RETURN_WINDOWS = (1, 5, 10, 20, 60)
VOLATILITY_WINDOWS = (5, 20, 60)
SMA_WINDOWS = (5, 20, 60)
AVG_VOLUME_WINDOWS = (5, 20)

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_60_ratio",
    "intraday_range_1d",
    "open_to_close_return_1d",
    "gap_return_1d",
    "volume_change_1d",
    "avg_volume_5d",
    "avg_volume_20d",
    "avg_dollar_volume_20d",
]

FEATURE_OUTPUT_COLUMNS = [
    DATE_COLUMN,
    SYMBOL_COLUMN,
    *FEATURE_COLUMNS,
]

REQUIRED_DAILY_FEATURE_INPUT_COLUMNS = list(DAILY_PRICE_COLUMNS)
