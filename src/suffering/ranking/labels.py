"""Pure label generation helpers for per-symbol daily data."""

from __future__ import annotations

import pandas as pd

from suffering.data.models import DAILY_PRICE_COLUMNS, DATE_COLUMN, SYMBOL_COLUMN

FUTURE_RETURN_5D_COLUMN = "future_return_5d"
LABEL_HORIZON_DAYS = 5
LABEL_OUTPUT_COLUMNS = [DATE_COLUMN, SYMBOL_COLUMN, FUTURE_RETURN_5D_COLUMN]


def build_daily_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the minimal 5d forward return label table for a single symbol."""
    missing_columns = [column for column in DAILY_PRICE_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing expected columns for label generation: {missing_text}")

    if frame.empty:
        return pd.DataFrame(columns=LABEL_OUTPUT_COLUMNS)

    output = frame.loc[:, DAILY_PRICE_COLUMNS].copy()
    output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    output[SYMBOL_COLUMN] = output[SYMBOL_COLUMN].astype(str).str.strip().str.upper()
    output = output.sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable").reset_index(drop=True)

    unique_symbols = output[SYMBOL_COLUMN].dropna().unique()
    if len(unique_symbols) != 1:
        raise ValueError("Daily label generation expects exactly one symbol per input frame.")

    open_price = output["open"].astype("float64")
    close_price = output["close"].astype("float64")

    # The label is attached to signal date t, while the realized path starts at open[t+1].
    next_open = open_price.shift(-1)
    future_close = close_price.shift(-LABEL_HORIZON_DAYS)
    safe_next_open = next_open.where(next_open != 0.0)

    output[FUTURE_RETURN_5D_COLUMN] = future_close / safe_next_open - 1.0

    return (
        output.loc[:, LABEL_OUTPUT_COLUMNS]
        .sort_values([SYMBOL_COLUMN, DATE_COLUMN], kind="stable")
        .reset_index(drop=True)
    )
