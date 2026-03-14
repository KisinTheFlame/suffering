"""yfinance-backed provider for daily OHLCV market data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from suffering.data.models import DAILY_PRICE_COLUMNS, DATE_COLUMN, SYMBOL_COLUMN, normalize_symbol


class YFinanceDailyProvider:
    provider_name = "yfinance"

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        frame = yf.download(
            tickers=normalized_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        if frame.empty:
            raise ValueError(f"No daily data returned for symbol: {normalized_symbol}")

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        output = frame.reset_index().rename(
            columns={
                "Date": DATE_COLUMN,
                "Datetime": DATE_COLUMN,
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        missing_columns = [column for column in DAILY_PRICE_COLUMNS if column not in output.columns]
        if SYMBOL_COLUMN in missing_columns:
            missing_columns.remove(SYMBOL_COLUMN)
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Missing expected columns from provider response: {missing_text}")

        output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN], utc=True).dt.tz_localize(None)
        output[DATE_COLUMN] = output[DATE_COLUMN].dt.normalize()
        output["volume"] = output["volume"].astype("int64")
        output[SYMBOL_COLUMN] = normalized_symbol

        ordered = output.loc[:, DAILY_PRICE_COLUMNS].sort_values(DATE_COLUMN).reset_index(drop=True)
        return ordered
