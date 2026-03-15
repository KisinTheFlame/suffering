import sys

import pandas as pd
import pytest

from suffering.data.providers.yfinance_provider import YFinanceDailyProvider


def test_yfinance_provider_normalizes_output(monkeypatch) -> None:
    sample = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.8, 101.8],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    sample.index.name = "Date"

    monkeypatch.setattr(
        "suffering.data.providers.yfinance_provider.yf.download",
        lambda **kwargs: sample,
    )

    provider = YFinanceDailyProvider()
    frame = provider.fetch_daily_data("aapl", start_date="2024-01-01", end_date="2024-01-31")

    assert list(frame.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "symbol",
    ]
    assert frame["symbol"].unique().tolist() == ["AAPL"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]


def test_yfinance_provider_suppresses_expected_empty_download_stderr(
    monkeypatch,
    capsys,
) -> None:
    def fake_download(**kwargs):
        print(
            "1 Failed download: ['AAPL']: "
            "YFPricesMissingError('possibly delisted; no price data found')",
            file=sys.stderr,
        )
        return pd.DataFrame()

    monkeypatch.setattr(
        "suffering.data.providers.yfinance_provider.yf.download",
        fake_download,
    )

    provider = YFinanceDailyProvider()
    with pytest.raises(ValueError, match="No daily data returned for symbol: AAPL"):
        provider.fetch_daily_data("aapl", start_date="2024-01-01", end_date="2024-01-31")

    captured = capsys.readouterr()
    assert captured.err == ""
