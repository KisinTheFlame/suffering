import pandas as pd

from suffering.cli import main


class FakeStorage:
    def __init__(self) -> None:
        self._frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "adj_close": [100.4],
                "volume": [1000],
                "symbol": ["AAPL"],
            }
        )

    def path_for_symbol(self, symbol: str) -> str:
        return f"data/raw/daily/{symbol.upper()}.csv"

    def read_daily_data(self, symbol: str) -> pd.DataFrame:
        if symbol.upper() != "AAPL":
            raise FileNotFoundError
        return self._frame


class FakeService:
    def __init__(self) -> None:
        self.storage = FakeStorage()

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.storage.read_daily_data("AAPL").assign(symbol=symbol.upper())


def test_data_fetch_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: FakeService())

    exit_code = main(["data-fetch", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AAPL: 1 rows cached at data/raw/daily/AAPL.csv" in captured.out


def test_data_show_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: FakeService())

    exit_code = main(["data-show", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "symbol: AAPL" in captured.out
    assert "rows: 1" in captured.out


def test_data_show_command_reports_missing_cache(monkeypatch, capsys) -> None:
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: FakeService())

    exit_code = main(["data-show", "MSFT"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "cached data not found for MSFT" in captured.out
