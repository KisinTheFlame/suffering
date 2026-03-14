import pandas as pd

from suffering.cli import main
from suffering.features.definitions import FEATURE_COLUMNS


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


class FakeFeatureStorage:
    def __init__(self) -> None:
        self._frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["AAPL", "AAPL"],
                **{column: [None, 0.1] for column in FEATURE_COLUMNS},
            }
        )

    def path_for_symbol(self, symbol: str) -> str:
        return f"data/features/daily/{symbol.upper()}.csv"

    def read_daily_features(self, symbol: str) -> pd.DataFrame:
        if symbol.upper() != "AAPL":
            raise FileNotFoundError
        return self._frame


class FakeFeatureService:
    def __init__(self) -> None:
        self.storage = FakeFeatureStorage()

    def build_features_for_symbol(self, symbol: str) -> pd.DataFrame:
        if symbol.upper() != "AAPL":
            raise FileNotFoundError(
                f"raw daily data not found for {symbol.upper()}. "
                f"Run `suffering data-fetch {symbol.upper()}` first."
            )
        return self.storage.read_daily_features("AAPL").assign(symbol=symbol.upper())

    def read_features(self, symbol: str) -> pd.DataFrame:
        return self.storage.read_daily_features(symbol)


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


def test_feature_build_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_feature_service",
        lambda settings=None: FakeFeatureService(),
    )

    exit_code = main(["feature-build", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AAPL: 2 rows, 18 feature columns cached at data/features/daily/AAPL.csv" in captured.out


def test_feature_show_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_feature_service",
        lambda settings=None: FakeFeatureService(),
    )

    exit_code = main(["feature-show", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "symbol: AAPL" in captured.out
    assert "feature_columns:" in captured.out


def test_feature_show_command_reports_missing_cache(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_feature_service",
        lambda settings=None: FakeFeatureService(),
    )

    exit_code = main(["feature-show", "MSFT"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "cached features not found for MSFT" in captured.out
