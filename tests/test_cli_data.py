import pandas as pd

from suffering.cli import main
from suffering.data.service import DailyDataUpdateResult
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
        self.last_max_workers: int | None = None

    def update_daily_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        refresh: bool = False,
    ) -> DailyDataUpdateResult:
        frame = self.storage.read_daily_data("AAPL").assign(symbol=symbol.upper())
        action = "full_refresh" if refresh else "cache_hit"
        fetched_rows = len(frame) if refresh else 0
        return DailyDataUpdateResult(
            frame=frame,
            action=action,
            fetched_rows=fetched_rows,
            cached_rows=len(frame),
        )

    def update_many_daily_data(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        refresh: bool = False,
        max_workers: int | None = None,
        retries: int = 1,
    ):
        from suffering.data.service import BatchDailyDataUpdateResult

        self.last_max_workers = max_workers
        return [
            BatchDailyDataUpdateResult(
                symbol=symbol.upper(),
                result=self.update_daily_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                ),
                error=None,
            )
            for symbol in symbols
        ]


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

    def update_features_for_symbol(self, symbol: str, refresh: bool = False):
        if symbol.upper() != "AAPL":
            raise FileNotFoundError(
                f"raw daily data not found for {symbol.upper()}. "
                f"Run `suffering data-fetch {symbol.upper()}` first."
            )
        from suffering.features.service import FeatureBuildResult

        frame = self.storage.read_daily_features("AAPL").assign(symbol=symbol.upper())
        action = "rebuilt" if refresh else "cache_hit"
        return FeatureBuildResult(frame=frame, action=action, cached_rows=len(frame))

    def read_features(self, symbol: str) -> pd.DataFrame:
        return self.storage.read_daily_features(symbol)


def test_data_fetch_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: FakeService())

    exit_code = main(["data-fetch", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AAPL: cache hit, 1 cached rows at data/raw/daily/AAPL.csv" in captured.out


def test_data_fetch_command_supports_refresh(monkeypatch, capsys) -> None:
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: FakeService())

    exit_code = main(["data-fetch", "AAPL", "--refresh"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AAPL: full refresh cached 1 rows at data/raw/daily/AAPL.csv" in captured.out


def test_data_fetch_command_accepts_max_workers(monkeypatch, capsys) -> None:
    fake_service = FakeService()
    monkeypatch.setattr("suffering.cli.build_data_service", lambda settings=None: fake_service)

    exit_code = main(["data-fetch", "AAPL", "--max-workers", "4"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "AAPL: cache hit, 1 cached rows at data/raw/daily/AAPL.csv" in captured.out
    assert fake_service.last_max_workers == 4


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
    assert (
        "AAPL: cache hit, 2 rows and 18 feature columns at data/features/daily/AAPL.csv"
        in captured.out
    )


def test_feature_build_command_supports_refresh(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_feature_service",
        lambda settings=None: FakeFeatureService(),
    )

    exit_code = main(["feature-build", "AAPL", "--refresh"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "AAPL: rebuilt 2 rows, 18 feature columns cached at data/features/daily/AAPL.csv"
        in captured.out
    )


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
