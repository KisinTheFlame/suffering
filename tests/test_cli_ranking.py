import pandas as pd

from suffering.cli import main
from suffering.ranking.service import DatasetBuildResult, LabelBuildResult


class FakeRankingStorage:
    def label_path_for_symbol(self, symbol: str) -> str:
        return f"data/labels/daily/{symbol.upper()}.csv"

    def dataset_path(self, dataset_name: str) -> str:
        return f"data/datasets/daily/{dataset_name}.csv"


class FakeRankingService:
    def __init__(self) -> None:
        self.storage = FakeRankingStorage()
        self._label_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["AAPL", "AAPL"],
                "future_return_5d": [0.05, None],
            }
        )
        self._dataset_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "symbol": ["AAPL", "MSFT"],
                "feature_alpha": [1.0, 2.0],
                "future_return_5d": [0.05, -0.02],
                "relevance_5d_5q": [1, 0],
            }
        )

    def update_label_for_symbol(self, symbol: str, refresh: bool = False) -> LabelBuildResult:
        if symbol.upper() != "AAPL":
            raise FileNotFoundError(
                f"raw daily data not found for {symbol.upper()}. "
                f"Run `suffering data-fetch {symbol.upper()}` first."
            )
        return LabelBuildResult(
            frame=self._label_frame,
            action="rebuilt" if refresh else "cache_hit",
            cached_rows=len(self._label_frame),
        )

    def update_panel_dataset(
        self,
        symbols: list[str] | None = None,
        dataset_name: str | None = None,
        write_cache: bool = True,
        refresh: bool = False,
    ) -> DatasetBuildResult:
        return DatasetBuildResult(
            frame=self._dataset_frame,
            action="rebuilt" if refresh else "cache_hit",
            cached_rows=len(self._dataset_frame),
        )

    def read_panel_dataset(self, dataset_name: str | None = None) -> pd.DataFrame:
        if dataset_name == "missing":
            raise FileNotFoundError
        return self._dataset_frame


def test_label_build_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["label-build", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "AAPL: cache hit, 2 rows and 1 labeled rows at data/labels/daily/AAPL.csv"
        in captured.out
    )


def test_label_build_command_supports_refresh(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["label-build", "AAPL", "--refresh"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "AAPL: rebuilt 2 rows, 1 labeled rows cached at data/labels/daily/AAPL.csv"
        in captured.out
    )


def test_dataset_build_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["dataset-build", "AAPL"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "action: cache_hit" in captured.out
    assert "rows: 2" in captured.out
    assert "cached at: data/datasets/daily/panel_5d.csv" in captured.out


def test_dataset_build_command_supports_refresh(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["dataset-build", "AAPL", "--refresh"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "action: rebuilt" in captured.out


def test_dataset_show_command_can_be_called(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["dataset-show"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset: panel_5d" in captured.out
    assert "has_future_return_5d: True" in captured.out
    assert "has_relevance_5d_5q: True" in captured.out


def test_dataset_show_command_reports_missing_cache(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_ranking_service",
        lambda settings=None: FakeRankingService(),
    )

    exit_code = main(["dataset-show", "--dataset-name", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "cached dataset not found: missing" in captured.out
