from pathlib import Path

from suffering.cli import main


class FakeTrainingService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.model_path = root_dir / "models" / "baseline_hist_gbr.pkl"
        self.metrics_path = root_dir / "reports" / "baseline_hist_gbr_metrics.json"
        self.validation_path = root_dir / "predictions" / "baseline_hist_gbr_validation.csv"
        self.test_path = root_dir / "predictions" / "baseline_hist_gbr_test.csv"
        self.walkforward_summary_path = (
            root_dir / "reports" / "baseline_hist_gbr_walkforward_summary.json"
        )
        self.walkforward_folds_path = (
            root_dir / "reports" / "baseline_hist_gbr_walkforward_folds.csv"
        )
        self.walkforward_predictions_path = (
            root_dir / "predictions" / "baseline_hist_gbr_walkforward_test_predictions.csv"
        )

    def train_baseline(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, object]:
        return {
            "dataset_name": dataset_name or "panel_5d",
            "model_name": model_name or "baseline_hist_gbr",
            "total_rows": 30,
            "feature_columns": ["feature_alpha", "feature_beta"],
            "feature_count": 2,
            "split_summary": {
                "train": {
                    "rows": 18,
                    "date_count": 9,
                    "date_start": "2024-01-02",
                    "date_end": "2024-01-12",
                },
                "validation": {
                    "rows": 6,
                    "date_count": 3,
                    "date_start": "2024-01-15",
                    "date_end": "2024-01-17",
                },
                "test": {
                    "rows": 6,
                    "date_count": 3,
                    "date_start": "2024-01-18",
                    "date_end": "2024-01-22",
                },
            },
            "validation_metrics": {
                "mae": 0.1,
                "rmse": 0.12,
                "overall_spearman_corr": 0.5,
                "daily_rank_ic_mean": 0.4,
                "daily_rank_ic_std": 0.1,
                "top_5_mean_future_return": 0.02,
                "top_10_mean_future_return": 0.015,
            },
            "test_metrics": {
                "mae": 0.11,
                "rmse": 0.13,
                "overall_spearman_corr": 0.45,
                "daily_rank_ic_mean": 0.35,
                "daily_rank_ic_std": 0.08,
                "top_5_mean_future_return": 0.018,
                "top_10_mean_future_return": 0.012,
            },
            "artifacts": {
                "model_path": str(self.model_path),
                "metrics_path": str(self.metrics_path),
                "validation_predictions_path": str(self.validation_path),
                "test_predictions_path": str(self.test_path),
            },
        }

    def read_training_report(self, model_name: str | None = None) -> dict[str, object]:
        if model_name == "missing":
            raise FileNotFoundError
        return self.train_baseline(model_name=model_name)

    def train_walkforward(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, object]:
        return {
            "dataset_name": dataset_name or "panel_5d",
            "model_name": model_name or "baseline_hist_gbr",
            "total_rows": 48,
            "date_count": 12,
            "feature_columns": ["feature_alpha", "feature_beta"],
            "feature_count": 2,
            "fold_count": 3,
            "folds": [
                {
                    "fold_id": 1,
                    "train_date_start": "2024-01-02",
                    "train_date_end": "2024-01-05",
                    "validation_date_start": "2024-01-08",
                    "validation_date_end": "2024-01-09",
                    "test_date_start": "2024-01-10",
                    "test_date_end": "2024-01-11",
                },
                {
                    "fold_id": 2,
                    "train_date_start": "2024-01-02",
                    "train_date_end": "2024-01-09",
                    "validation_date_start": "2024-01-10",
                    "validation_date_end": "2024-01-11",
                    "test_date_start": "2024-01-12",
                    "test_date_end": "2024-01-15",
                },
                {
                    "fold_id": 3,
                    "train_date_start": "2024-01-02",
                    "train_date_end": "2024-01-11",
                    "validation_date_start": "2024-01-12",
                    "validation_date_end": "2024-01-15",
                    "test_date_start": "2024-01-16",
                    "test_date_end": "2024-01-17",
                },
            ],
            "notes": [],
            "test_metrics_summary": {
                "mae": {"mean": 0.1, "std": 0.01, "min": 0.09, "max": 0.11},
                "rmse": {"mean": 0.12, "std": 0.01, "min": 0.11, "max": 0.13},
                "overall_spearman_corr": {
                    "mean": 0.5,
                    "std": 0.05,
                    "min": 0.45,
                    "max": 0.55,
                },
                "daily_rank_ic_mean": {
                    "mean": 0.4,
                    "std": 0.04,
                    "min": 0.35,
                    "max": 0.44,
                },
                "daily_rank_ic_std": {
                    "mean": 0.1,
                    "std": 0.02,
                    "min": 0.08,
                    "max": 0.12,
                },
                "top_5_mean_future_return": {
                    "mean": 0.02,
                    "std": 0.001,
                    "min": 0.019,
                    "max": 0.021,
                },
                "top_10_mean_future_return": {
                    "mean": 0.015,
                    "std": 0.001,
                    "min": 0.014,
                    "max": 0.016,
                },
            },
            "artifacts": {
                "summary_path": str(self.walkforward_summary_path),
                "folds_path": str(self.walkforward_folds_path),
                "predictions_path": str(self.walkforward_predictions_path),
            },
        }

    def read_walkforward_report(self, model_name: str | None = None) -> dict[str, object]:
        if model_name == "missing":
            raise FileNotFoundError
        return self.train_walkforward(model_name=model_name)


def test_train_baseline_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-baseline"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset: panel_5d" in captured.out
    assert "model: baseline_hist_gbr" in captured.out
    assert "feature_count: 2" in captured.out


def test_train_show_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    service = FakeTrainingService(tmp_path)
    for path in (
        service.model_path,
        service.metrics_path,
        service.validation_path,
        service.test_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: service,
    )

    exit_code = main(["train-show"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model_exists: True" in captured.out
    assert "validation_predictions_exists: True" in captured.out
    assert "test_predictions_exists: True" in captured.out


def test_train_show_command_reports_missing_report(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-show", "--model-name", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "training report not found for model: missing" in captured.out


def test_train_walkforward_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset: panel_5d" in captured.out
    assert "fold_count: 3" in captured.out
    assert "walkforward_test_metric_means:" in captured.out


def test_train_walkforward_show_command_can_be_called(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    service = FakeTrainingService(tmp_path)
    for path in (
        service.walkforward_summary_path,
        service.walkforward_folds_path,
        service.walkforward_predictions_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: service,
    )

    exit_code = main(["train-walkforward-show"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "summary_exists: True" in captured.out
    assert "folds_exists: True" in captured.out
    assert "predictions_exists: True" in captured.out


def test_train_walkforward_show_reports_missing_report(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward-show", "--model-name", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "walk-forward report not found for model: missing" in captured.out
