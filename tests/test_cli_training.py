from pathlib import Path

from suffering.cli import main


class FakeTrainingService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.model_path = root_dir / "models" / "baseline_hist_gbr.pkl"
        self.metrics_path = root_dir / "reports" / "baseline_hist_gbr_metrics.json"
        self.validation_path = root_dir / "predictions" / "baseline_hist_gbr_validation.csv"
        self.test_path = root_dir / "predictions" / "baseline_hist_gbr_test.csv"

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
