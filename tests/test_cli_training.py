from pathlib import Path

from suffering.cli import main

SUPPORTED_MODELS = {"hist_gbr", "xgb_regressor", "xgb_ranker"}


class FakeTrainingService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _artifact_paths(self, model_name: str) -> dict[str, Path]:
        return {
            "model_path": self.root_dir / "models" / f"{model_name}.pkl",
            "metrics_path": self.root_dir / "reports" / f"{model_name}_metrics.json",
            "validation_path": self.root_dir / "predictions" / f"{model_name}_validation.csv",
            "test_path": self.root_dir / "predictions" / f"{model_name}_test.csv",
            "walkforward_summary_path": (
                self.root_dir / "reports" / f"{model_name}_walkforward_summary.json"
            ),
            "walkforward_folds_path": (
                self.root_dir / "reports" / f"{model_name}_walkforward_folds.csv"
            ),
            "walkforward_predictions_path": (
                self.root_dir / "predictions" / f"{model_name}_walkforward_test_predictions.csv"
            ),
        }

    def _resolve_model_name(self, model_name: str | None) -> str:
        resolved_model_name = model_name or "hist_gbr"
        if resolved_model_name not in SUPPORTED_MODELS:
            raise ValueError(
                "Unsupported training model: "
                f"{resolved_model_name}. Supported models: hist_gbr, xgb_regressor, xgb_ranker"
            )
        return resolved_model_name

    def train_baseline(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, object]:
        resolved_model_name = self._resolve_model_name(model_name)
        artifacts = self._artifact_paths(resolved_model_name)
        task_type = "ranking" if resolved_model_name == "xgb_ranker" else "regression"
        training_label_column = (
            "relevance_5d_5q" if resolved_model_name == "xgb_ranker" else "future_return_5d"
        )
        return {
            "dataset_name": dataset_name or "panel_5d",
            "model_name": resolved_model_name,
            "task_type": task_type,
            "training_label_column": training_label_column,
            "evaluation_return_column": "future_return_5d",
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
                "mae": None if task_type == "ranking" else 0.1,
                "rmse": None if task_type == "ranking" else 0.12,
                "overall_spearman_corr": 0.5,
                "daily_rank_ic_mean": 0.4,
                "daily_rank_ic_std": 0.1,
                "top_5_mean_future_return": 0.02,
                "top_10_mean_future_return": 0.015,
                "ndcg_at_5_mean": 0.81,
            },
            "test_metrics": {
                "mae": None if task_type == "ranking" else 0.11,
                "rmse": None if task_type == "ranking" else 0.13,
                "overall_spearman_corr": 0.45,
                "daily_rank_ic_mean": 0.35,
                "daily_rank_ic_std": 0.08,
                "top_5_mean_future_return": 0.018,
                "top_10_mean_future_return": 0.012,
                "ndcg_at_5_mean": 0.79,
            },
            "artifacts": {
                "model_path": str(artifacts["model_path"]),
                "metrics_path": str(artifacts["metrics_path"]),
                "validation_predictions_path": str(artifacts["validation_path"]),
                "test_predictions_path": str(artifacts["test_path"]),
            },
        }

    def read_training_report(self, model_name: str | None = None) -> dict[str, object]:
        resolved_model_name = self._resolve_model_name(model_name)
        if resolved_model_name == "hist_gbr_missing":
            raise FileNotFoundError
        return self.train_baseline(model_name=resolved_model_name)

    def train_walkforward(
        self,
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, object]:
        resolved_model_name = self._resolve_model_name(model_name)
        artifacts = self._artifact_paths(resolved_model_name)
        task_type = "ranking" if resolved_model_name == "xgb_ranker" else "regression"
        return {
            "dataset_name": dataset_name or "panel_5d",
            "model_name": resolved_model_name,
            "task_type": task_type,
            "training_label_column": (
                "relevance_5d_5q" if resolved_model_name == "xgb_ranker" else "future_return_5d"
            ),
            "evaluation_return_column": "future_return_5d",
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
                "mae": {
                    "mean": None if task_type == "ranking" else 0.1,
                    "std": None if task_type == "ranking" else 0.01,
                    "min": None if task_type == "ranking" else 0.09,
                    "max": None if task_type == "ranking" else 0.11,
                },
                "rmse": {
                    "mean": None if task_type == "ranking" else 0.12,
                    "std": None if task_type == "ranking" else 0.01,
                    "min": None if task_type == "ranking" else 0.11,
                    "max": None if task_type == "ranking" else 0.13,
                },
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
                "ndcg_at_5_mean": {
                    "mean": 0.8,
                    "std": 0.02,
                    "min": 0.78,
                    "max": 0.82,
                },
            },
            "artifacts": {
                "summary_path": str(artifacts["walkforward_summary_path"]),
                "folds_path": str(artifacts["walkforward_folds_path"]),
                "predictions_path": str(artifacts["walkforward_predictions_path"]),
            },
        }

    def read_walkforward_report(self, model_name: str | None = None) -> dict[str, object]:
        resolved_model_name = self._resolve_model_name(model_name)
        if resolved_model_name == "hist_gbr_missing":
            raise FileNotFoundError
        return self.train_walkforward(model_name=resolved_model_name)


def test_train_baseline_command_supports_hist_gbr(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-baseline", "--model", "hist_gbr"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset: panel_5d" in captured.out
    assert "model: hist_gbr" in captured.out
    assert "task_type: regression" in captured.out
    assert "feature_count: 2" in captured.out


def test_train_baseline_command_supports_xgb_regressor(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-baseline", "--model", "xgb_regressor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_regressor" in captured.out
    assert "xgb_regressor_metrics.json" in captured.out


def test_train_baseline_command_supports_xgb_ranker(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-baseline", "--model", "xgb_ranker"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_ranker" in captured.out
    assert "task_type: ranking" in captured.out
    assert "ndcg_at_5_mean:" in captured.out


def test_train_baseline_command_reports_unknown_model(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-baseline", "--model", "unknown_model"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unsupported training model: unknown_model" in captured.out


def test_train_show_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    service = FakeTrainingService(tmp_path)
    artifacts = service._artifact_paths("xgb_regressor")
    for path in (
        artifacts["model_path"],
        artifacts["metrics_path"],
        artifacts["validation_path"],
        artifacts["test_path"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: service,
    )

    exit_code = main(["train-show", "--model", "xgb_regressor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_regressor" in captured.out
    assert "task_type: regression" in captured.out
    assert "model_exists: True" in captured.out
    assert "validation_predictions_exists: True" in captured.out
    assert "test_predictions_exists: True" in captured.out


def test_train_walkforward_command_supports_hist_gbr(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward", "--model", "hist_gbr"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset: panel_5d" in captured.out
    assert "model: hist_gbr" in captured.out
    assert "task_type: regression" in captured.out
    assert "fold_count: 3" in captured.out
    assert "walkforward_test_metric_means:" in captured.out


def test_train_walkforward_command_supports_xgb_regressor(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward", "--model", "xgb_regressor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_regressor" in captured.out
    assert "xgb_regressor_walkforward_summary.json" in captured.out


def test_train_walkforward_command_supports_xgb_ranker(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward", "--model", "xgb_ranker"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_ranker" in captured.out
    assert "task_type: ranking" in captured.out
    assert "xgb_ranker_walkforward_summary.json" in captured.out


def test_train_walkforward_show_command_can_be_called(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    service = FakeTrainingService(tmp_path)
    artifacts = service._artifact_paths("xgb_regressor")
    for path in (
        artifacts["walkforward_summary_path"],
        artifacts["walkforward_folds_path"],
        artifacts["walkforward_predictions_path"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: service,
    )

    exit_code = main(["train-walkforward-show", "--model", "xgb_regressor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_regressor" in captured.out
    assert "task_type: regression" in captured.out
    assert "summary_exists: True" in captured.out
    assert "folds_exists: True" in captured.out
    assert "predictions_exists: True" in captured.out


def test_train_walkforward_show_reports_unknown_model(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_training_service",
        lambda settings=None: FakeTrainingService(tmp_path),
    )

    exit_code = main(["train-walkforward-show", "--model", "unknown_model"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unsupported training model: unknown_model" in captured.out
