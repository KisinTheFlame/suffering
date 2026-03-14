from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings
from suffering.ranking.storage import RankingStorage
from suffering.training.service import TrainingService
from suffering.training.walkforward import generate_walkforward_folds, run_walkforward_baseline


def build_panel_dataset(
    date_count: int = 12,
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "META"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(
        pd.date_range("2024-01-02", periods=date_count, freq="B")
    ):
        for symbol_index, symbol in enumerate(symbols):
            feature_alpha = float(date_index + symbol_index)
            feature_beta = float((date_index + 1) * (symbol_index + 1))
            rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "feature_alpha": feature_alpha,
                    "feature_beta": feature_beta,
                    "future_return_5d": feature_alpha * 0.01 - feature_beta * 0.001,
                    "relevance_5d_5q": symbol_index,
                }
            )
    return pd.DataFrame(rows)


def test_generate_walkforward_folds_keeps_dates_disjoint_and_ordered() -> None:
    frame = build_panel_dataset(date_count=12)

    folds = generate_walkforward_folds(
        frame=frame,
        validation_ratio=0.2,
        test_ratio=0.2,
        step_ratio=0.2,
    )

    assert len(folds) == 4
    for fold in folds:
        train_dates = set(fold.train_frame["date"].dt.strftime("%Y-%m-%d"))
        validation_dates = set(fold.validation_frame["date"].dt.strftime("%Y-%m-%d"))
        test_dates = set(fold.test_frame["date"].dt.strftime("%Y-%m-%d"))

        assert train_dates
        assert validation_dates
        assert test_dates
        assert train_dates.isdisjoint(validation_dates)
        assert train_dates.isdisjoint(test_dates)
        assert validation_dates.isdisjoint(test_dates)
        assert fold.train_frame["date"].max() < fold.validation_frame["date"].min()
        assert fold.validation_frame["date"].max() < fold.test_frame["date"].min()


def test_generate_walkforward_folds_reports_small_sample_requirement_clearly() -> None:
    frame = build_panel_dataset(date_count=4)

    try:
        generate_walkforward_folds(
            frame=frame,
            validation_ratio=0.2,
            test_ratio=0.2,
            step_ratio=0.2,
            min_folds=3,
        )
    except ValueError as exc:
        assert "fewer than the required minimum 3" in str(exc)
    else:
        raise AssertionError("expected insufficient walk-forward folds to raise ValueError")


def test_run_walkforward_baseline_produces_metrics_and_predictions() -> None:
    frame = build_panel_dataset(date_count=12)

    result = run_walkforward_baseline(
        frame=frame,
        validation_ratio=0.2,
        test_ratio=0.2,
        step_ratio=0.2,
        random_state=7,
    )

    assert result.feature_columns == ["feature_alpha", "feature_beta"]
    assert len(result.fold_results) == 4
    assert set(result.test_metrics_summary["mae"]) == {"mean", "std", "min", "max"}
    assert result.test_metrics_summary["mae"]["mean"] is not None
    assert list(result.combined_test_predictions.columns) == [
        "fold_id",
        "date",
        "symbol",
        "y_true",
        "y_pred",
    ]
    assert len(result.combined_test_predictions) == 4 * 2 * 4

    first_fold = result.fold_results[0]
    assert first_fold.test_metrics["mae"] is not None
    assert first_fold.test_metrics["rmse"] is not None


def test_training_service_runs_walkforward_end_to_end_from_cached_dataset(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        default_symbols=["AAPL", "MSFT", "NVDA", "META"],
    )
    dataset_frame = build_panel_dataset(date_count=12)
    ranking_storage = RankingStorage.from_settings(settings)
    ranking_storage.write_daily_dataset(settings.default_dataset_name, dataset_frame)

    service = TrainingService.from_settings(settings)
    summary = service.train_walkforward()

    assert summary["dataset_name"] == settings.default_dataset_name
    assert summary["model_name"] == settings.default_model_name
    assert summary["fold_count"] == 4
    assert Path(summary["artifacts"]["summary_path"]).exists()
    assert Path(summary["artifacts"]["folds_path"]).exists()
    assert Path(summary["artifacts"]["predictions_path"]).exists()

    report = service.read_walkforward_report()
    assert report["fold_count"] == 4
    assert report["test_metrics_summary"]["mae"]["mean"] is not None
