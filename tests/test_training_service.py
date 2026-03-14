from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings
from suffering.ranking.storage import RankingStorage
from suffering.training.service import TrainingService


def build_panel_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(pd.date_range("2024-01-02", periods=15, freq="B")):
        for symbol_index, symbol in enumerate(("AAPL", "MSFT", "NVDA", "META")):
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


def test_training_service_runs_end_to_end_from_cached_dataset(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        default_symbols=["AAPL", "MSFT", "NVDA", "META"],
    )
    dataset_frame = build_panel_dataset()
    ranking_storage = RankingStorage.from_settings(settings)
    ranking_storage.write_daily_dataset(settings.default_dataset_name, dataset_frame)

    service = TrainingService.from_settings(settings)
    summary = service.train_baseline()

    assert summary["dataset_name"] == settings.default_dataset_name
    assert summary["model_name"] == settings.default_model_name
    assert summary["feature_count"] == 2
    assert Path(summary["artifacts"]["model_path"]).exists()
    assert Path(summary["artifacts"]["metrics_path"]).exists()
    assert Path(summary["artifacts"]["validation_predictions_path"]).exists()
    assert Path(summary["artifacts"]["test_predictions_path"]).exists()

    report = service.read_training_report()
    assert report["model_name"] == settings.default_model_name
    assert report["test_metrics"]["mae"] is not None
