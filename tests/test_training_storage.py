import pickle
from pathlib import Path

import pandas as pd

from suffering.training.storage import TrainingStorage


def test_training_storage_writes_and_reads_artifacts(tmp_path: Path) -> None:
    storage = TrainingStorage(artifacts_dir=tmp_path)
    model_payload = {"name": "hist_gbr"}
    metrics_report = {"model_name": "hist_gbr", "mae": 0.1}
    walkforward_report = {"model_name": "hist_gbr", "fold_count": 3}
    prediction_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "future_return_5d": [0.03],
            "predicted_future_return_5d": [0.02],
        }
    )
    walkforward_fold_frame = pd.DataFrame(
        {
            "fold_id": [1],
            "train_date_start": ["2024-01-02"],
            "test_date_end": ["2024-01-05"],
            "mae": [0.1],
        }
    )
    walkforward_prediction_frame = pd.DataFrame(
        {
            "fold_id": [1],
            "date": pd.to_datetime(["2024-01-05"]),
            "symbol": ["AAPL"],
            "y_true": [0.03],
            "y_pred": [0.02],
        }
    )

    model_path = storage.write_model("hist_gbr", model_payload)
    report_path = storage.write_metrics_report("hist_gbr", metrics_report)
    prediction_path = storage.write_predictions("hist_gbr", "test", prediction_frame)
    walkforward_summary_path = storage.write_walkforward_summary(
        "hist_gbr",
        walkforward_report,
    )
    walkforward_folds_path = storage.write_walkforward_folds(
        "hist_gbr",
        walkforward_fold_frame,
    )
    walkforward_predictions_path = storage.write_walkforward_predictions(
        "hist_gbr",
        walkforward_prediction_frame,
    )

    assert model_path == tmp_path / "models" / "hist_gbr.pkl"
    assert report_path == tmp_path / "reports" / "hist_gbr_metrics.json"
    assert prediction_path == tmp_path / "predictions" / "hist_gbr_test.csv"
    assert walkforward_summary_path == tmp_path / "reports" / "hist_gbr_walkforward_summary.json"
    assert walkforward_folds_path == tmp_path / "reports" / "hist_gbr_walkforward_folds.csv"
    assert (
        walkforward_predictions_path
        == tmp_path / "predictions" / "hist_gbr_walkforward_test_predictions.csv"
    )
    assert storage.read_metrics_report("hist_gbr") == metrics_report
    assert storage.read_walkforward_summary("hist_gbr") == walkforward_report
    assert pd.read_csv(prediction_path).shape[0] == 1
    assert pd.read_csv(walkforward_folds_path).shape[0] == 1
    assert pd.read_csv(walkforward_predictions_path).shape[0] == 1

    with model_path.open("rb") as file:
        assert pickle.load(file) == model_payload

    assert storage.model_path("hist_gbr") != storage.model_path("xgb_regressor")
    assert storage.metrics_report_path("hist_gbr") != storage.metrics_report_path("xgb_regressor")
    assert storage.walkforward_summary_path("hist_gbr") != storage.walkforward_summary_path(
        "xgb_regressor"
    )
    assert storage.model_path("xgb_regressor") != storage.model_path("xgb_ranker")
    assert storage.prediction_path("xgb_regressor", "test") != storage.prediction_path(
        "xgb_ranker",
        "test",
    )
