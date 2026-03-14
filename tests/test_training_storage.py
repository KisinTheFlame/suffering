import pickle
from pathlib import Path

import pandas as pd

from suffering.training.storage import TrainingStorage


def test_training_storage_writes_and_reads_artifacts(tmp_path: Path) -> None:
    storage = TrainingStorage(artifacts_dir=tmp_path)
    model_payload = {"name": "baseline_hist_gbr"}
    metrics_report = {"model_name": "baseline_hist_gbr", "mae": 0.1}
    prediction_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "future_return_5d": [0.03],
            "predicted_future_return_5d": [0.02],
        }
    )

    model_path = storage.write_model("baseline_hist_gbr", model_payload)
    report_path = storage.write_metrics_report("baseline_hist_gbr", metrics_report)
    prediction_path = storage.write_predictions("baseline_hist_gbr", "test", prediction_frame)

    assert model_path == tmp_path / "models" / "baseline_hist_gbr.pkl"
    assert report_path == tmp_path / "reports" / "baseline_hist_gbr_metrics.json"
    assert prediction_path == tmp_path / "predictions" / "baseline_hist_gbr_test.csv"
    assert storage.read_metrics_report("baseline_hist_gbr") == metrics_report
    assert pd.read_csv(prediction_path).shape[0] == 1

    with model_path.open("rb") as file:
        assert pickle.load(file) == model_payload
