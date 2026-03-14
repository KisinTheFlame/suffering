import pandas as pd

from suffering.training.baseline import PREDICTION_COLUMN
from suffering.training.evaluate import evaluate_predictions


def test_evaluate_predictions_returns_expected_metric_fields() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "NVDA", "AAPL", "MSFT", "NVDA"],
            "future_return_5d": [0.05, 0.01, -0.02, 0.03, 0.02, -0.01],
            PREDICTION_COLUMN: [0.04, 0.02, -0.01, 0.025, 0.015, -0.02],
        }
    )

    metrics = evaluate_predictions(frame)

    assert set(metrics) == {
        "mae",
        "rmse",
        "overall_spearman_corr",
        "daily_rank_ic_mean",
        "daily_rank_ic_std",
        "top_5_mean_future_return",
        "top_10_mean_future_return",
    }
    assert metrics["mae"] is not None
    assert metrics["rmse"] is not None
    assert metrics["daily_rank_ic_mean"] is not None
    assert metrics["top_5_mean_future_return"] is not None
    assert metrics["top_10_mean_future_return"] is not None


def test_evaluate_predictions_is_robust_on_tiny_daily_cross_sections() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "future_return_5d": [0.01, 0.02],
            PREDICTION_COLUMN: [0.015, 0.018],
        }
    )

    metrics = evaluate_predictions(frame)

    assert metrics["mae"] is not None
    assert metrics["rmse"] is not None
    assert metrics["daily_rank_ic_mean"] is None
    assert metrics["daily_rank_ic_std"] is None
