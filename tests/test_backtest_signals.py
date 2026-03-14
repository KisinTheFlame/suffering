import pandas as pd

from suffering.backtest.signals import normalize_walkforward_predictions


def test_normalize_walkforward_predictions_maps_regression_columns() -> None:
    frame = pd.DataFrame(
        {
            "fold_id": [1, 1],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["MSFT", "AAPL"],
            "y_true": [0.02, 0.03],
            "y_pred": [0.4, 0.8],
        }
    )

    normalized = normalize_walkforward_predictions(frame=frame, model_name="hist_gbr")

    assert list(normalized.columns) == [
        "fold_id",
        "date",
        "symbol",
        "signal_score",
        "future_return_5d",
        "model_name",
    ]
    assert normalized["symbol"].tolist() == ["AAPL", "MSFT"]
    assert normalized["signal_score"].tolist() == [0.8, 0.4]
    assert normalized["future_return_5d"].tolist() == [0.03, 0.02]
    assert normalized["model_name"].tolist() == ["hist_gbr", "hist_gbr"]


def test_normalize_walkforward_predictions_maps_ranking_columns() -> None:
    frame = pd.DataFrame(
        {
            "fold_id": [2, 2],
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "symbol": ["NVDA", "AAPL"],
            "future_return_5d": [0.01, 0.05],
            "score_pred": [0.3, 0.9],
            "model_name": ["xgb_ranker", "xgb_ranker"],
        }
    )

    normalized = normalize_walkforward_predictions(frame=frame, model_name="xgb_ranker")

    assert normalized["symbol"].tolist() == ["AAPL", "NVDA"]
    assert normalized["signal_score"].tolist() == [0.9, 0.3]
    assert normalized["future_return_5d"].tolist() == [0.05, 0.01]
    assert normalized["model_name"].tolist() == ["xgb_ranker", "xgb_ranker"]
