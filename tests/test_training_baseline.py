import pandas as pd
import pytest

from suffering.config.settings import Settings
from suffering.training.baseline import (
    PREDICTION_COLUMN,
    select_numeric_feature_columns,
    train_baseline_regressor,
    train_hist_gradient_boosting_baseline,
)
from suffering.training.splits import split_panel_dataset_by_date


def build_training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(pd.date_range("2024-01-02", periods=20, freq="B")):
        for symbol_index, symbol in enumerate(("AAPL", "MSFT", "NVDA")):
            feature_alpha = float(date_index + symbol_index)
            feature_beta = float(date_index * 2 - symbol_index)
            rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "feature_alpha": feature_alpha,
                    "feature_beta": feature_beta,
                    "sector_name": "tech",
                    "future_return_5d": feature_alpha * 0.01 + feature_beta * 0.001,
                    "relevance_5d_5q": symbol_index,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_training_uses_only_numeric_feature_columns_and_predicts() -> None:
    frame = build_training_frame()
    split = split_panel_dataset_by_date(
        frame=frame,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    feature_columns = select_numeric_feature_columns(frame)
    result = train_hist_gradient_boosting_baseline(
        train_frame=split.train_frame,
        validation_frame=split.validation_frame,
        test_frame=split.test_frame,
        feature_columns=feature_columns,
        random_state=7,
    )

    assert feature_columns == ["feature_alpha", "feature_beta"]
    assert result.feature_columns == ["feature_alpha", "feature_beta"]
    assert len(result.validation_predictions) == len(split.validation_frame)
    assert len(result.test_predictions) == len(split.test_frame)
    assert PREDICTION_COLUMN in result.validation_predictions.columns
    assert PREDICTION_COLUMN in result.test_predictions.columns


@pytest.mark.parametrize("model_name", ["hist_gbr", "xgb_regressor"])
def test_train_baseline_regressor_supports_multiple_models(model_name: str) -> None:
    frame = build_training_frame()
    split = split_panel_dataset_by_date(
        frame=frame,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    result = train_baseline_regressor(
        train_frame=split.train_frame,
        validation_frame=split.validation_frame,
        test_frame=split.test_frame,
        model_name=model_name,
        settings=Settings(xgb_n_estimators=12),
        random_state=7,
    )

    assert result.feature_columns == ["feature_alpha", "feature_beta"]
    assert len(result.validation_predictions) == len(split.validation_frame)
    assert len(result.test_predictions) == len(split.test_frame)
    assert PREDICTION_COLUMN in result.validation_predictions.columns
    assert PREDICTION_COLUMN in result.test_predictions.columns
