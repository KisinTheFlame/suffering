"""Baseline regression model training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN

PREDICTION_COLUMN = "predicted_future_return_5d"
EXCLUDED_FEATURE_COLUMNS = {
    DATE_COLUMN,
    SYMBOL_COLUMN,
    FUTURE_RETURN_5D_COLUMN,
    RELEVANCE_5D_5Q_COLUMN,
}


@dataclass(frozen=True)
class BaselineTrainingResult:
    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def train_hist_gradient_boosting_baseline(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 7,
) -> BaselineTrainingResult:
    resolved_feature_columns = feature_columns or select_numeric_feature_columns(train_frame)
    if not resolved_feature_columns:
        raise ValueError("No numeric feature columns available for baseline training")

    model = HistGradientBoostingRegressor(random_state=random_state)
    model.fit(train_frame.loc[:, resolved_feature_columns], train_frame[FUTURE_RETURN_5D_COLUMN])

    validation_predictions = build_prediction_frame(
        frame=validation_frame,
        model=model,
        feature_columns=resolved_feature_columns,
    )
    test_predictions = build_prediction_frame(
        frame=test_frame,
        model=model,
        feature_columns=resolved_feature_columns,
    )
    return BaselineTrainingResult(
        model=model,
        feature_columns=resolved_feature_columns,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
    )


def select_numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column not in EXCLUDED_FEATURE_COLUMNS]


def build_prediction_frame(
    frame: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    feature_columns: list[str],
) -> pd.DataFrame:
    prediction_frame = frame.loc[:, [DATE_COLUMN, SYMBOL_COLUMN, FUTURE_RETURN_5D_COLUMN]].copy()
    prediction_frame[PREDICTION_COLUMN] = model.predict(frame.loc[:, feature_columns])
    return prediction_frame
