"""Walk-forward prediction loading and signal normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.ranking import SCORE_PREDICTION_COLUMN
from suffering.training.storage import TrainingStorage

FOLD_ID_COLUMN = "fold_id"
MODEL_NAME_COLUMN = "model_name"
SIGNAL_SCORE_COLUMN = "signal_score"
REGRESSION_PREDICTION_COLUMN = "y_pred"
REGRESSION_TARGET_COLUMN = "y_true"

NORMALIZED_SIGNAL_COLUMNS = [
    FOLD_ID_COLUMN,
    DATE_COLUMN,
    SYMBOL_COLUMN,
    SIGNAL_SCORE_COLUMN,
    FUTURE_RETURN_5D_COLUMN,
    MODEL_NAME_COLUMN,
]


def load_walkforward_test_signals(
    model_name: str,
    training_storage: TrainingStorage,
) -> pd.DataFrame:
    path = training_storage.walkforward_predictions_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward test predictions not found for model: {model_name}")

    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    return normalize_walkforward_predictions(frame=frame, model_name=model_name, source_path=path)


def normalize_walkforward_predictions(
    frame: pd.DataFrame,
    model_name: str,
    source_path: Path | None = None,
) -> pd.DataFrame:
    required_columns = {DATE_COLUMN, SYMBOL_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Walk-forward predictions are missing required columns: {missing_display}"
        )

    score_column = _resolve_signal_score_column(frame)
    future_return_column = _resolve_future_return_column(frame)

    normalized = frame.copy()
    normalized[DATE_COLUMN] = pd.to_datetime(normalized[DATE_COLUMN]).dt.tz_localize(None)
    if FOLD_ID_COLUMN not in normalized.columns:
        normalized[FOLD_ID_COLUMN] = 1

    if MODEL_NAME_COLUMN in normalized.columns:
        normalized[MODEL_NAME_COLUMN] = normalized[MODEL_NAME_COLUMN].fillna(model_name)
    else:
        normalized[MODEL_NAME_COLUMN] = model_name

    normalized[SIGNAL_SCORE_COLUMN] = normalized[score_column]
    normalized[FUTURE_RETURN_5D_COLUMN] = normalized[future_return_column]
    normalized["_row_order"] = range(len(normalized))

    output = (
        normalized.loc[:, NORMALIZED_SIGNAL_COLUMNS + ["_row_order"]]
        .sort_values(
            [DATE_COLUMN, FOLD_ID_COLUMN, SIGNAL_SCORE_COLUMN, SYMBOL_COLUMN, "_row_order"],
            ascending=[True, True, False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    output = output.loc[:, NORMALIZED_SIGNAL_COLUMNS]
    if source_path is not None:
        output.attrs["source_path"] = str(source_path)
    return output


def _resolve_signal_score_column(frame: pd.DataFrame) -> str:
    if SCORE_PREDICTION_COLUMN in frame.columns:
        return SCORE_PREDICTION_COLUMN
    if REGRESSION_PREDICTION_COLUMN in frame.columns:
        return REGRESSION_PREDICTION_COLUMN
    raise ValueError(
        "Walk-forward predictions must contain either `score_pred` or `y_pred` as the signal score"
    )


def _resolve_future_return_column(frame: pd.DataFrame) -> str:
    if FUTURE_RETURN_5D_COLUMN in frame.columns:
        return FUTURE_RETURN_5D_COLUMN
    if REGRESSION_TARGET_COLUMN in frame.columns:
        return REGRESSION_TARGET_COLUMN
    raise ValueError(
        "Walk-forward predictions must contain either `future_return_5d` or `y_true`"
    )
