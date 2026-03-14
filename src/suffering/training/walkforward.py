"""Minimal walk-forward validation helpers for the baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.baseline import (
    PREDICTION_COLUMN,
    select_numeric_feature_columns,
    train_hist_gradient_boosting_baseline,
)
from suffering.training.evaluate import evaluate_predictions, summarize_metric_collection
from suffering.training.splits import build_frame_date_summary


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold_id: int
    split_summary: dict[str, dict[str, int | str | None]]
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    test_predictions: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardTrainingResult:
    feature_columns: list[str]
    fold_results: list[WalkForwardFoldResult]
    test_metrics_summary: dict[str, dict[str, float | None]]
    combined_test_predictions: pd.DataFrame


def generate_walkforward_folds(
    frame: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
    step_ratio: float,
    min_folds: int = 1,
) -> list[WalkForwardFold]:
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")

    normalized = frame.copy()
    normalized[DATE_COLUMN] = (
        pd.to_datetime(normalized[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    )
    unique_dates = pd.Index(sorted(normalized[DATE_COLUMN].dropna().unique()))
    if len(unique_dates) < 3:
        raise ValueError("At least 3 unique dates are required for walk-forward validation")

    validation_date_count = max(1, int(len(unique_dates) * validation_ratio))
    test_date_count = max(1, int(len(unique_dates) * test_ratio))
    step_date_count = max(1, int(len(unique_dates) * step_ratio))

    max_train_end = len(unique_dates) - validation_date_count - test_date_count
    if max_train_end < 1:
        raise ValueError(
            "Not enough unique dates to form walk-forward train/validation/test windows"
        )

    fold_count = 1 + max(0, (max_train_end - 1) // step_date_count)
    initial_train_end = max_train_end - step_date_count * (fold_count - 1)

    folds: list[WalkForwardFold] = []
    for fold_index in range(fold_count):
        train_end = initial_train_end + fold_index * step_date_count
        validation_end = train_end + validation_date_count
        test_end = validation_end + test_date_count

        train_dates = unique_dates[:train_end]
        validation_dates = unique_dates[train_end:validation_end]
        test_dates = unique_dates[validation_end:test_end]
        _validate_ordered_dates(
            train_dates=train_dates,
            validation_dates=validation_dates,
            test_dates=test_dates,
        )
        folds.append(
            WalkForwardFold(
                fold_id=fold_index + 1,
                train_frame=_slice_frame_by_dates(normalized, train_dates),
                validation_frame=_slice_frame_by_dates(normalized, validation_dates),
                test_frame=_slice_frame_by_dates(normalized, test_dates),
            )
        )

    if len(folds) < min_folds:
        raise ValueError(
            "Walk-forward validation produced "
            f"{len(folds)} fold(s), which is fewer than the required minimum {min_folds}"
        )
    return folds


def run_walkforward_baseline(
    frame: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
    step_ratio: float,
    random_state: int = 7,
    min_folds: int = 1,
) -> WalkForwardTrainingResult:
    feature_columns = select_numeric_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for walk-forward baseline training")

    folds = generate_walkforward_folds(
        frame=frame,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        step_ratio=step_ratio,
        min_folds=min_folds,
    )

    fold_results: list[WalkForwardFoldResult] = []
    combined_predictions: list[pd.DataFrame] = []
    for fold in folds:
        training_result = train_hist_gradient_boosting_baseline(
            train_frame=fold.train_frame,
            validation_frame=fold.validation_frame,
            test_frame=fold.test_frame,
            feature_columns=feature_columns,
            random_state=random_state,
        )

        validation_metrics = evaluate_predictions(training_result.validation_predictions)
        test_metrics = evaluate_predictions(training_result.test_predictions)
        fold_predictions = (
            training_result.test_predictions.rename(
                columns={
                    FUTURE_RETURN_5D_COLUMN: "y_true",
                    PREDICTION_COLUMN: "y_pred",
                }
            )
            .assign(fold_id=fold.fold_id)
            .loc[:, ["fold_id", DATE_COLUMN, SYMBOL_COLUMN, "y_true", "y_pred"]]
        )

        fold_results.append(
            WalkForwardFoldResult(
                fold_id=fold.fold_id,
                split_summary={
                    "train": build_frame_date_summary(fold.train_frame),
                    "validation": build_frame_date_summary(fold.validation_frame),
                    "test": build_frame_date_summary(fold.test_frame),
                },
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                test_predictions=fold_predictions,
            )
        )
        combined_predictions.append(fold_predictions)

    combined_test_predictions = (
        pd.concat(combined_predictions, ignore_index=True)
        .sort_values(["fold_id", DATE_COLUMN, SYMBOL_COLUMN], kind="stable")
        .reset_index(drop=True)
    )

    return WalkForwardTrainingResult(
        feature_columns=feature_columns,
        fold_results=fold_results,
        test_metrics_summary=summarize_metric_collection(
            [result.test_metrics for result in fold_results]
        ),
        combined_test_predictions=combined_test_predictions,
    )


def _validate_ordered_dates(
    train_dates: pd.Index,
    validation_dates: pd.Index,
    test_dates: pd.Index,
) -> None:
    if len(train_dates) == 0 or len(validation_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Each walk-forward fold must contain train/validation/test dates")
    if len(train_dates.intersection(validation_dates)) > 0:
        raise ValueError("Walk-forward train and validation dates overlap")
    if len(train_dates.intersection(test_dates)) > 0:
        raise ValueError("Walk-forward train and test dates overlap")
    if len(validation_dates.intersection(test_dates)) > 0:
        raise ValueError("Walk-forward validation and test dates overlap")
    if not (train_dates.max() < validation_dates.min() < test_dates.min()):
        raise ValueError("Walk-forward dates must be strictly increasing")


def _slice_frame_by_dates(frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    return (
        frame.loc[frame[DATE_COLUMN].isin(dates)]
        .sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable")
        .reset_index(drop=True)
    )
