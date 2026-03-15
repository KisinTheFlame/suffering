"""Minimal ranking model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from suffering.config.settings import Settings
from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN
from suffering.training.baseline import select_numeric_feature_columns
from suffering.training.models import build_ranker
from suffering.training.prediction import predict_with_model

SCORE_PREDICTION_COLUMN = "score_pred"


@dataclass(frozen=True)
class RankingQueryGroups:
    group_sizes: list[int]
    qid: pd.Series


@dataclass(frozen=True)
class RankingTrainingResult:
    model: Any
    feature_columns: list[str]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def train_xgb_ranker(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    settings: Settings | None = None,
    random_state: int = 7,
) -> RankingTrainingResult:
    return train_ranker(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        model_name="xgb_ranker",
        feature_columns=feature_columns,
        settings=settings,
        random_state=random_state,
    )


def train_ranker(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_name: str = "xgb_ranker",
    feature_columns: list[str] | None = None,
    settings: Settings | None = None,
    random_state: int = 7,
) -> RankingTrainingResult:
    resolved_feature_columns = feature_columns or select_numeric_feature_columns(train_frame)
    if not resolved_feature_columns:
        raise ValueError("No numeric feature columns available for ranking training")

    prepared_train = prepare_ranking_frame(train_frame)
    prepared_validation = prepare_ranking_frame(validation_frame)
    prepared_test = prepare_ranking_frame(test_frame)
    query_groups = build_date_query_groups(prepared_train)

    model = build_ranker(
        model_name=model_name,
        settings=settings,
        random_state=random_state,
    )
    model.fit(
        prepared_train.loc[:, resolved_feature_columns],
        prepared_train[RELEVANCE_5D_5Q_COLUMN],
        group=query_groups.group_sizes,
    )

    validation_predictions = build_score_prediction_frame(
        frame=prepared_validation,
        model=model,
        feature_columns=resolved_feature_columns,
        model_name=model_name,
    )
    test_predictions = build_score_prediction_frame(
        frame=prepared_test,
        model=model,
        feature_columns=resolved_feature_columns,
        model_name=model_name,
    )

    return RankingTrainingResult(
        model=model,
        feature_columns=resolved_feature_columns,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
    )


def build_date_query_groups(frame: pd.DataFrame) -> RankingQueryGroups:
    prepared = prepare_ranking_frame(frame)
    qid = pd.Series(index=prepared.index, dtype="int64")
    group_sizes: list[int] = []

    for query_id, (_, group) in enumerate(prepared.groupby(DATE_COLUMN, sort=False)):
        group_sizes.append(int(len(group)))
        qid.loc[group.index] = query_id

    return RankingQueryGroups(
        group_sizes=group_sizes,
        qid=qid.astype("int64"),
    )


def prepare_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        DATE_COLUMN,
        SYMBOL_COLUMN,
        FUTURE_RETURN_5D_COLUMN,
        RELEVANCE_5D_5Q_COLUMN,
    }
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for ranking training: {missing_text}")

    prepared = frame.copy()
    prepared[DATE_COLUMN] = (
        pd.to_datetime(prepared[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    )
    prepared[SYMBOL_COLUMN] = prepared[SYMBOL_COLUMN].astype(str).str.strip().str.upper()
    return prepared.sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable").reset_index(drop=True)


def build_score_prediction_frame(
    frame: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    prediction_frame = frame.loc[
        :,
        [
            DATE_COLUMN,
            SYMBOL_COLUMN,
            FUTURE_RETURN_5D_COLUMN,
            RELEVANCE_5D_5Q_COLUMN,
        ],
    ].copy()
    prediction_frame[SCORE_PREDICTION_COLUMN] = predict_with_model(
        model=model,
        feature_frame=frame.loc[:, feature_columns],
    )
    prediction_frame["model_name"] = model_name
    return prediction_frame
