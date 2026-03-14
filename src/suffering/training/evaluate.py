"""Minimal regression and ranking-style evaluation helpers."""

from __future__ import annotations

import math

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.baseline import PREDICTION_COLUMN


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, float | None]:
    valid = frame.loc[
        frame[FUTURE_RETURN_5D_COLUMN].notna() & frame[PREDICTION_COLUMN].notna()
    ].copy()
    if valid.empty:
        return {
            "mae": None,
            "rmse": None,
            "overall_spearman_corr": None,
            "daily_rank_ic_mean": None,
            "daily_rank_ic_std": None,
            "top_5_mean_future_return": None,
            "top_10_mean_future_return": None,
        }

    absolute_error = (valid[FUTURE_RETURN_5D_COLUMN] - valid[PREDICTION_COLUMN]).abs()
    squared_error = (valid[FUTURE_RETURN_5D_COLUMN] - valid[PREDICTION_COLUMN]) ** 2
    daily_rank_ic = _daily_rank_ic_series(valid)

    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(math.sqrt(squared_error.mean())),
        "overall_spearman_corr": _safe_spearman(
            valid[FUTURE_RETURN_5D_COLUMN],
            valid[PREDICTION_COLUMN],
        ),
        "daily_rank_ic_mean": float(daily_rank_ic.mean()) if not daily_rank_ic.empty else None,
        "daily_rank_ic_std": float(daily_rank_ic.std(ddof=0)) if not daily_rank_ic.empty else None,
        "top_5_mean_future_return": _top_k_mean_future_return(valid, top_k=5),
        "top_10_mean_future_return": _top_k_mean_future_return(valid, top_k=10),
    }


def _daily_rank_ic_series(frame: pd.DataFrame) -> pd.Series:
    rank_ic_values: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=True):
        correlation = _safe_spearman(
            group[FUTURE_RETURN_5D_COLUMN],
            group[PREDICTION_COLUMN],
        )
        if correlation is not None:
            rank_ic_values.append(correlation)
    return pd.Series(rank_ic_values, dtype="float64")


def _top_k_mean_future_return(frame: pd.DataFrame, top_k: int) -> float | None:
    daily_means: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=True):
        ranked = group.sort_values(
            [PREDICTION_COLUMN, SYMBOL_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
        top_group = ranked.head(top_k)
        if not top_group.empty:
            daily_means.append(float(top_group[FUTURE_RETURN_5D_COLUMN].mean()))
    return float(pd.Series(daily_means, dtype="float64").mean()) if daily_means else None


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float | None:
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if len(aligned) < 2:
        return None

    actual_rank = aligned["actual"].rank(method="average")
    predicted_rank = aligned["predicted"].rank(method="average")
    if actual_rank.nunique() < 2 or predicted_rank.nunique() < 2:
        return None

    correlation = actual_rank.corr(predicted_rank)
    return float(correlation) if pd.notna(correlation) else None
