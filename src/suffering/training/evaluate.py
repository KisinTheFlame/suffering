"""Minimal regression and ranking-style evaluation helpers."""

from __future__ import annotations

import math

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN
from suffering.training.baseline import PREDICTION_COLUMN

METRIC_NAMES = (
    "mae",
    "rmse",
    "overall_spearman_corr",
    "daily_rank_ic_mean",
    "daily_rank_ic_std",
    "top_5_mean_future_return",
    "top_10_mean_future_return",
    "ndcg_at_5_mean",
)


def evaluate_predictions(
    frame: pd.DataFrame,
    prediction_column: str = PREDICTION_COLUMN,
    include_error_metrics: bool = True,
) -> dict[str, float | None]:
    if prediction_column not in frame.columns:
        raise ValueError(f"Missing required prediction column: {prediction_column}")

    valid = frame.loc[
        frame[FUTURE_RETURN_5D_COLUMN].notna() & frame[prediction_column].notna()
    ].copy()
    if valid.empty:
        return {name: None for name in METRIC_NAMES}

    daily_rank_ic = _daily_rank_ic_series(valid, prediction_column=prediction_column)
    mae: float | None = None
    rmse: float | None = None
    if include_error_metrics:
        absolute_error = (valid[FUTURE_RETURN_5D_COLUMN] - valid[prediction_column]).abs()
        squared_error = (valid[FUTURE_RETURN_5D_COLUMN] - valid[prediction_column]) ** 2
        mae = float(absolute_error.mean())
        rmse = float(math.sqrt(squared_error.mean()))

    return {
        "mae": mae,
        "rmse": rmse,
        "overall_spearman_corr": _safe_spearman(
            valid[FUTURE_RETURN_5D_COLUMN],
            valid[prediction_column],
        ),
        "daily_rank_ic_mean": float(daily_rank_ic.mean()) if not daily_rank_ic.empty else None,
        "daily_rank_ic_std": float(daily_rank_ic.std(ddof=0)) if not daily_rank_ic.empty else None,
        "top_5_mean_future_return": _top_k_mean_future_return(
            valid,
            prediction_column=prediction_column,
            top_k=5,
        ),
        "top_10_mean_future_return": _top_k_mean_future_return(
            valid,
            prediction_column=prediction_column,
            top_k=10,
        ),
        "ndcg_at_5_mean": _mean_daily_ndcg_at_k(
            valid,
            prediction_column=prediction_column,
            top_k=5,
        ),
    }


def summarize_metric_collection(
    metrics_collection: list[dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for metric_name in METRIC_NAMES:
        values = [
            value
            for item in metrics_collection
            for value in [item.get(metric_name)]
            if value is not None
        ]
        if not values:
            summary[metric_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        series = pd.Series(values, dtype="float64")
        summary[metric_name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return summary


def _daily_rank_ic_series(
    frame: pd.DataFrame,
    prediction_column: str,
) -> pd.Series:
    rank_ic_values: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=True):
        correlation = _safe_spearman(
            group[FUTURE_RETURN_5D_COLUMN],
            group[prediction_column],
        )
        if correlation is not None:
            rank_ic_values.append(correlation)
    return pd.Series(rank_ic_values, dtype="float64")


def _top_k_mean_future_return(
    frame: pd.DataFrame,
    prediction_column: str,
    top_k: int,
) -> float | None:
    daily_means: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=True):
        ranked = group.sort_values(
            [prediction_column, SYMBOL_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
        top_group = ranked.head(top_k)
        if not top_group.empty:
            daily_means.append(float(top_group[FUTURE_RETURN_5D_COLUMN].mean()))
    return float(pd.Series(daily_means, dtype="float64").mean()) if daily_means else None


def _mean_daily_ndcg_at_k(
    frame: pd.DataFrame,
    prediction_column: str,
    top_k: int,
) -> float | None:
    if RELEVANCE_5D_5Q_COLUMN not in frame.columns:
        return None

    ndcg_values: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=True):
        valid_group = group.loc[group[RELEVANCE_5D_5Q_COLUMN].notna()].copy()
        if valid_group.empty:
            continue

        predicted = valid_group.sort_values(
            [prediction_column, SYMBOL_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
        ideal = valid_group.sort_values(
            [RELEVANCE_5D_5Q_COLUMN, SYMBOL_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
        ideal_dcg = _discounted_cumulative_gain(ideal[RELEVANCE_5D_5Q_COLUMN], top_k=top_k)
        if ideal_dcg == 0.0:
            continue

        dcg = _discounted_cumulative_gain(predicted[RELEVANCE_5D_5Q_COLUMN], top_k=top_k)
        ndcg_values.append(dcg / ideal_dcg)

    return float(pd.Series(ndcg_values, dtype="float64").mean()) if ndcg_values else None


def _discounted_cumulative_gain(relevance: pd.Series, top_k: int) -> float:
    gains = relevance.astype("float64").head(top_k).tolist()
    if not gains:
        return 0.0

    total = 0.0
    for rank_index, gain in enumerate(gains, start=1):
        total += (2.0**gain - 1.0) / math.log2(rank_index + 1.0)
    return total


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
