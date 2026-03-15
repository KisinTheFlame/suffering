"""Helpers for the minimal robustness and sensitivity analysis layer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

ROBUSTNESS_TABLE_COLUMNS = [
    "strategy_name",
    "task_type",
    "model_name",
    "top_k",
    "holding_days",
    "cost_bps_per_side",
    "total_return_net",
    "sharpe_ratio_net",
    "max_drawdown_net",
    "annualized_return_net",
    "annualized_volatility",
    "average_daily_turnover",
    "average_active_positions",
    "start_date",
    "end_date",
]

REFERENCE_BENCHMARK_STRATEGY_NAMES = (
    "qqq_buy_and_hold",
    "equal_weight_universe_buy_and_hold",
)


def build_robustness_parameter_grid(
    top_k_values: Iterable[int],
    holding_days_values: Iterable[int],
    cost_bps_values: Iterable[float],
) -> list[tuple[int, int, float]]:
    resolved_top_k_values = _normalize_positive_int_values(top_k_values, "top_k_values")
    resolved_holding_days_values = _normalize_positive_int_values(
        holding_days_values,
        "holding_days_values",
    )
    resolved_cost_bps_values = _normalize_non_negative_float_values(
        cost_bps_values,
        "cost_bps_values",
    )

    return [
        (top_k, holding_days, cost_bps)
        for top_k in resolved_top_k_values
        for holding_days in resolved_holding_days_values
        for cost_bps in resolved_cost_bps_values
    ]


def build_robustness_row(
    *,
    strategy_name: str,
    task_type: str,
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "task_type": task_type,
        "model_name": model_name,
        "top_k": int(top_k) if top_k is not None else None,
        "holding_days": int(holding_days) if holding_days is not None else None,
        "cost_bps_per_side": float(cost_bps_per_side),
        "total_return_net": _safe_float(summary.get("total_return_net")),
        "sharpe_ratio_net": _safe_float(summary.get("sharpe_ratio_net")),
        "max_drawdown_net": _safe_float(summary.get("max_drawdown_net")),
        "annualized_return_net": _safe_float(summary.get("annualized_return_net")),
        "annualized_volatility": _safe_float(summary.get("annualized_volatility")),
        "average_daily_turnover": _safe_float(summary.get("average_daily_turnover")),
        "average_active_positions": _safe_float(summary.get("average_active_positions")),
        "start_date": summary.get("portfolio_date_start"),
        "end_date": summary.get("portfolio_date_end"),
    }


def build_robustness_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=ROBUSTNESS_TABLE_COLUMNS)

    table = pd.DataFrame(rows)
    for column in ROBUSTNESS_TABLE_COLUMNS:
        if column not in table.columns:
            table[column] = None

    return table.loc[:, ROBUSTNESS_TABLE_COLUMNS].sort_values(
        ["sharpe_ratio_net", "total_return_net", "strategy_name", "model_name"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_robustness_summary(
    *,
    model_name: str,
    top_k_values: list[int],
    holding_days_values: list[int],
    cost_bps_values: list[float],
    model_rows: list[dict[str, Any]],
    simple_momentum_rows: list[dict[str, Any]],
    benchmark_reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_model_by_sharpe = _select_best_row(
        model_rows,
        primary_metric="sharpe_ratio_net",
        secondary_metric="total_return_net",
    )
    best_model_by_total_return = _select_best_row(
        model_rows,
        primary_metric="total_return_net",
        secondary_metric="sharpe_ratio_net",
    )
    best_momentum_by_sharpe = _select_best_row(
        simple_momentum_rows,
        primary_metric="sharpe_ratio_net",
        secondary_metric="total_return_net",
    )
    best_momentum_by_total_return = _select_best_row(
        simple_momentum_rows,
        primary_metric="total_return_net",
        secondary_metric="sharpe_ratio_net",
    )

    benchmark_reference: dict[str, dict[str, Any] | None] = {}
    for strategy_name in REFERENCE_BENCHMARK_STRATEGY_NAMES:
        strategy_rows = [
            row for row in benchmark_reference_rows if row["strategy_name"] == strategy_name
        ]
        benchmark_reference[strategy_name] = _snapshot_row(
            _select_best_row(
                strategy_rows,
                primary_metric="sharpe_ratio_net",
                secondary_metric="total_return_net",
            )
        )

    return {
        "model_name": model_name,
        "total_configs_evaluated": len(model_rows),
        "top_k_values": [int(value) for value in top_k_values],
        "holding_days_values": [int(value) for value in holding_days_values],
        "cost_bps_values": [float(value) for value in cost_bps_values],
        "best_config_by_sharpe_net": _snapshot_row(best_model_by_sharpe),
        "best_config_by_total_return_net": _snapshot_row(best_model_by_total_return),
        "benchmark_reference": benchmark_reference,
        "simple_momentum_best_sharpe_net": _snapshot_row(best_momentum_by_sharpe),
        "simple_momentum_best_total_return_net": _snapshot_row(best_momentum_by_total_return),
        "whether_model_beats_simple_momentum_on_best_sharpe": _beats_benchmark(
            challenger=best_model_by_sharpe,
            benchmark=best_momentum_by_sharpe,
            primary_metric="sharpe_ratio_net",
            secondary_metric="total_return_net",
        ),
        "whether_model_beats_simple_momentum_on_best_total_return": _beats_benchmark(
            challenger=best_model_by_total_return,
            benchmark=best_momentum_by_total_return,
            primary_metric="total_return_net",
            secondary_metric="sharpe_ratio_net",
        ),
        "robustness_notes": _build_robustness_notes(model_rows),
    }


def _build_robustness_notes(model_rows: list[dict[str, Any]]) -> list[str]:
    if not model_rows:
        return ["no model configurations were evaluated"]

    frame = pd.DataFrame(model_rows)
    notes: list[str] = []

    sharpe_series = pd.to_numeric(frame["sharpe_ratio_net"], errors="coerce")
    best_sharpe = sharpe_series.max()
    if pd.notna(best_sharpe) and best_sharpe > 0:
        near_best = frame.loc[sharpe_series >= best_sharpe * 0.8].copy()
        if len(near_best) <= 1:
            notes.append("only performs well at one narrow configuration")
        elif near_best["top_k"].nunique(dropna=True) >= 2:
            notes.append("model remains competitive across multiple top_k values")

    cost_summary = (
        frame.groupby("cost_bps_per_side", dropna=False)
        .agg(
            mean_sharpe_ratio_net=("sharpe_ratio_net", "mean"),
            mean_total_return_net=("total_return_net", "mean"),
        )
        .reset_index()
        .sort_values("cost_bps_per_side", kind="stable")
    )
    if len(cost_summary) >= 2:
        low_cost = cost_summary.iloc[0]
        high_cost = cost_summary.iloc[-1]
        low_cost_sharpe = _safe_float(low_cost["mean_sharpe_ratio_net"])
        high_cost_sharpe = _safe_float(high_cost["mean_sharpe_ratio_net"])
        low_cost_return = _safe_float(low_cost["mean_total_return_net"])
        high_cost_return = _safe_float(high_cost["mean_total_return_net"])

        if (
            low_cost_sharpe is not None
            and low_cost_sharpe > 0
            and high_cost_sharpe is not None
            and high_cost_sharpe < low_cost_sharpe * 0.5
        ) or (
            low_cost_return is not None
            and low_cost_return > 0
            and high_cost_return is not None
            and high_cost_return < low_cost_return * 0.5
        ):
            notes.append("performance degrades materially as costs increase")

    if not notes:
        notes.append("performance is directionally consistent across the small default grid")

    return notes


def _snapshot_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {column: row.get(column) for column in ROBUSTNESS_TABLE_COLUMNS}


def _select_best_row(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    secondary_metric: str,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _metric_sort_value(row.get(primary_metric)),
            _metric_sort_value(row.get(secondary_metric)),
            row.get("strategy_name") or "",
            row.get("model_name") or "",
        ),
    )


def _beats_benchmark(
    *,
    challenger: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    primary_metric: str,
    secondary_metric: str,
) -> bool | None:
    if challenger is None or benchmark is None:
        return None

    challenger_primary = _metric_sort_value(challenger.get(primary_metric))
    benchmark_primary = _metric_sort_value(benchmark.get(primary_metric))
    if challenger_primary > benchmark_primary:
        return True
    if challenger_primary < benchmark_primary:
        return False

    return _metric_sort_value(challenger.get(secondary_metric)) > _metric_sort_value(
        benchmark.get(secondary_metric)
    )


def _metric_sort_value(value: Any) -> float:
    resolved = _safe_float(value)
    if resolved is None:
        return float("-inf")
    return resolved


def _normalize_positive_int_values(values: Iterable[int], argument_name: str) -> list[int]:
    resolved_values: list[int] = []
    seen: set[int] = set()
    for raw_value in values:
        value = int(raw_value)
        if value < 1:
            raise ValueError(f"{argument_name} must contain only positive integers")
        if value in seen:
            continue
        seen.add(value)
        resolved_values.append(value)
    if not resolved_values:
        raise ValueError(f"{argument_name} must not be empty")
    return resolved_values


def _normalize_non_negative_float_values(
    values: Iterable[float],
    argument_name: str,
) -> list[float]:
    resolved_values: list[float] = []
    seen: set[float] = set()
    for raw_value in values:
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"{argument_name} must contain only non-negative values")
        if value in seen:
            continue
        seen.add(value)
        resolved_values.append(value)
    if not resolved_values:
        raise ValueError(f"{argument_name} must not be empty")
    return resolved_values


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
