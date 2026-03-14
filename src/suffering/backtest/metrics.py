"""Portfolio-level summary metrics for the minimal backtest layer."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from suffering.data.models import DATE_COLUMN

TRADING_DAYS_PER_YEAR = 252


def compute_backtest_metrics(
    daily_returns: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    if daily_returns.empty:
        raise ValueError("daily_returns must not be empty")

    gross_returns = daily_returns["gross_return"]
    net_returns = daily_returns["net_return"]
    gross_equity = daily_returns["gross_equity"]
    net_equity = daily_returns["net_equity"]
    period_count = len(daily_returns)

    metrics: dict[str, Any] = {
        "portfolio_date_start": _format_date(daily_returns[DATE_COLUMN].min()),
        "portfolio_date_end": _format_date(daily_returns[DATE_COLUMN].max()),
        "daily_observation_count": int(period_count),
        "trade_count": int(len(trades)),
        "total_return_gross": float(gross_equity.iloc[-1] - 1.0),
        "total_return_net": float(net_equity.iloc[-1] - 1.0),
        "annualized_return_gross": _annualized_return(gross_equity.iloc[-1], period_count),
        "annualized_return_net": _annualized_return(net_equity.iloc[-1], period_count),
        "annualized_volatility": _annualized_volatility(net_returns),
        "annualized_volatility_gross": _annualized_volatility(gross_returns),
        "annualized_volatility_net": _annualized_volatility(net_returns),
        "sharpe_ratio_gross": _sharpe_ratio(gross_returns),
        "sharpe_ratio_net": _sharpe_ratio(net_returns),
        "max_drawdown_gross": _max_drawdown(gross_equity),
        "max_drawdown_net": _max_drawdown(net_equity),
        "daily_hit_rate_gross": float((gross_returns > 0).mean()),
        "daily_hit_rate_net": float((net_returns > 0).mean()),
        "average_daily_turnover": float(daily_returns["turnover"].mean()),
        "average_active_positions": float(daily_returns["active_positions"].mean()),
        "average_active_cohorts": float(daily_returns["active_cohorts"].mean()),
    }
    return metrics


def _annualized_return(final_equity: float, period_count: int) -> float:
    if period_count < 1 or final_equity <= 0:
        return 0.0
    return float(final_equity ** (TRADING_DAYS_PER_YEAR / period_count) - 1.0)


def _annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    volatility = daily_returns.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return 0.0
    return float((daily_returns.mean() / volatility) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def _format_date(value: pd.Timestamp | Any) -> str | None:
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%Y-%m-%d")
