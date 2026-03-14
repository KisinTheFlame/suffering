"""Minimal top-k cohort construction and overlapping portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from suffering.backtest.signals import FOLD_ID_COLUMN, SIGNAL_SCORE_COLUMN
from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN


@dataclass(frozen=True)
class PortfolioBacktestResult:
    cohorts: pd.DataFrame
    trades: pd.DataFrame
    skipped_trades: pd.DataFrame
    daily_returns: pd.DataFrame
    equity_curve: pd.DataFrame


def build_top_k_cohorts(
    signals: pd.DataFrame,
    top_k: int,
    holding_days: int,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    if signals.empty:
        raise ValueError("Walk-forward test predictions are empty")

    ordered = signals.copy()
    ordered[DATE_COLUMN] = pd.to_datetime(ordered[DATE_COLUMN]).dt.tz_localize(None)
    ordered = ordered.sort_values(
        [DATE_COLUMN, FOLD_ID_COLUMN, SIGNAL_SCORE_COLUMN, SYMBOL_COLUMN],
        ascending=[True, True, False, True],
        kind="stable",
    )

    cohorts = ordered.groupby(DATE_COLUMN, sort=True).head(top_k).copy()
    cohorts["signal_date"] = cohorts[DATE_COLUMN]
    cohorts["position_rank"] = cohorts.groupby("signal_date").cumcount() + 1
    cohorts["selected_count"] = cohorts.groupby("signal_date")[SYMBOL_COLUMN].transform("size")
    cohorts["cohort_weight"] = 1.0 / float(holding_days)
    cohorts["position_weight"] = 1.0 / cohorts["selected_count"]
    cohorts["portfolio_weight"] = cohorts["cohort_weight"] * cohorts["position_weight"]

    return cohorts.loc[
        :,
        [
            FOLD_ID_COLUMN,
            "signal_date",
            SYMBOL_COLUMN,
            SIGNAL_SCORE_COLUMN,
            FUTURE_RETURN_5D_COLUMN,
            "position_rank",
            "selected_count",
            "cohort_weight",
            "position_weight",
            "portfolio_weight",
        ],
    ].reset_index(drop=True)


def simulate_overlapping_portfolio(
    cohorts: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    holding_days: int,
    cost_bps_per_side: float,
) -> PortfolioBacktestResult:
    if cohorts.empty:
        raise ValueError("No cohorts available for portfolio simulation")

    cost_rate = float(cost_bps_per_side) / 10_000.0
    trade_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    daily_trade_frames: list[pd.DataFrame] = []

    trade_id = 0
    for signal_date, cohort_frame in cohorts.groupby("signal_date", sort=True):
        cohort_paths: list[tuple[pd.Series, dict[str, Any], pd.DataFrame]] = []
        for row in cohort_frame.itertuples(index=False):
            price_frame = price_frames.get(row.symbol)
            if price_frame is None:
                skipped_records.append(
                    {
                        "signal_date": signal_date,
                        "symbol": row.symbol,
                        "reason": "price_frame_missing",
                    }
                )
                continue

            trade_path = _build_trade_path(
                signal_row=pd.Series(row._asdict()),
                price_frame=price_frame,
                holding_days=holding_days,
                cost_rate=cost_rate,
            )
            if trade_path is None:
                skipped_records.append(
                    {
                        "signal_date": signal_date,
                        "symbol": row.symbol,
                        "reason": "insufficient_price_history_for_trade",
                    }
                )
                continue
            cohort_paths.append(trade_path)

        if not cohort_paths:
            continue

        valid_trade_count = len(cohort_paths)
        cohort_weight = 1.0 / float(holding_days)
        position_weight = 1.0 / float(valid_trade_count)
        portfolio_weight = cohort_weight * position_weight

        for signal_row, trade_record, trade_daily_frame in cohort_paths:
            trade_id += 1
            enriched_trade = {
                **trade_record,
                "trade_id": trade_id,
                "cohort_trade_count": valid_trade_count,
                "cohort_weight": cohort_weight,
                "position_weight": position_weight,
                "portfolio_weight": portfolio_weight,
                "cost_bps_per_side": float(cost_bps_per_side),
            }
            trade_records.append(enriched_trade)

            daily_frame = trade_daily_frame.copy()
            daily_frame["trade_id"] = trade_id
            daily_frame["signal_date"] = signal_row["signal_date"]
            daily_frame[SYMBOL_COLUMN] = signal_row[SYMBOL_COLUMN]
            daily_frame["portfolio_weight"] = portfolio_weight
            daily_frame["gross_component"] = (
                daily_frame["trade_gross_return"] * daily_frame["portfolio_weight"]
            )
            daily_frame["net_component"] = (
                daily_frame["trade_net_return"] * daily_frame["portfolio_weight"]
            )
            daily_frame["entry_turnover"] = 0.0
            daily_frame["exit_turnover"] = 0.0
            daily_frame.loc[daily_frame.index[0], "entry_turnover"] = portfolio_weight
            daily_frame.loc[daily_frame.index[-1], "exit_turnover"] = portfolio_weight
            daily_trade_frames.append(daily_frame)

    trades = pd.DataFrame(trade_records)
    skipped_trades = pd.DataFrame(
        skipped_records,
        columns=["signal_date", SYMBOL_COLUMN, "reason"],
    )
    if trades.empty:
        raise ValueError(
            "No valid trades could be built from the walk-forward test predictions and raw prices"
        )

    trade_daily = pd.concat(daily_trade_frames, ignore_index=True)
    trade_daily = trade_daily.sort_values([DATE_COLUMN, "trade_id"], kind="stable")

    daily_returns = (
        trade_daily.groupby(DATE_COLUMN, sort=True)
        .agg(
            gross_return=("gross_component", "sum"),
            net_return=("net_component", "sum"),
            turnover=("entry_turnover", "sum"),
            exit_turnover=("exit_turnover", "sum"),
            active_positions=("trade_id", "nunique"),
            active_cohorts=("signal_date", "nunique"),
        )
        .reset_index()
    )
    daily_returns["turnover"] = daily_returns["turnover"] + daily_returns["exit_turnover"]
    daily_returns = daily_returns.drop(columns=["exit_turnover"])
    daily_returns["gross_equity"] = (1.0 + daily_returns["gross_return"]).cumprod()
    daily_returns["net_equity"] = (1.0 + daily_returns["net_return"]).cumprod()

    equity_curve = daily_returns.loc[:, [DATE_COLUMN, "gross_equity", "net_equity"]].copy()
    trades = trades.sort_values(["signal_date", "trade_id"], kind="stable").reset_index(drop=True)
    if not skipped_trades.empty:
        skipped_trades = skipped_trades.sort_values(
            ["signal_date", SYMBOL_COLUMN],
            kind="stable",
        ).reset_index(drop=True)
    daily_returns = daily_returns.reset_index(drop=True)
    equity_curve = equity_curve.reset_index(drop=True)

    return PortfolioBacktestResult(
        cohorts=cohorts.reset_index(drop=True),
        trades=trades,
        skipped_trades=skipped_trades,
        daily_returns=daily_returns,
        equity_curve=equity_curve,
    )


def _build_trade_path(
    signal_row: pd.Series,
    price_frame: pd.DataFrame,
    holding_days: int,
    cost_rate: float,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame] | None:
    normalized_prices = price_frame.copy()
    normalized_prices[DATE_COLUMN] = pd.to_datetime(normalized_prices[DATE_COLUMN]).dt.tz_localize(
        None
    )
    normalized_prices = normalized_prices.sort_values(DATE_COLUMN, kind="stable").reset_index(
        drop=True
    )

    signal_matches = normalized_prices.index[
        normalized_prices[DATE_COLUMN] == signal_row["signal_date"]
    ]
    if len(signal_matches) == 0:
        return None

    signal_index = int(signal_matches[0])
    entry_index = signal_index + 1
    exit_index = signal_index + holding_days
    if exit_index >= len(normalized_prices):
        return None

    trade_window = normalized_prices.loc[
        entry_index:exit_index,
        [DATE_COLUMN, "open", "close"],
    ].copy()
    if len(trade_window) != holding_days:
        return None
    if pd.isna(trade_window.iloc[0]["open"]) or trade_window["close"].isna().any():
        return None

    # 用入场日开盘价作为第一天的前值，把整笔交易拆成可聚合的日度收益路径。
    previous_close = trade_window["close"].shift(1)
    previous_close.iloc[0] = trade_window.iloc[0]["open"]
    trade_window["trade_gross_return"] = trade_window["close"] / previous_close - 1.0
    trade_window["trade_net_return"] = trade_window["trade_gross_return"].copy()
    trade_window.iloc[0, trade_window.columns.get_loc("trade_net_return")] -= cost_rate
    trade_window.iloc[-1, trade_window.columns.get_loc("trade_net_return")] -= cost_rate

    gross_trade_return = float((1.0 + trade_window["trade_gross_return"]).prod() - 1.0)
    net_trade_return = float((1.0 + trade_window["trade_net_return"]).prod() - 1.0)
    trade_record = {
        FOLD_ID_COLUMN: int(signal_row[FOLD_ID_COLUMN]),
        "signal_date": signal_row["signal_date"],
        SYMBOL_COLUMN: signal_row[SYMBOL_COLUMN],
        SIGNAL_SCORE_COLUMN: float(signal_row[SIGNAL_SCORE_COLUMN]),
        FUTURE_RETURN_5D_COLUMN: float(signal_row[FUTURE_RETURN_5D_COLUMN]),
        "position_rank": int(signal_row["position_rank"]),
        "selected_count": int(signal_row["selected_count"]),
        "entry_date": trade_window.iloc[0][DATE_COLUMN],
        "exit_date": trade_window.iloc[-1][DATE_COLUMN],
        "entry_open": float(trade_window.iloc[0]["open"]),
        "exit_close": float(trade_window.iloc[-1]["close"]),
        "gross_trade_return": gross_trade_return,
        "net_trade_return": net_trade_return,
    }
    return signal_row, trade_record, trade_window.loc[
        :, [DATE_COLUMN, "trade_gross_return", "trade_net_return"]
    ]
