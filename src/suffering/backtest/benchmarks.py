"""Minimal benchmark builders aligned with the existing backtest layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from suffering.backtest.metrics import compute_backtest_metrics
from suffering.backtest.portfolio import build_top_k_cohorts, simulate_overlapping_portfolio
from suffering.backtest.signals import FOLD_ID_COLUMN, MODEL_NAME_COLUMN, SIGNAL_SCORE_COLUMN
from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.features.definitions import FEATURE_COLUMNS
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN


@dataclass(frozen=True)
class BenchmarkBacktestResult:
    strategy_name: str
    task_type: str
    summary: dict[str, Any]
    daily_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    skipped_trades: pd.DataFrame


def build_qqq_buy_and_hold_benchmark(
    target_dates: pd.Series,
    price_frame: pd.DataFrame,
    cost_bps_per_side: float,
) -> BenchmarkBacktestResult:
    return build_buy_and_hold_benchmark(
        strategy_name="qqq_buy_and_hold",
        target_dates=target_dates,
        price_frames={"QQQ": price_frame},
        cost_bps_per_side=cost_bps_per_side,
    )


def build_equal_weight_universe_buy_and_hold_benchmark(
    target_dates: pd.Series,
    price_frames: dict[str, pd.DataFrame],
    cost_bps_per_side: float,
) -> BenchmarkBacktestResult:
    return build_buy_and_hold_benchmark(
        strategy_name="equal_weight_universe_buy_and_hold",
        target_dates=target_dates,
        price_frames=price_frames,
        cost_bps_per_side=cost_bps_per_side,
    )


def build_buy_and_hold_benchmark(
    strategy_name: str,
    target_dates: pd.Series,
    price_frames: dict[str, pd.DataFrame],
    cost_bps_per_side: float,
) -> BenchmarkBacktestResult:
    normalized_target_dates = _normalize_target_dates(target_dates)
    if len(normalized_target_dates) < 1:
        raise ValueError("target_dates must not be empty")

    cost_rate = float(cost_bps_per_side) / 10_000.0
    eligible_trade_records: list[dict[str, Any]] = []
    eligible_trade_frames: list[pd.DataFrame] = []
    skipped_records: list[dict[str, Any]] = []

    for symbol, price_frame in sorted(price_frames.items()):
        trade_path = _build_buy_and_hold_trade_path(
            symbol=symbol,
            price_frame=price_frame,
            target_dates=normalized_target_dates,
            cost_rate=cost_rate,
        )
        if trade_path is None:
            skipped_records.append(
                {
                    "signal_date": normalized_target_dates[0],
                    SYMBOL_COLUMN: symbol,
                    "reason": "insufficient_price_history_for_buy_and_hold",
                }
            )
            continue
        trade_record, trade_daily_frame = trade_path
        eligible_trade_records.append(trade_record)
        eligible_trade_frames.append(trade_daily_frame)

    if not eligible_trade_records:
        raise ValueError(f"No eligible symbols available for benchmark: {strategy_name}")

    position_weight = 1.0 / float(len(eligible_trade_records))
    weighted_frames: list[pd.DataFrame] = []
    trades_output: list[dict[str, Any]] = []
    for trade_id, (trade_record, trade_daily_frame) in enumerate(
        zip(eligible_trade_records, eligible_trade_frames, strict=True),
        start=1,
    ):
        trades_output.append(
            {
                **trade_record,
                "trade_id": trade_id,
                "cohort_trade_count": len(eligible_trade_records),
                "cohort_weight": 1.0,
                "position_weight": position_weight,
                "portfolio_weight": position_weight,
                "cost_bps_per_side": float(cost_bps_per_side),
            }
        )
        weighted_frame = trade_daily_frame.copy()
        weighted_frame["trade_id"] = trade_id
        weighted_frame["signal_date"] = normalized_target_dates[0]
        weighted_frame["portfolio_weight"] = position_weight
        weighted_frame["gross_component"] = weighted_frame["trade_gross_return"] * position_weight
        weighted_frame["net_component"] = weighted_frame["trade_net_return"] * position_weight
        weighted_frame["entry_turnover"] = 0.0
        weighted_frame["exit_turnover"] = 0.0
        weighted_frame.loc[weighted_frame.index[0], "entry_turnover"] = position_weight
        weighted_frame.loc[weighted_frame.index[-1], "exit_turnover"] = position_weight
        weighted_frames.append(weighted_frame)

    trade_daily = pd.concat(weighted_frames, ignore_index=True).sort_values(
        [DATE_COLUMN, "trade_id"],
        kind="stable",
    )
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
    daily_returns = align_daily_returns_to_target_dates(daily_returns, normalized_target_dates)
    equity_curve = daily_returns.loc[:, [DATE_COLUMN, "gross_equity", "net_equity"]].copy()

    trades = pd.DataFrame(trades_output).sort_values(["trade_id", SYMBOL_COLUMN], kind="stable")
    skipped_trades = pd.DataFrame(
        skipped_records,
        columns=["signal_date", SYMBOL_COLUMN, "reason"],
    )
    if not skipped_trades.empty:
        skipped_trades = skipped_trades.sort_values(["signal_date", SYMBOL_COLUMN], kind="stable")

    metrics = compute_backtest_metrics(daily_returns=daily_returns, trades=trades)
    summary = {
        "strategy_name": strategy_name,
        "task_type": "benchmark",
        "symbol_count": int(len(price_frames)),
        "eligible_symbol_count": int(len(eligible_trade_records)),
        "skipped_symbol_count": int(len(skipped_records)),
        **metrics,
    }
    return BenchmarkBacktestResult(
        strategy_name=strategy_name,
        task_type="benchmark",
        summary=summary,
        daily_returns=daily_returns.reset_index(drop=True),
        equity_curve=equity_curve.reset_index(drop=True),
        trades=trades.reset_index(drop=True),
        skipped_trades=skipped_trades.reset_index(drop=True),
    )


def build_simple_momentum_top_k_benchmark(
    target_dates: pd.Series,
    candidate_frame: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    top_k: int,
    holding_days: int,
    cost_bps_per_side: float,
    momentum_feature: str,
) -> BenchmarkBacktestResult:
    normalized_target_dates = _normalize_target_dates(target_dates)
    if momentum_feature not in FEATURE_COLUMNS:
        raise ValueError(f"Unsupported benchmark momentum feature: {momentum_feature}")

    required_columns = {FOLD_ID_COLUMN, DATE_COLUMN, SYMBOL_COLUMN, momentum_feature}
    missing_columns = required_columns.difference(candidate_frame.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Momentum candidate frame is missing required columns: {missing_display}"
        )

    momentum_signals = candidate_frame.copy()
    momentum_signals[DATE_COLUMN] = pd.to_datetime(momentum_signals[DATE_COLUMN]).dt.tz_localize(
        None
    )
    momentum_signals = momentum_signals[
        momentum_signals[DATE_COLUMN].isin(normalized_target_dates)
    ].copy()
    momentum_signals = momentum_signals.dropna(subset=[momentum_feature]).reset_index(drop=True)
    if momentum_signals.empty:
        raise ValueError("No valid momentum signals available in the requested comparison window")

    momentum_signals[SIGNAL_SCORE_COLUMN] = momentum_signals[momentum_feature]
    momentum_signals[FUTURE_RETURN_5D_COLUMN] = float("nan")
    if MODEL_NAME_COLUMN not in momentum_signals.columns:
        momentum_signals[MODEL_NAME_COLUMN] = "simple_momentum_top_k"

    cohorts = build_top_k_cohorts(
        signals=momentum_signals,
        top_k=top_k,
        holding_days=holding_days,
    )
    portfolio_result = simulate_overlapping_portfolio(
        cohorts=cohorts,
        price_frames=price_frames,
        holding_days=holding_days,
        cost_bps_per_side=cost_bps_per_side,
    )
    daily_returns = align_daily_returns_to_target_dates(
        portfolio_result.daily_returns,
        normalized_target_dates,
    )
    equity_curve = daily_returns.loc[:, [DATE_COLUMN, "gross_equity", "net_equity"]].copy()
    metrics = compute_backtest_metrics(daily_returns=daily_returns, trades=portfolio_result.trades)
    summary = {
        "strategy_name": "simple_momentum_top_k",
        "task_type": "benchmark",
        "momentum_feature": momentum_feature,
        "candidate_row_count": int(len(momentum_signals)),
        "selected_row_count": int(len(cohorts)),
        "cohort_count": int(cohorts["signal_date"].nunique()),
        "trade_count": int(len(portfolio_result.trades)),
        "skipped_trade_count": int(len(portfolio_result.skipped_trades)),
        **metrics,
    }
    return BenchmarkBacktestResult(
        strategy_name="simple_momentum_top_k",
        task_type="benchmark",
        summary=summary,
        daily_returns=daily_returns.reset_index(drop=True),
        equity_curve=equity_curve.reset_index(drop=True),
        trades=portfolio_result.trades.reset_index(drop=True),
        skipped_trades=portfolio_result.skipped_trades.reset_index(drop=True),
    )


def align_daily_returns_to_target_dates(
    daily_returns: pd.DataFrame,
    target_dates: pd.Series,
) -> pd.DataFrame:
    normalized_target_dates = _normalize_target_dates(target_dates)
    aligned = pd.DataFrame({DATE_COLUMN: normalized_target_dates})
    if daily_returns.empty:
        aligned["gross_return"] = 0.0
        aligned["net_return"] = 0.0
        aligned["turnover"] = 0.0
        aligned["active_positions"] = 0.0
        aligned["active_cohorts"] = 0.0
    else:
        normalized_daily = daily_returns.copy()
        normalized_daily[DATE_COLUMN] = (
            pd.to_datetime(normalized_daily[DATE_COLUMN]).dt.tz_localize(None)
        )
        aligned = aligned.merge(normalized_daily, on=DATE_COLUMN, how="left")
        for column in ("gross_return", "net_return", "turnover"):
            aligned[column] = aligned[column].fillna(0.0)
        for column in ("active_positions", "active_cohorts"):
            aligned[column] = aligned[column].fillna(0.0)

    aligned["gross_equity"] = (1.0 + aligned["gross_return"]).cumprod()
    aligned["net_equity"] = (1.0 + aligned["net_return"]).cumprod()
    return aligned.loc[
        :,
        [
            DATE_COLUMN,
            "gross_return",
            "net_return",
            "turnover",
            "active_positions",
            "active_cohorts",
            "gross_equity",
            "net_equity",
        ],
    ]


def build_candidate_frame_from_features(
    model_signals: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
    momentum_feature: str,
) -> pd.DataFrame:
    feature_records: list[pd.DataFrame] = []
    for symbol, frame in feature_frames.items():
        feature_records.append(frame.loc[:, [DATE_COLUMN, SYMBOL_COLUMN, momentum_feature]].copy())
    if not feature_records:
        raise ValueError("No feature frames available for benchmark candidate construction")

    feature_panel = pd.concat(feature_records, ignore_index=True)
    feature_panel[DATE_COLUMN] = pd.to_datetime(feature_panel[DATE_COLUMN]).dt.tz_localize(None)
    base = model_signals.loc[:, [FOLD_ID_COLUMN, DATE_COLUMN, SYMBOL_COLUMN]].copy()
    base[DATE_COLUMN] = pd.to_datetime(base[DATE_COLUMN]).dt.tz_localize(None)
    return base.merge(feature_panel, on=[DATE_COLUMN, SYMBOL_COLUMN], how="left")


def extract_price_frames_for_symbols(
    price_frames: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    return {symbol: price_frames[symbol] for symbol in symbols if symbol in price_frames}


def _build_buy_and_hold_trade_path(
    symbol: str,
    price_frame: pd.DataFrame,
    target_dates: pd.Series,
    cost_rate: float,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    normalized_prices = price_frame.copy()
    normalized_prices[DATE_COLUMN] = pd.to_datetime(normalized_prices[DATE_COLUMN]).dt.tz_localize(
        None
    )
    normalized_prices = normalized_prices.sort_values(DATE_COLUMN, kind="stable")
    indexed_prices = normalized_prices.set_index(DATE_COLUMN)
    if not indexed_prices.index.is_unique:
        indexed_prices = indexed_prices[~indexed_prices.index.duplicated(keep="last")]

    trade_window = indexed_prices.reindex(target_dates).reset_index()
    if len(trade_window) != len(target_dates):
        return None
    if trade_window["open"].isna().iloc[0] or trade_window["close"].isna().any():
        return None
    if trade_window[SYMBOL_COLUMN].isna().any():
        return None

    previous_close = trade_window["close"].shift(1)
    previous_close.iloc[0] = trade_window.iloc[0]["open"]
    trade_window["trade_gross_return"] = trade_window["close"] / previous_close - 1.0
    trade_window["trade_net_return"] = trade_window["trade_gross_return"].copy()
    trade_window.iloc[0, trade_window.columns.get_loc("trade_net_return")] -= cost_rate
    trade_window.iloc[-1, trade_window.columns.get_loc("trade_net_return")] -= cost_rate

    gross_trade_return = float((1.0 + trade_window["trade_gross_return"]).prod() - 1.0)
    net_trade_return = float((1.0 + trade_window["trade_net_return"]).prod() - 1.0)
    trade_record = {
        "signal_date": target_dates[0],
        SYMBOL_COLUMN: symbol,
        "entry_date": trade_window.iloc[0][DATE_COLUMN],
        "exit_date": trade_window.iloc[-1][DATE_COLUMN],
        "entry_open": float(trade_window.iloc[0]["open"]),
        "exit_close": float(trade_window.iloc[-1]["close"]),
        "gross_trade_return": gross_trade_return,
        "net_trade_return": net_trade_return,
        "position_rank": 1,
        "selected_count": 1,
    }
    return trade_record, trade_window.loc[
        :,
        [DATE_COLUMN, "trade_gross_return", "trade_net_return"],
    ]


def _normalize_target_dates(target_dates: pd.Series) -> pd.Series:
    normalized = pd.Series(pd.to_datetime(target_dates), name=DATE_COLUMN)
    normalized = normalized.dt.tz_localize(None)
    normalized = normalized.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    return normalized
