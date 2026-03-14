import pandas as pd

from suffering.backtest.benchmarks import (
    align_daily_returns_to_target_dates,
    build_equal_weight_universe_buy_and_hold_benchmark,
    build_qqq_buy_and_hold_benchmark,
    build_simple_momentum_top_k_benchmark,
)


def build_price_frame(symbol: str, close_values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=len(close_values), freq="B")
    opens = [close_values[0]] + close_values[:-1]
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": close_values,
            "low": opens,
            "close": close_values,
            "adj_close": close_values,
            "volume": [1_000_000] * len(close_values),
            "symbol": [symbol] * len(close_values),
        }
    )


def test_qqq_buy_and_hold_benchmark_builds_gross_and_net_returns() -> None:
    target_dates = pd.Series(pd.date_range("2024-01-02", periods=4, freq="B"))
    result = build_qqq_buy_and_hold_benchmark(
        target_dates=target_dates,
        price_frame=build_price_frame("QQQ", [100.0, 102.0, 103.0, 105.0]),
        cost_bps_per_side=5,
    )

    assert result.strategy_name == "qqq_buy_and_hold"
    assert result.daily_returns["date"].tolist() == target_dates.tolist()
    assert result.summary["trade_count"] == 1
    assert result.summary["total_return_gross"] > result.summary["total_return_net"]
    assert result.trades.iloc[0]["entry_date"].strftime("%Y-%m-%d") == "2024-01-02"
    assert result.trades.iloc[0]["exit_date"].strftime("%Y-%m-%d") == "2024-01-05"


def test_equal_weight_universe_buy_and_hold_skips_symbols_without_full_coverage() -> None:
    target_dates = pd.Series(pd.date_range("2024-01-02", periods=4, freq="B"))
    result = build_equal_weight_universe_buy_and_hold_benchmark(
        target_dates=target_dates,
        price_frames={
            "AAPL": build_price_frame("AAPL", [100.0, 101.0, 102.0, 103.0]),
            "MSFT": build_price_frame("MSFT", [100.0, 101.0, 102.0]),
        },
        cost_bps_per_side=5,
    )

    assert result.summary["symbol_count"] == 2
    assert result.summary["eligible_symbol_count"] == 1
    assert result.summary["skipped_symbol_count"] == 1
    assert result.skipped_trades.iloc[0]["symbol"] == "MSFT"
    assert result.daily_returns["date"].tolist() == target_dates.tolist()


def test_simple_momentum_top_k_uses_feature_scores_to_select_symbols() -> None:
    target_dates = pd.Series(pd.date_range("2024-01-02", periods=4, freq="B"))
    candidate_frame = pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "return_20d": [0.30, 0.10, 0.05, 0.40],
        }
    )
    result = build_simple_momentum_top_k_benchmark(
        target_dates=target_dates,
        candidate_frame=candidate_frame,
        price_frames={
            "AAPL": build_price_frame("AAPL", [100.0, 102.0, 103.0, 104.0, 105.0]),
            "MSFT": build_price_frame("MSFT", [100.0, 101.0, 103.0, 106.0, 108.0]),
        },
        top_k=1,
        holding_days=2,
        cost_bps_per_side=5,
        momentum_feature="return_20d",
    )

    assert result.strategy_name == "simple_momentum_top_k"
    assert result.trades["symbol"].tolist() == ["AAPL", "MSFT"]
    assert result.trades["entry_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-03",
        "2024-01-04",
    ]
    assert result.daily_returns["date"].tolist() == target_dates.tolist()


def test_align_daily_returns_to_target_dates_fills_missing_days_without_future_info() -> None:
    target_dates = pd.Series(pd.date_range("2024-01-02", periods=4, freq="B"))
    daily_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "gross_return": [0.01, -0.02],
            "net_return": [0.009, -0.021],
            "turnover": [0.5, 0.0],
            "active_positions": [1, 1],
            "active_cohorts": [1, 1],
        }
    )

    aligned = align_daily_returns_to_target_dates(daily_returns, target_dates)

    assert aligned["date"].tolist() == target_dates.tolist()
    assert aligned.iloc[0]["gross_return"] == 0.0
    assert aligned.iloc[0]["net_equity"] == 1.0
    assert aligned.iloc[-1]["gross_return"] == 0.0
