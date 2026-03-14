import math

import pandas as pd

from suffering.backtest.metrics import compute_backtest_metrics
from suffering.backtest.portfolio import build_top_k_cohorts, simulate_overlapping_portfolio


def build_signal_frame_for_selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1, 1, 1],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "NVDA", "AAPL", "MSFT", "NVDA"],
            "signal_score": [0.9, 0.8, 0.1, 0.2, 0.7, 0.6],
            "future_return_5d": [0.05, 0.04, -0.01, -0.02, 0.03, 0.02],
            "model_name": ["xgb_ranker"] * 6,
        }
    )


def build_price_frame(
    start_date: str,
    periods: int,
    daily_return: float = 0.1,
) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=periods, freq="B")
    closes = [100.0]
    for _ in range(periods - 1):
        closes.append(closes[-1] * (1.0 + daily_return))
    opens = [100.0] + closes[:-1]
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": closes,
            "low": opens,
            "close": closes,
            "adj_close": closes,
            "volume": [1_000_000] * periods,
            "symbol": ["AAPL"] * periods,
        }
    )


def test_build_top_k_cohorts_selects_daily_top_names_with_equal_weight() -> None:
    cohorts = build_top_k_cohorts(
        signals=build_signal_frame_for_selection(),
        top_k=2,
        holding_days=5,
    )

    assert cohorts["symbol"].tolist() == ["AAPL", "MSFT", "MSFT", "NVDA"]
    assert cohorts["position_rank"].tolist() == [1, 2, 1, 2]
    assert cohorts["selected_count"].tolist() == [2, 2, 2, 2]
    assert cohorts["cohort_weight"].tolist() == [0.2, 0.2, 0.2, 0.2]
    assert cohorts["position_weight"].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert cohorts["portfolio_weight"].tolist() == [0.1, 0.1, 0.1, 0.1]


def test_simulate_overlapping_portfolio_uses_t_plus_1_entry_and_t_plus_5_exit() -> None:
    signals = pd.DataFrame(
        {
            "fold_id": [1],
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "signal_score": [0.9],
            "future_return_5d": [0.61051],
            "model_name": ["hist_gbr"],
        }
    )
    cohorts = build_top_k_cohorts(signals=signals, top_k=1, holding_days=5)
    result = simulate_overlapping_portfolio(
        cohorts=cohorts,
        price_frames={"AAPL": build_price_frame("2024-01-02", periods=6)},
        holding_days=5,
        cost_bps_per_side=5,
    )

    trade = result.trades.iloc[0]
    first_day = result.daily_returns.iloc[0]
    last_day = result.daily_returns.iloc[-1]

    assert trade["entry_date"].strftime("%Y-%m-%d") == "2024-01-03"
    assert trade["exit_date"].strftime("%Y-%m-%d") == "2024-01-09"
    assert math.isclose(trade["gross_trade_return"], 0.61051, rel_tol=1e-6)
    assert trade["net_trade_return"] < trade["gross_trade_return"]
    assert math.isclose(first_day["gross_return"], 0.02, rel_tol=1e-6)
    assert math.isclose(first_day["net_return"], 0.0199, rel_tol=1e-6)
    assert math.isclose(last_day["gross_return"], 0.02, rel_tol=1e-6)
    assert math.isclose(last_day["net_return"], 0.0199, rel_tol=1e-6)


def test_simulate_overlapping_portfolio_caps_active_cohorts_by_holding_days() -> None:
    signal_dates = pd.date_range("2024-01-02", periods=5, freq="B")
    signals = pd.DataFrame(
        {
            "fold_id": [1] * len(signal_dates),
            "date": signal_dates,
            "symbol": ["AAPL"] * len(signal_dates),
            "signal_score": [1.0, 0.9, 0.8, 0.7, 0.6],
            "future_return_5d": [0.1] * len(signal_dates),
            "model_name": ["hist_gbr"] * len(signal_dates),
        }
    )
    cohorts = build_top_k_cohorts(signals=signals, top_k=1, holding_days=3)
    result = simulate_overlapping_portfolio(
        cohorts=cohorts,
        price_frames={"AAPL": build_price_frame("2024-01-02", periods=8)},
        holding_days=3,
        cost_bps_per_side=5,
    )

    assert result.daily_returns["active_cohorts"].max() == 3
    assert result.daily_returns["active_positions"].max() == 3


def test_compute_backtest_metrics_returns_required_summary_fields() -> None:
    signals = pd.DataFrame(
        {
            "fold_id": [1],
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "signal_score": [0.9],
            "future_return_5d": [0.61051],
            "model_name": ["hist_gbr"],
        }
    )
    cohorts = build_top_k_cohorts(signals=signals, top_k=1, holding_days=5)
    result = simulate_overlapping_portfolio(
        cohorts=cohorts,
        price_frames={"AAPL": build_price_frame("2024-01-02", periods=6)},
        holding_days=5,
        cost_bps_per_side=5,
    )

    metrics = compute_backtest_metrics(result.daily_returns, result.trades)

    assert metrics["total_return_gross"] > 0
    assert metrics["total_return_net"] < metrics["total_return_gross"]
    assert metrics["annualized_return_gross"] > 0
    assert metrics["annualized_return_net"] > 0
    assert metrics["annualized_volatility"] >= 0
    assert isinstance(metrics["sharpe_ratio_gross"], float)
    assert isinstance(metrics["sharpe_ratio_net"], float)
    assert metrics["max_drawdown_gross"] <= 0
    assert metrics["max_drawdown_net"] <= 0
    assert metrics["daily_hit_rate_gross"] == 1.0
    assert metrics["daily_hit_rate_net"] == 1.0
    assert metrics["average_daily_turnover"] > 0
    assert metrics["average_active_positions"] == 1.0
