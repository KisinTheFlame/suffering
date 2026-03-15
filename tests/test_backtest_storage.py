from pathlib import Path

import pandas as pd

from suffering.backtest.storage import BacktestStorage


def test_backtest_storage_writes_and_reads_artifacts(tmp_path: Path) -> None:
    storage = BacktestStorage(artifacts_dir=tmp_path)
    summary = {"model_name": "xgb_ranker", "top_k": 5, "holding_days": 5}
    daily_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "gross_return": [0.01],
            "net_return": [0.009],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "gross_equity": [1.01],
            "net_equity": [1.009],
        }
    )
    trades = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02"]),
            "entry_date": pd.to_datetime(["2024-01-03"]),
            "exit_date": pd.to_datetime(["2024-01-09"]),
            "symbol": ["AAPL"],
            "gross_trade_return": [0.05],
        }
    )

    summary_path = storage.write_summary("xgb_ranker", 5, 5, 5, summary)
    daily_path = storage.write_daily_returns("xgb_ranker", 5, 5, 5, daily_returns)
    equity_path = storage.write_equity_curve("xgb_ranker", 5, 5, 5, equity_curve)
    trades_path = storage.write_trades("xgb_ranker", 5, 5, 5, trades)

    assert summary_path == tmp_path / "backtests" / "xgb_ranker_top5_h5_cost10_summary.json"
    assert daily_path == tmp_path / "backtests" / "xgb_ranker_top5_h5_cost10_daily_returns.csv"
    assert equity_path == tmp_path / "backtests" / "xgb_ranker_top5_h5_cost10_equity_curve.csv"
    assert trades_path == tmp_path / "backtests" / "xgb_ranker_top5_h5_cost10_trades.csv"

    assert storage.read_summary("xgb_ranker", 5, 5, 5) == summary
    assert storage.read_daily_returns("xgb_ranker", 5, 5, 5).shape[0] == 1
    assert storage.read_equity_curve("xgb_ranker", 5, 5, 5).shape[0] == 1
    assert storage.read_trades("xgb_ranker", 5, 5, 5).shape[0] == 1


def test_backtest_storage_writes_and_reads_comparison_artifacts(tmp_path: Path) -> None:
    storage = BacktestStorage(artifacts_dir=tmp_path)
    comparison_summary = {"model_name": "xgb_ranker", "benchmark_count": 3}
    comparison_table = pd.DataFrame(
        {
            "strategy_name": ["xgb_ranker", "simple_momentum_top_k"],
            "task_type": ["model", "benchmark"],
            "start_date": ["2024-01-03", "2024-01-03"],
            "end_date": ["2024-01-09", "2024-01-09"],
        }
    )

    summary_path = storage.write_comparison_summary("xgb_ranker", 5, 5, 5, comparison_summary)
    table_path = storage.write_comparison_table("xgb_ranker", 5, 5, 5, comparison_table)

    assert (
        summary_path
        == tmp_path
        / "backtests"
        / "comparisons"
        / "xgb_ranker_top5_h5_cost10_comparison_summary.json"
    )
    assert (
        table_path
        == tmp_path / "backtests" / "comparisons" / "xgb_ranker_top5_h5_cost10_comparison_table.csv"
    )
    assert storage.read_comparison_summary("xgb_ranker", 5, 5, 5) == comparison_summary
    assert storage.read_comparison_table("xgb_ranker", 5, 5, 5).shape[0] == 2


def test_backtest_storage_writes_and_reads_robustness_artifacts(tmp_path: Path) -> None:
    storage = BacktestStorage(artifacts_dir=tmp_path)
    robustness_summary = {"model_name": "xgb_ranker", "total_configs_evaluated": 27}
    robustness_table = pd.DataFrame(
        {
            "strategy_name": ["model_strategy", "simple_momentum_top_k"],
            "task_type": ["ranking", "benchmark"],
            "model_name": ["xgb_ranker", None],
            "top_k": [5, 5],
            "holding_days": [5, 5],
            "cost_bps_per_side": [5.0, 5.0],
        }
    )

    summary_path = storage.write_robustness_summary("xgb_ranker", robustness_summary)
    table_path = storage.write_robustness_table("xgb_ranker", robustness_table)

    assert (
        summary_path
        == tmp_path / "backtests" / "robustness" / "xgb_ranker_robustness_summary.json"
    )
    assert (
        table_path
        == tmp_path / "backtests" / "robustness" / "xgb_ranker_robustness_table.csv"
    )
    assert storage.read_robustness_summary("xgb_ranker") == robustness_summary
    assert storage.read_robustness_table("xgb_ranker").shape[0] == 2
