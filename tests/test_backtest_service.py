from pathlib import Path

import pandas as pd

from suffering.backtest.service import BacktestService
from suffering.config.settings import Settings
from suffering.data.storage import DailyDataStorage
from suffering.features.storage import DailyFeatureStorage
from suffering.training.storage import TrainingStorage


def build_price_frame(symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    closes = [100.0, 101.0, 102.0, 104.0, 103.0, 105.0, 107.0]
    opens = [100.0] + closes[:-1]
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": closes,
            "low": opens,
            "close": closes,
            "adj_close": closes,
            "volume": [1_000_000] * len(dates),
            "symbol": [symbol] * len(dates),
        }
    )


def test_backtest_service_runs_end_to_end_and_writes_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        default_backtest_model="hist_gbr",
    )
    training_storage = TrainingStorage.from_settings(settings)
    daily_storage = DailyDataStorage.from_settings(settings)

    walkforward_predictions = pd.DataFrame(
        {
            "fold_id": [1, 1],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["AAPL", "MSFT"],
            "y_true": [0.05, 0.02],
            "y_pred": [0.8, 0.6],
        }
    )
    training_storage.write_walkforward_predictions("hist_gbr", walkforward_predictions)
    daily_storage.write_daily_data("AAPL", build_price_frame("AAPL"))
    daily_storage.write_daily_data("MSFT", build_price_frame("MSFT"))

    service = BacktestService.from_settings(settings)
    summary = service.run_walkforward_backtest(
        model_name="hist_gbr",
        top_k=1,
        holding_days=5,
        cost_bps_per_side=5,
    )

    assert summary["model_name"] == "hist_gbr"
    assert summary["top_k"] == 1
    assert summary["holding_days"] == 5
    assert summary["trade_count"] == 1
    assert Path(summary["artifacts"]["summary_path"]).exists()
    assert Path(summary["artifacts"]["daily_returns_path"]).exists()
    assert Path(summary["artifacts"]["equity_curve_path"]).exists()
    assert Path(summary["artifacts"]["trades_path"]).exists()

    report = service.read_backtest_summary(
        model_name="hist_gbr",
        top_k=1,
        holding_days=5,
        cost_bps_per_side=5,
    )
    assert report["trade_count"] == 1
    assert report["total_return_gross"] > report["total_return_net"]

    storage = service.storage
    assert storage.read_daily_returns("hist_gbr", 1, 5, 5).shape[0] == 5
    assert storage.read_equity_curve("hist_gbr", 1, 5, 5).shape[0] == 5
    assert storage.read_trades("hist_gbr", 1, 5, 5).shape[0] == 1


def test_backtest_service_runs_comparison_and_writes_comparison_artifacts(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        default_backtest_model="hist_gbr",
    )
    training_storage = TrainingStorage.from_settings(settings)
    daily_storage = DailyDataStorage.from_settings(settings)
    feature_storage = DailyFeatureStorage.from_settings(settings)

    walkforward_predictions = pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "y_true": [0.05, 0.02, 0.03, 0.04],
            "y_pred": [0.9, 0.6, 0.2, 0.8],
        }
    )
    training_storage.write_walkforward_predictions("hist_gbr", walkforward_predictions)

    for symbol in ("AAPL", "MSFT", "QQQ"):
        daily_storage.write_daily_data(symbol, build_price_frame(symbol))

    feature_storage.write_daily_features(
        "AAPL",
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=7, freq="B"),
                "symbol": ["AAPL"] * 7,
                "return_1d": [0.0] * 7,
                "return_5d": [0.0] * 7,
                "return_10d": [0.0] * 7,
                "return_20d": [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02],
                "return_60d": [0.0] * 7,
                "volatility_5d": [0.0] * 7,
                "volatility_20d": [0.0] * 7,
                "volatility_60d": [0.0] * 7,
                "sma_5_ratio": [1.0] * 7,
                "sma_20_ratio": [1.0] * 7,
                "sma_60_ratio": [1.0] * 7,
                "intraday_range_1d": [0.0] * 7,
                "open_to_close_return_1d": [0.0] * 7,
                "gap_return_1d": [0.0] * 7,
                "volume_change_1d": [0.0] * 7,
                "avg_volume_5d": [1.0] * 7,
                "avg_volume_20d": [1.0] * 7,
                "avg_dollar_volume_20d": [1.0] * 7,
            }
        ),
    )
    feature_storage.write_daily_features(
        "MSFT",
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=7, freq="B"),
                "symbol": ["MSFT"] * 7,
                "return_1d": [0.0] * 7,
                "return_5d": [0.0] * 7,
                "return_10d": [0.0] * 7,
                "return_20d": [0.10, 0.40, 0.50, 0.20, 0.15, 0.10, 0.05],
                "return_60d": [0.0] * 7,
                "volatility_5d": [0.0] * 7,
                "volatility_20d": [0.0] * 7,
                "volatility_60d": [0.0] * 7,
                "sma_5_ratio": [1.0] * 7,
                "sma_20_ratio": [1.0] * 7,
                "sma_60_ratio": [1.0] * 7,
                "intraday_range_1d": [0.0] * 7,
                "open_to_close_return_1d": [0.0] * 7,
                "gap_return_1d": [0.0] * 7,
                "volume_change_1d": [0.0] * 7,
                "avg_volume_5d": [1.0] * 7,
                "avg_volume_20d": [1.0] * 7,
                "avg_dollar_volume_20d": [1.0] * 7,
            }
        ),
    )

    service = BacktestService.from_settings(settings)
    service.run_walkforward_backtest(
        model_name="hist_gbr",
        top_k=1,
        holding_days=5,
        cost_bps_per_side=5,
    )
    comparison = service.run_backtest_comparison(
        model_name="hist_gbr",
        top_k=1,
        holding_days=5,
        cost_bps_per_side=5,
    )

    assert comparison["benchmark_count"] == 4
    assert Path(comparison["artifacts"]["summary_path"]).exists()
    assert Path(comparison["artifacts"]["table_path"]).exists()
    assert comparison["best_benchmark_by_sharpe_net"]["strategy_name"] in {
        "long_short_qqq",
        "qqq_buy_and_hold",
        "equal_weight_universe_buy_and_hold",
        "simple_momentum_top_k",
    }

    report = service.read_backtest_comparison(
        model_name="hist_gbr",
        top_k=1,
        holding_days=5,
        cost_bps_per_side=5,
    )
    strategy_names = [row["strategy_name"] for row in report["table_rows"]]
    assert "hist_gbr" in strategy_names
    assert "long_short_qqq" in strategy_names
    assert "qqq_buy_and_hold" in strategy_names
    assert "equal_weight_universe_buy_and_hold" in strategy_names
    assert "simple_momentum_top_k" in strategy_names
    assert set(report["table_rows"][0].keys()) == {
        "strategy_name",
        "task_type",
        "total_return_net",
        "sharpe_ratio_net",
        "max_drawdown_net",
        "annualized_return_net",
        "annualized_volatility",
        "average_daily_turnover",
        "start_date",
        "end_date",
    }

    benchmark_daily_returns = pd.read_csv(
        service.storage.comparison_daily_returns_path(
            "hist_gbr",
            1,
            5,
            5,
            "simple_momentum_top_k",
        ),
        parse_dates=["date"],
    )
    model_daily_returns = service.storage.read_daily_returns("hist_gbr", 1, 5, 5)
    assert benchmark_daily_returns["date"].shape[0] == model_daily_returns["date"].shape[0]
