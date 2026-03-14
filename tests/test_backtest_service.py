from pathlib import Path

import pandas as pd

from suffering.backtest.service import BacktestService
from suffering.config.settings import Settings
from suffering.data.storage import DailyDataStorage
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
