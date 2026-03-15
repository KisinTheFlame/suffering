from pathlib import Path

import pandas as pd

from suffering.backtest.robustness import build_robustness_parameter_grid
from suffering.backtest.service import BacktestService
from suffering.config.settings import Settings
from suffering.data.storage import DailyDataStorage
from suffering.features.storage import DailyFeatureStorage
from suffering.training.storage import TrainingStorage


def build_price_frame(symbol: str, close_offset: float) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    closes = [100.0 + close_offset + (index * (1.2 + close_offset * 0.05)) for index in range(20)]
    opens = [closes[0]] + closes[:-1]
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


def build_feature_frame(symbol: str, stronger_first_half: bool) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    base_scores = [0.30, 0.25, 0.35, 0.20, 0.40, 0.15, 0.45, 0.10, 0.50, 0.05]
    if not stronger_first_half:
        base_scores = list(reversed(base_scores))
    repeated_scores = (base_scores * 2)[:20]
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": [symbol] * 20,
            "return_1d": [0.0] * 20,
            "return_5d": [0.0] * 20,
            "return_10d": [0.0] * 20,
            "return_20d": repeated_scores,
            "return_60d": [0.0] * 20,
            "volatility_5d": [0.0] * 20,
            "volatility_20d": [0.0] * 20,
            "volatility_60d": [0.0] * 20,
            "sma_5_ratio": [1.0] * 20,
            "sma_20_ratio": [1.0] * 20,
            "sma_60_ratio": [1.0] * 20,
            "intraday_range_1d": [0.0] * 20,
            "open_to_close_return_1d": [0.0] * 20,
            "gap_return_1d": [0.0] * 20,
            "volume_change_1d": [0.0] * 20,
            "avg_volume_5d": [1.0] * 20,
            "avg_volume_20d": [1.0] * 20,
            "avg_dollar_volume_20d": [1.0] * 20,
        }
    )


def test_build_robustness_parameter_grid_generates_expected_combinations() -> None:
    grid = build_robustness_parameter_grid(
        top_k_values=[3, 5],
        holding_days_values=[3, 5],
        cost_bps_values=[0, 5],
    )

    assert grid == [
        (3, 3, 0.0),
        (3, 3, 5.0),
        (3, 5, 0.0),
        (3, 5, 5.0),
        (5, 3, 0.0),
        (5, 3, 5.0),
        (5, 5, 0.0),
        (5, 5, 5.0),
    ]


def test_backtest_service_runs_robustness_and_writes_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        default_backtest_model="xgb_ranker",
    )
    training_storage = TrainingStorage.from_settings(settings)
    daily_storage = DailyDataStorage.from_settings(settings)
    feature_storage = DailyFeatureStorage.from_settings(settings)

    prediction_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-04",
            "2024-01-04",
            "2024-01-05",
            "2024-01-05",
            "2024-01-08",
            "2024-01-08",
        ]
    )
    walkforward_predictions = pd.DataFrame(
        {
            "fold_id": [1] * len(prediction_dates),
            "date": prediction_dates,
            "symbol": ["AAPL", "MSFT"] * 5,
            "future_return_5d": [0.06, 0.03, 0.01, 0.05, 0.04, 0.02, 0.03, 0.01, 0.02, 0.04],
            "score_pred": [0.90, 0.70, 0.20, 0.85, 0.82, 0.50, 0.65, 0.40, 0.45, 0.75],
        }
    )
    training_storage.write_walkforward_predictions("xgb_ranker", walkforward_predictions)

    daily_storage.write_daily_data("AAPL", build_price_frame("AAPL", close_offset=0.0))
    daily_storage.write_daily_data("MSFT", build_price_frame("MSFT", close_offset=3.0))
    daily_storage.write_daily_data("QQQ", build_price_frame("QQQ", close_offset=1.5))

    feature_storage.write_daily_features(
        "AAPL",
        build_feature_frame("AAPL", stronger_first_half=True),
    )
    feature_storage.write_daily_features(
        "MSFT",
        build_feature_frame("MSFT", stronger_first_half=False),
    )

    service = BacktestService.from_settings(settings)
    summary = service.run_backtest_robustness(
        model_name="xgb_ranker",
        top_k_values=[1, 2],
        holding_days_values=[3, 5],
        cost_bps_values=[0, 5],
    )

    assert summary["model_name"] == "xgb_ranker"
    assert summary["total_configs_evaluated"] == 8
    assert summary["row_count"] == 24
    assert summary["best_config_by_sharpe_net"] is not None
    assert summary["simple_momentum_best_sharpe_net"] is not None
    assert summary["whether_model_beats_simple_momentum_on_best_sharpe"] in {True, False}
    assert summary["robustness_notes"]
    assert Path(summary["artifacts"]["summary_path"]).exists()
    assert Path(summary["artifacts"]["table_path"]).exists()

    report = service.read_backtest_robustness(model_name="xgb_ranker")
    table_frame = pd.DataFrame(report["table_rows"])
    assert set(table_frame["strategy_name"]) == {
        "model_strategy",
        "simple_momentum_top_k",
        "qqq_buy_and_hold",
        "equal_weight_universe_buy_and_hold",
    }
    assert set(table_frame.columns) == {
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
    }
