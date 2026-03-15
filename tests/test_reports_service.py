from pathlib import Path

import pandas as pd
import pytest

from suffering.backtest.storage import BacktestStorage
from suffering.config.settings import Settings
from suffering.reports import build_report_service
from suffering.training.storage import TrainingStorage


def build_walkforward_summary() -> dict[str, object]:
    return {
        "dataset_name": "panel_5d",
        "model_name": "xgb_ranker",
        "task_type": "ranking",
        "fold_count": 3,
        "test_metrics_summary": {
            "daily_rank_ic_mean": {
                "mean": -0.01,
                "std": 0.03,
                "min": -0.05,
                "max": 0.02,
            },
            "daily_rank_ic_std": {
                "mean": 0.45,
                "std": 0.02,
                "min": 0.42,
                "max": 0.47,
            },
            "top_5_mean_future_return": {
                "mean": 0.004,
                "std": 0.006,
                "min": -0.003,
                "max": 0.011,
            },
            "top_10_mean_future_return": {
                "mean": 0.003,
                "std": 0.005,
                "min": -0.002,
                "max": 0.009,
            },
            "ndcg_at_5_mean": {
                "mean": 0.61,
                "std": 0.01,
                "min": 0.60,
                "max": 0.62,
            },
        },
        "notes": [],
    }


def build_backtest_summary() -> dict[str, object]:
    return {
        "model_name": "xgb_ranker",
        "top_k": 5,
        "holding_days": 5,
        "cost_bps_per_side": 5.0,
        "portfolio_date_start": "2021-01-04",
        "portfolio_date_end": "2024-12-30",
        "total_return_gross": 1.20,
        "total_return_net": 0.80,
        "annualized_return_gross": 0.22,
        "annualized_return_net": 0.16,
        "sharpe_ratio_gross": 0.92,
        "sharpe_ratio_net": 0.70,
        "max_drawdown_gross": -0.31,
        "max_drawdown_net": -0.41,
        "average_daily_turnover": 0.45,
        "average_active_positions": 24.0,
        "skipped_trade_count": 1,
    }


def write_full_artifacts(tmp_path: Path) -> Settings:
    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        default_report_model="xgb_ranker",
    )
    training_storage = TrainingStorage.from_settings(settings)
    backtest_storage = BacktestStorage.from_settings(settings)

    training_storage.write_walkforward_summary("xgb_ranker", build_walkforward_summary())
    training_storage.write_walkforward_folds(
        "xgb_ranker",
        pd.DataFrame(
            {
                "fold_id": [1, 2, 3],
                "test_date_start": pd.to_datetime(["2021-01-04", "2022-01-03", "2023-01-03"]),
                "test_date_end": pd.to_datetime(["2021-12-30", "2022-12-30", "2023-12-29"]),
                "daily_rank_ic_mean": [0.01, -0.03, -0.01],
                "top_5_mean_future_return": [0.01, -0.004, 0.006],
                "ndcg_at_5_mean": [0.60, 0.61, 0.62],
            }
        ),
    )
    training_storage.write_walkforward_predictions(
        "xgb_ranker",
        pd.DataFrame(
            {
                "fold_id": [1, 1, 1],
                "date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05"]),
                "symbol": ["AAPL", "MSFT", "NVDA"],
                "future_return_5d": [0.01, 0.02, -0.01],
                "score_pred": [0.8, 0.7, 0.6],
            }
        ),
    )

    backtest_storage.write_summary("xgb_ranker", 5, 5, 5.0, build_backtest_summary())
    comparison_rows = pd.DataFrame(
        [
            {
                "strategy_name": "xgb_ranker",
                "task_type": "model",
                "total_return_net": 0.80,
                "sharpe_ratio_net": 0.70,
                "max_drawdown_net": -0.41,
                "annualized_return_net": 0.16,
                "average_daily_turnover": 0.45,
            },
            {
                "strategy_name": "simple_momentum_top_k",
                "task_type": "benchmark",
                "total_return_net": 0.90,
                "sharpe_ratio_net": 0.85,
                "max_drawdown_net": -0.30,
                "annualized_return_net": 0.18,
                "average_daily_turnover": 0.42,
            },
            {
                "strategy_name": "qqq_buy_and_hold",
                "task_type": "benchmark",
                "total_return_net": 0.55,
                "sharpe_ratio_net": 0.62,
                "max_drawdown_net": -0.35,
                "annualized_return_net": 0.12,
                "average_daily_turnover": 0.00,
            },
            {
                "strategy_name": "equal_weight_universe_buy_and_hold",
                "task_type": "benchmark",
                "total_return_net": 1.10,
                "sharpe_ratio_net": 0.95,
                "max_drawdown_net": -0.38,
                "annualized_return_net": 0.20,
                "average_daily_turnover": 0.00,
            },
        ]
    )
    backtest_storage.write_comparison_table("xgb_ranker", 5, 5, 5.0, comparison_rows)
    backtest_storage.write_comparison_summary(
        "xgb_ranker",
        5,
        5,
        5.0,
        {
            "model_name": "xgb_ranker",
            "model_strategy": {
                "strategy_name": "xgb_ranker",
                "total_return_net": 0.80,
                "sharpe_ratio_net": 0.70,
                "max_drawdown_net": -0.41,
                "annualized_return_net": 0.16,
                "average_daily_turnover": 0.45,
            },
            "strategies": comparison_rows.to_dict(orient="records"),
            "comparison_date_start": "2021-01-04",
            "comparison_date_end": "2024-12-30",
        },
    )
    robustness_rows = pd.DataFrame(
        [
            {
                "strategy_name": "model_strategy",
                "task_type": "ranking",
                "model_name": "xgb_ranker",
                "top_k": 5,
                "holding_days": 5,
                "cost_bps_per_side": 5.0,
                "total_return_net": 0.80,
                "sharpe_ratio_net": 0.70,
                "max_drawdown_net": -0.41,
                "annualized_return_net": 0.16,
                "annualized_volatility": 0.22,
                "average_daily_turnover": 0.45,
                "average_active_positions": 24.0,
                "start_date": "2021-01-04",
                "end_date": "2024-12-30",
            },
            {
                "strategy_name": "model_strategy",
                "task_type": "ranking",
                "model_name": "xgb_ranker",
                "top_k": 10,
                "holding_days": 5,
                "cost_bps_per_side": 5.0,
                "total_return_net": 0.60,
                "sharpe_ratio_net": 0.55,
                "max_drawdown_net": -0.35,
                "annualized_return_net": 0.12,
                "annualized_volatility": 0.20,
                "average_daily_turnover": 0.30,
                "average_active_positions": 30.0,
                "start_date": "2021-01-04",
                "end_date": "2024-12-30",
            },
        ]
    )
    backtest_storage.write_robustness_table("xgb_ranker", robustness_rows)
    backtest_storage.write_robustness_summary(
        "xgb_ranker",
        {
            "model_name": "xgb_ranker",
            "top_k_values": [5, 10],
            "holding_days_values": [5],
            "cost_bps_values": [5.0],
            "best_config_by_sharpe_net": robustness_rows.iloc[0].to_dict(),
            "best_config_by_total_return_net": robustness_rows.iloc[0].to_dict(),
            "simple_momentum_best_sharpe_net": {
                "strategy_name": "simple_momentum_top_k",
                "top_k": 5,
                "holding_days": 5,
                "cost_bps_per_side": 5.0,
                "sharpe_ratio_net": 0.85,
            },
            "simple_momentum_best_total_return_net": {
                "strategy_name": "simple_momentum_top_k",
                "top_k": 5,
                "holding_days": 5,
                "cost_bps_per_side": 5.0,
                "total_return_net": 0.90,
            },
            "whether_model_beats_simple_momentum_on_best_sharpe": False,
            "whether_model_beats_simple_momentum_on_best_total_return": False,
            "robustness_notes": ["only performs well at one narrow configuration"],
        },
    )
    return settings


def test_report_service_generates_markdown_and_summary(tmp_path: Path) -> None:
    settings = write_full_artifacts(tmp_path)
    service = build_report_service(settings)

    summary = service.generate_research_report(
        model_name="xgb_ranker",
        top_k=5,
        holding_days=5,
        cost_bps_per_side=5.0,
    )

    assert summary["model_name"] == "xgb_ranker"
    assert "walkforward" in summary["available_sections"]
    assert "robustness" in summary["available_sections"]
    report_path = Path(summary["report_path"])
    assert report_path.exists()

    markdown = report_path.read_text(encoding="utf-8")
    assert "## 基准对比" in markdown
    assert "## 执行摘要" not in markdown
    assert "下一步研究建议" not in markdown
    assert "建议先回头审视特征与标签设计" not in markdown


def test_report_service_supports_partial_report_when_robustness_missing(tmp_path: Path) -> None:
    settings = write_full_artifacts(tmp_path)
    robustness_dir = settings.artifacts_dir / "backtests" / "robustness"
    for path in robustness_dir.glob("*"):
        path.unlink()

    service = build_report_service(settings)
    summary = service.generate_research_report(
        model_name="xgb_ranker",
        top_k=5,
        holding_days=5,
        cost_bps_per_side=5.0,
    )

    assert "robustness" in summary["missing_sections"]
    report_text = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "对应 artifact 缺失：未找到 robustness summary" in report_text


def test_report_service_requires_at_least_one_input_artifact(tmp_path: Path) -> None:
    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        default_report_model="xgb_ranker",
    )
    service = build_report_service(settings)

    with pytest.raises(FileNotFoundError):
        service.generate_research_report(model_name="xgb_ranker")
