from suffering.reports.markdown_report import render_markdown_report


def build_report_context() -> dict[str, object]:
    return {
        "metadata": {
            "model_name": "xgb_ranker",
            "task_type": "ranking",
            "generated_at": "2026-03-15 10:00:00 CST",
            "dataset_name": "panel_5d",
            "universe_description": (
                "来自 walk-forward 预测的 6 只股票："
                "AAPL, AMZN, GOOGL, META, MSFT, NVDA"
            ),
            "date_range": "2021-01-04 -> 2024-12-30",
            "note": "本报告仅基于 walk-forward 样本外预测结果生成。",
        },
        "available_artifacts": [{"name": "walkforward_summary", "path": "a.json"}],
        "missing_artifacts": [{"name": "robustness_summary", "path": "b.json"}],
        "walkforward": {
            "available": True,
            "fold_count": 4,
            "daily_rank_ic_mean": "-0.0031",
            "top_5_mean_future_return": "0.56%",
            "top_10_mean_future_return": "0.55%",
            "ndcg_at_5_mean": "0.6038",
            "notes": [],
            "metric_rows": [
                {
                    "metric": "daily_rank_ic_mean",
                    "mean": "-0.0031",
                    "std": "0.0325",
                    "min": "-0.0409",
                    "max": "0.0466",
                }
            ],
            "metric_columns": ["metric", "mean", "std", "min", "max"],
            "fold_rows": [
                {
                    "fold_id": 1,
                    "test_date_start": "2020-12-31",
                    "test_date_end": "2021-12-28",
                    "daily_rank_ic_mean": "0.0019",
                    "top_5_mean_future_return": "0.76%",
                    "ndcg_at_5_mean": "0.6017",
                }
            ],
            "fold_columns": [
                "fold_id",
                "test_date_start",
                "test_date_end",
                "daily_rank_ic_mean",
                "top_5_mean_future_return",
                "ndcg_at_5_mean",
            ],
            "missing_items": [],
        },
        "backtest": {
            "available": True,
            "total_return_gross": "166.17%",
            "total_return_net": "117.95%",
            "annualized_return_gross": "27.85%",
            "annualized_return_net": "21.60%",
            "sharpe_ratio_gross": "1.0458",
            "sharpe_ratio_net": "0.8598",
            "max_drawdown_gross": "-43.82%",
            "max_drawdown_net": "-46.46%",
            "average_daily_turnover": "39.84%",
            "average_active_positions": "24.9004",
            "skipped_trade_count": 0,
            "assumptions": [
                "每个信号日按 `top_k=5` 选股。",
                "交易假设为 `t+1` 开盘买入、`t+5` 收盘卖出。",
            ],
        },
        "benchmark_comparison": {
            "available": True,
            "columns": [
                "strategy_name",
                "total_return_net",
                "sharpe_ratio_net",
                "max_drawdown_net",
                "annualized_return_net",
                "average_daily_turnover",
            ],
            "table_rows": [
                {
                    "strategy_name": "xgb_ranker",
                    "total_return_net": "117.95%",
                    "sharpe_ratio_net": "0.8598",
                    "max_drawdown_net": "-46.46%",
                    "annualized_return_net": "21.60%",
                    "average_daily_turnover": "39.84%",
                },
                {
                    "strategy_name": "simple_momentum_top_k",
                    "total_return_net": "102.33%",
                    "sharpe_ratio_net": "0.7862",
                    "max_drawdown_net": "-45.81%",
                    "annualized_return_net": "19.35%",
                    "average_daily_turnover": "39.80%",
                },
            ],
            "missing_items": ["qqq_buy_and_hold"],
        },
        "robustness": {
            "available": False,
            "missing_message": "未找到 robustness summary",
        },
    }


def test_render_markdown_report_includes_required_sections_and_tables() -> None:
    markdown = render_markdown_report(build_report_context())

    assert "# 研究报告：xgb_ranker" in markdown
    assert "## Walk-Forward 验证摘要" in markdown
    assert "## 回测摘要" in markdown
    assert "## 基准对比" in markdown
    assert "## 稳健性摘要" in markdown
    assert "| 策略名 | 净总收益 | 净 Sharpe |" in markdown
    assert "对应 artifact 缺失：未找到 robustness summary" in markdown
    assert "缺失的可选 artifact：qqq_buy_and_hold" in markdown
    assert "执行摘要" not in markdown
    assert "下一步研究建议" not in markdown


def test_render_markdown_report_marks_missing_artifacts_in_metadata() -> None:
    markdown = render_markdown_report(build_report_context())

    assert "- 已读取 artifact：walkforward_summary" in markdown
    assert "- 缺失 artifact：robustness_summary" in markdown
