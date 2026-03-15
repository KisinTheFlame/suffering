"""Markdown renderer for the minimal research report layer."""

from __future__ import annotations

from typing import Any


def render_markdown_report(context: dict[str, Any]) -> str:
    metadata = context["metadata"]
    lines = [
        f"# 研究报告：{metadata['model_name']}",
        "",
        f"- 模型名：`{metadata['model_name']}`",
        f"- 任务类型：`{metadata['task_type']}`",
        f"- 生成时间：`{metadata['generated_at']}`",
        f"- 数据集：`{metadata['dataset_name']}`",
        f"- 股票池：{metadata['universe_description']}",
        f"- 日期范围：{metadata['date_range']}",
        f"- 说明：{metadata['note']}",
        f"- 已读取 artifact：{_format_artifact_names(context['available_artifacts'])}",
        f"- 缺失 artifact：{_format_artifact_names(context['missing_artifacts'])}",
        "",
        "## Walk-Forward 验证摘要",
        "",
    ]
    lines.extend(_render_walkforward_section(context["walkforward"]))

    lines.extend(["", "## 回测摘要", ""])
    lines.extend(_render_backtest_section(context["backtest"]))

    lines.extend(["", "## 基准对比", ""])
    lines.extend(_render_benchmark_section(context["benchmark_comparison"]))

    lines.extend(["", "## 稳健性摘要", ""])
    lines.extend(_render_robustness_section(context["robustness"]))
    lines.append("")
    return "\n".join(lines)


def _render_walkforward_section(section: dict[str, Any]) -> list[str]:
    if not section["available"]:
        return [f"对应 artifact 缺失：{section['missing_message']}"]

    lines = [
        f"- fold 数：{section['fold_count']}",
        f"- `daily_rank_ic_mean`：{section['daily_rank_ic_mean']}",
        f"- `top_5_mean_future_return`：{section['top_5_mean_future_return']}",
        f"- `top_10_mean_future_return`：{section['top_10_mean_future_return']}",
        f"- `ndcg_at_5_mean`：{section['ndcg_at_5_mean']}",
    ]
    if section["notes"]:
        lines.append(f"- artifact notes：{'; '.join(section['notes'])}")
    if section["metric_rows"]:
        lines.extend(["", _render_table(section["metric_rows"], section["metric_columns"])])
    if section["fold_rows"]:
        lines.extend(
            [
                "",
                "Fold 概览：",
                "",
                _render_table(section["fold_rows"], section["fold_columns"]),
            ]
        )
    if section["missing_items"]:
        lines.append("")
        lines.append(f"缺失的可选 artifact：{', '.join(section['missing_items'])}")
    return lines


def _render_backtest_section(section: dict[str, Any]) -> list[str]:
    if not section["available"]:
        return [f"对应 artifact 缺失：{section['missing_message']}"]

    lines = [
        (
            f"- `total_return_gross / net`：{section['total_return_gross']} / "
            f"{section['total_return_net']}"
        ),
        (
            f"- `annualized_return_gross / net`：{section['annualized_return_gross']} / "
            f"{section['annualized_return_net']}"
        ),
        (
            f"- `sharpe_ratio_gross / net`：{section['sharpe_ratio_gross']} / "
            f"{section['sharpe_ratio_net']}"
        ),
        (
            f"- `max_drawdown_gross / net`：{section['max_drawdown_gross']} / "
            f"{section['max_drawdown_net']}"
        ),
        f"- `average_daily_turnover`：{section['average_daily_turnover']}",
        f"- `average_active_positions`：{section['average_active_positions']}",
        f"- `skipped_trade_count`：{section['skipped_trade_count']}",
        "",
        "当前回测假设：",
    ]
    lines.extend(_render_bullets(section["assumptions"]))
    return lines


def _render_benchmark_section(section: dict[str, Any]) -> list[str]:
    if not section["available"]:
        return [f"对应 artifact 缺失：{section['missing_message']}"]

    lines: list[str] = []
    if section["table_rows"]:
        lines.append(_render_table(section["table_rows"], section["columns"]))
    if section["missing_items"]:
        lines.append("")
        lines.append(f"缺失的可选 artifact：{', '.join(section['missing_items'])}")
    return lines


def _render_robustness_section(section: dict[str, Any]) -> list[str]:
    if not section["available"]:
        return [f"对应 artifact 缺失：{section['missing_message']}"]

    lines = [
        f"- 固定网格：{section['fixed_grid']}",
        f"- `best_config_by_sharpe_net`：{section['best_config_by_sharpe_net']}",
        f"- `best_config_by_total_return_net`：{section['best_config_by_total_return_net']}",
        f"- `simple_momentum_best_sharpe_net`：{section['simple_momentum_best_sharpe_net']}",
        (
            "- `simple_momentum_best_total_return_net`："
            f"{section['simple_momentum_best_total_return_net']}"
        ),
        (
            "- `whether_model_beats_simple_momentum_on_best_sharpe`："
            f"{section['whether_model_beats_simple_momentum_on_best_sharpe']}"
        ),
        (
            "- `whether_model_beats_simple_momentum_on_best_total_return`："
            f"{section['whether_model_beats_simple_momentum_on_best_total_return']}"
        ),
    ]
    if section["robustness_notes"]:
        lines.append(f"- `robustness_notes`：{'; '.join(section['robustness_notes'])}")
    if section["top_config_rows"]:
        lines.extend(
            [
                "",
                _render_table(section["top_config_rows"], section["top_config_columns"]),
            ]
        )
    if section["missing_items"]:
        lines.append("")
        lines.append(f"缺失的可选 artifact：{', '.join(section['missing_items'])}")
    return lines


def _render_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""

    header = "| " + " | ".join(_translate_column(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_render_cell_value(row.get(column, ""), column) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _format_artifact_names(items: list[dict[str, str]]) -> str:
    if not items:
        return "无"
    return ", ".join(item["name"] for item in items)


def _translate_column(column: str) -> str:
    mapping = {
        "metric": "指标",
        "mean": "均值",
        "std": "标准差",
        "min": "最小值",
        "max": "最大值",
        "fold_id": "fold_id",
        "test_date_start": "测试开始日",
        "test_date_end": "测试结束日",
        "strategy_name": "策略名",
        "total_return_net": "净总收益",
        "sharpe_ratio_net": "净 Sharpe",
        "max_drawdown_net": "净最大回撤",
        "annualized_return_net": "净年化收益",
        "average_daily_turnover": "日均换手",
        "top_k": "top_k",
        "holding_days": "holding_days",
        "cost_bps_per_side": "单边成本(bps)",
    }
    return mapping.get(column, column)


def _render_cell_value(value: Any, column: str) -> str:
    if column == "strategy_name":
        mapping = {
            "xgb_ranker": "xgb_ranker",
            "hist_gbr": "hist_gbr",
            "xgb_regressor": "xgb_regressor",
            "simple_momentum_top_k": "simple_momentum_top_k",
            "qqq_buy_and_hold": "qqq_buy_and_hold",
            "equal_weight_universe_buy_and_hold": "equal_weight_universe_buy_and_hold",
            "model_strategy": "model_strategy",
        }
        return mapping.get(str(value), str(value))
    return str(value)
