"""Service layer for building minimal markdown research reports from artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd

from suffering.backtest.storage import BacktestStorage
from suffering.config.settings import Settings, get_settings
from suffering.reports.markdown_report import render_markdown_report
from suffering.reports.storage import ReportStorage
from suffering.training.models import resolve_model_name, resolve_model_task
from suffering.training.storage import TrainingStorage

WALKFORWARD_METRIC_NAMES = [
    "daily_rank_ic_mean",
    "daily_rank_ic_std",
    "top_5_mean_future_return",
    "top_10_mean_future_return",
    "ndcg_at_5_mean",
]

BENCHMARK_STRATEGY_ORDER = [
    "model_strategy",
    "simple_momentum_top_k",
    "qqq_buy_and_hold",
    "equal_weight_universe_buy_and_hold",
]


class ReportService:
    def __init__(
        self,
        storage: ReportStorage,
        training_storage: TrainingStorage,
        backtest_storage: BacktestStorage,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.training_storage = training_storage
        self.backtest_storage = backtest_storage
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ReportService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=ReportStorage.from_settings(resolved_settings),
            training_storage=TrainingStorage.from_settings(resolved_settings),
            backtest_storage=BacktestStorage.from_settings(resolved_settings),
            settings=resolved_settings,
        )

    def generate_research_report(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
        resolved_model_name = resolve_model_name(
            model_name or self.settings.default_report_model,
            settings=self.settings,
        )
        resolved_top_k = top_k or self.settings.default_report_top_k
        resolved_holding_days = holding_days or self.settings.default_report_holding_days
        resolved_cost = (
            float(cost_bps_per_side)
            if cost_bps_per_side is not None
            else float(self.settings.default_report_cost_bps_per_side)
        )

        artifact_bundle = self._load_artifact_bundle(
            model_name=resolved_model_name,
            top_k=resolved_top_k,
            holding_days=resolved_holding_days,
            cost_bps_per_side=resolved_cost,
        )
        if not artifact_bundle["available_artifacts"]:
            raise FileNotFoundError(
                "没有找到可用于生成报告的 artifact。请先至少生成 "
                "`train-walkforward`、`backtest-walkforward`、"
                "`backtest-compare` 或 `backtest-robustness` 之一。"
            )

        context = self._build_report_context(
            model_name=resolved_model_name,
            top_k=resolved_top_k,
            holding_days=resolved_holding_days,
            cost_bps_per_side=resolved_cost,
            artifact_bundle=artifact_bundle,
        )
        markdown_text = render_markdown_report(context)
        report_path = self.storage.write_research_report(resolved_model_name, markdown_text)

        available_sections = [
            name
            for name in (
                "walkforward",
                "backtest",
                "benchmark_comparison",
                "robustness",
            )
            if context[name]["available"]
        ]
        missing_sections = [
            name
            for name in (
                "walkforward",
                "backtest",
                "benchmark_comparison",
                "robustness",
            )
            if not context[name]["available"]
        ]
        return {
            "model_name": resolved_model_name,
            "task_type": context["metadata"]["task_type"],
            "report_path": str(report_path),
            "available_artifacts": artifact_bundle["available_artifacts"],
            "missing_artifacts": artifact_bundle["missing_artifacts"],
            "available_sections": available_sections,
            "missing_sections": missing_sections,
        }

    def read_research_report(self, model_name: str | None = None) -> dict[str, str]:
        resolved_model_name = resolve_model_name(
            model_name or self.settings.default_report_model,
            settings=self.settings,
        )
        return {
            "model_name": resolved_model_name,
            "report_path": str(self.storage.research_report_path(resolved_model_name)),
            "content": self.storage.read_research_report(resolved_model_name),
        }

    def _load_artifact_bundle(
        self,
        *,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> dict[str, Any]:
        available_artifacts: list[dict[str, str]] = []
        missing_artifacts: list[dict[str, str]] = []

        walkforward_summary_path = self.training_storage.walkforward_summary_path(model_name)
        walkforward_summary = self._load_optional_json(
            name="walkforward_summary",
            path=walkforward_summary_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )
        walkforward_folds_path = self.training_storage.walkforward_folds_path(model_name)
        walkforward_folds = self._load_optional_frame(
            name="walkforward_folds",
            path=walkforward_folds_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )
        walkforward_predictions_path = self.training_storage.walkforward_predictions_path(
            model_name
        )
        walkforward_predictions = self._load_optional_frame(
            name="walkforward_predictions",
            path=walkforward_predictions_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )

        backtest_summary_path = self.backtest_storage.summary_path(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
        )
        backtest_summary = self._load_optional_json(
            name="backtest_summary",
            path=backtest_summary_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )

        comparison_summary_path = self.backtest_storage.comparison_summary_path(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
        )
        comparison_summary = self._load_optional_json(
            name="comparison_summary",
            path=comparison_summary_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )
        comparison_table_path = self.backtest_storage.comparison_table_path(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
        )
        comparison_table = self._load_optional_frame(
            name="comparison_table",
            path=comparison_table_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )

        robustness_summary_path = self.backtest_storage.robustness_summary_path(model_name)
        robustness_summary = self._load_optional_json(
            name="robustness_summary",
            path=robustness_summary_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )
        robustness_table_path = self.backtest_storage.robustness_table_path(model_name)
        robustness_table = self._load_optional_frame(
            name="robustness_table",
            path=robustness_table_path,
            available_artifacts=available_artifacts,
            missing_artifacts=missing_artifacts,
        )

        return {
            "available_artifacts": available_artifacts,
            "missing_artifacts": missing_artifacts,
            "walkforward_summary": walkforward_summary,
            "walkforward_folds": walkforward_folds,
            "walkforward_predictions": walkforward_predictions,
            "backtest_summary": backtest_summary,
            "comparison_summary": comparison_summary,
            "comparison_table": comparison_table,
            "robustness_summary": robustness_summary,
            "robustness_table": robustness_table,
        }

    def _build_report_context(
        self,
        *,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        artifact_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        walkforward_summary = artifact_bundle["walkforward_summary"]
        walkforward_folds = artifact_bundle["walkforward_folds"]
        walkforward_predictions = artifact_bundle["walkforward_predictions"]
        backtest_summary = artifact_bundle["backtest_summary"]
        comparison_summary = artifact_bundle["comparison_summary"]
        comparison_table = artifact_bundle["comparison_table"]
        robustness_summary = artifact_bundle["robustness_summary"]
        robustness_table = artifact_bundle["robustness_table"]

        task_type = (
            str(walkforward_summary.get("task_type"))
            if walkforward_summary
            else resolve_model_task(model_name=model_name, settings=self.settings)
        )
        dataset_name = (
            str(walkforward_summary.get("dataset_name"))
            if walkforward_summary
            else self.settings.default_dataset_name
        )

        comparison_rows = self._build_comparison_rows(
            model_name=model_name,
            comparison_summary=comparison_summary,
            comparison_table=comparison_table,
        )
        universe_description = self._build_universe_description(walkforward_predictions)
        date_range = self._derive_date_range(
            backtest_summary=backtest_summary,
            comparison_summary=comparison_summary,
            robustness_table=robustness_table,
            walkforward_folds=walkforward_folds,
        )
        walkforward_section = self._build_walkforward_section(
            model_name=model_name,
            walkforward_summary=walkforward_summary,
            walkforward_folds=walkforward_folds,
        )
        backtest_section = self._build_backtest_section(
            model_name=model_name,
            backtest_summary=backtest_summary,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
        benchmark_section = self._build_benchmark_section(comparison_rows)
        robustness_section = self._build_robustness_section(
            robustness_summary=robustness_summary,
            robustness_table=robustness_table,
        )

        return {
            "metadata": {
                "model_name": model_name,
                "task_type": task_type,
                "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "dataset_name": dataset_name,
                "universe_description": universe_description,
                "date_range": date_range,
                "note": "本报告仅基于 walk-forward 样本外预测结果生成。",
            },
            "available_artifacts": artifact_bundle["available_artifacts"],
            "missing_artifacts": artifact_bundle["missing_artifacts"],
            "walkforward": walkforward_section,
            "backtest": backtest_section,
            "benchmark_comparison": benchmark_section,
            "robustness": robustness_section,
        }

    def _load_optional_json(
        self,
        *,
        name: str,
        path: Path,
        available_artifacts: list[dict[str, str]],
        missing_artifacts: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        if not path.exists():
            missing_artifacts.append({"name": name, "path": str(path)})
            return None
        available_artifacts.append({"name": name, "path": str(path)})
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_optional_frame(
        self,
        *,
        name: str,
        path: Path,
        available_artifacts: list[dict[str, str]],
        missing_artifacts: list[dict[str, str]],
    ) -> pd.DataFrame | None:
        if not path.exists():
            missing_artifacts.append({"name": name, "path": str(path)})
            return None
        available_artifacts.append({"name": name, "path": str(path)})
        frame = pd.read_csv(path)
        date_columns = [column for column in frame.columns if "date" in column]
        for column in date_columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                frame[column] = parsed
        return frame

    def _build_universe_description(self, predictions: pd.DataFrame | None) -> str:
        if predictions is None or "symbol" not in predictions.columns or predictions.empty:
            return "基于当前可用 artifact 推断"

        symbols = sorted(predictions["symbol"].dropna().astype(str).unique().tolist())
        if not symbols:
            return "基于当前可用 artifact 推断"
        if len(symbols) <= 8:
            return f"来自 walk-forward 预测的 {len(symbols)} 只股票：{', '.join(symbols)}"
        preview = ", ".join(symbols[:8])
        return f"来自 walk-forward 预测的 {len(symbols)} 只股票：{preview}, ..."

    def _derive_date_range(
        self,
        *,
        backtest_summary: dict[str, Any] | None,
        comparison_summary: dict[str, Any] | None,
        robustness_table: pd.DataFrame | None,
        walkforward_folds: pd.DataFrame | None,
    ) -> str:
        if backtest_summary:
            return (
                f"{backtest_summary.get('portfolio_date_start', 'n/a')} -> "
                f"{backtest_summary.get('portfolio_date_end', 'n/a')}"
            )
        if comparison_summary:
            return (
                f"{comparison_summary.get('comparison_date_start', 'n/a')} -> "
                f"{comparison_summary.get('comparison_date_end', 'n/a')}"
            )
        if robustness_table is not None and not robustness_table.empty:
            start_date = _format_date_value(robustness_table["start_date"].dropna().min())
            end_date = _format_date_value(robustness_table["end_date"].dropna().max())
            if start_date and end_date:
                return f"{start_date} -> {end_date}"
        if walkforward_folds is not None and not walkforward_folds.empty:
            return (
                f"{_format_date_value(walkforward_folds['test_date_start'].min())} -> "
                f"{_format_date_value(walkforward_folds['test_date_end'].max())}"
            )
        return "n/a"

    def _build_walkforward_section(
        self,
        *,
        model_name: str,
        walkforward_summary: dict[str, Any] | None,
        walkforward_folds: pd.DataFrame | None,
    ) -> dict[str, Any]:
        if walkforward_summary is None:
            return {
                "available": False,
                "missing_message": str(
                    self.training_storage.walkforward_summary_path(model_name)
                ),
            }

        metrics_summary = walkforward_summary.get("test_metrics_summary", {})
        metric_rows = []
        for metric_name in WALKFORWARD_METRIC_NAMES:
            summary = metrics_summary.get(metric_name, {})
            if not summary:
                continue
            is_percentage_metric = _is_percentage_metric(metric_name)
            metric_rows.append(
                {
                    "metric": metric_name,
                    "mean": _format_optional_value(
                        summary.get("mean"),
                        percentage=is_percentage_metric,
                    ),
                    "std": _format_optional_value(
                        summary.get("std"),
                        percentage=is_percentage_metric,
                    ),
                    "min": _format_optional_value(
                        summary.get("min"),
                        percentage=is_percentage_metric,
                    ),
                    "max": _format_optional_value(
                        summary.get("max"),
                        percentage=is_percentage_metric,
                    ),
                }
            )

        fold_rows: list[dict[str, Any]] = []
        missing_items: list[str] = []
        if walkforward_folds is not None and not walkforward_folds.empty:
            fold_snapshot = walkforward_folds.copy()
            for column in ("test_date_start", "test_date_end"):
                if column in fold_snapshot.columns:
                    fold_snapshot[column] = fold_snapshot[column].map(_format_date_value)
            for column in ("daily_rank_ic_mean", "top_5_mean_future_return", "ndcg_at_5_mean"):
                if column in fold_snapshot.columns:
                    fold_snapshot[column] = fold_snapshot[column].map(
                        lambda value: _format_optional_value(
                            value,
                            percentage=column == "top_5_mean_future_return",
                        )
                    )
            fold_rows = fold_snapshot.loc[
                :,
                [
                    column
                    for column in (
                        "fold_id",
                        "test_date_start",
                        "test_date_end",
                        "daily_rank_ic_mean",
                        "top_5_mean_future_return",
                        "ndcg_at_5_mean",
                    )
                    if column in fold_snapshot.columns
                ],
            ].to_dict(orient="records")
        else:
            missing_items.append("walkforward_folds")

        return {
            "available": True,
            "fold_count": int(walkforward_summary.get("fold_count", 0)),
            "daily_rank_ic_mean": _format_optional_value(
                metrics_summary.get("daily_rank_ic_mean", {}).get("mean")
            ),
            "top_5_mean_future_return": _format_optional_value(
                metrics_summary.get("top_5_mean_future_return", {}).get("mean"),
                percentage=True,
            ),
            "top_10_mean_future_return": _format_optional_value(
                metrics_summary.get("top_10_mean_future_return", {}).get("mean"),
                percentage=True,
            ),
            "ndcg_at_5_mean": _format_optional_value(
                metrics_summary.get("ndcg_at_5_mean", {}).get("mean")
            ),
            "notes": [str(note) for note in list(walkforward_summary.get("notes", []))],
            "metric_rows": metric_rows,
            "metric_columns": ["metric", "mean", "std", "min", "max"],
            "fold_rows": fold_rows,
            "fold_columns": [
                column
                for column in (
                    "fold_id",
                    "test_date_start",
                    "test_date_end",
                    "daily_rank_ic_mean",
                    "top_5_mean_future_return",
                    "ndcg_at_5_mean",
                )
            ],
            "missing_items": missing_items,
        }

    def _build_backtest_section(
        self,
        *,
        model_name: str,
        backtest_summary: dict[str, Any] | None,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> dict[str, Any]:
        if backtest_summary is None:
            return {
                "available": False,
                "missing_message": str(
                    self.backtest_storage.summary_path(
                        model_name,
                        top_k,
                        holding_days,
                        cost_bps_per_side,
                    )
                ),
            }

        return {
            "available": True,
            "total_return_gross": _format_optional_value(
                backtest_summary.get("total_return_gross"),
                percentage=True,
            ),
            "total_return_net": _format_optional_value(
                backtest_summary.get("total_return_net"),
                percentage=True,
            ),
            "annualized_return_gross": _format_optional_value(
                backtest_summary.get("annualized_return_gross"),
                percentage=True,
            ),
            "annualized_return_net": _format_optional_value(
                backtest_summary.get("annualized_return_net"),
                percentage=True,
            ),
            "sharpe_ratio_gross": _format_optional_value(
                backtest_summary.get("sharpe_ratio_gross")
            ),
            "sharpe_ratio_net": _format_optional_value(backtest_summary.get("sharpe_ratio_net")),
            "max_drawdown_gross": _format_optional_value(
                backtest_summary.get("max_drawdown_gross"),
                percentage=True,
            ),
            "max_drawdown_net": _format_optional_value(
                backtest_summary.get("max_drawdown_net"),
                percentage=True,
            ),
            "average_daily_turnover": _format_optional_value(
                backtest_summary.get("average_daily_turnover"),
                percentage=True,
            ),
            "average_active_positions": _format_optional_value(
                backtest_summary.get("average_active_positions")
            ),
            "skipped_trade_count": int(backtest_summary.get("skipped_trade_count", 0)),
            "assumptions": [
                f"每个信号日按 `top_k={top_k}` 选股。",
                f"持有期采用 `holding_days={holding_days}`。",
                f"交易成本采用 `cost_bps_per_side={cost_bps_per_side:.1f}` bps。",
                "启用重叠 cohort，且 cohort 内等权。",
                f"交易假设为 `t+1` 开盘买入、`t+{holding_days}` 收盘卖出。",
            ],
        }

    def _build_comparison_rows(
        self,
        *,
        model_name: str,
        comparison_summary: dict[str, Any] | None,
        comparison_table: pd.DataFrame | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if comparison_table is not None and not comparison_table.empty:
            source_rows = comparison_table.to_dict(orient="records")
        elif comparison_summary and comparison_summary.get("strategies"):
            source_rows = list(comparison_summary["strategies"])
        else:
            source_rows = []

        by_name = {
            str(row.get("strategy_name")): row
            for row in source_rows
            if row.get("strategy_name") is not None
        }

        if (
            model_name not in by_name
            and comparison_summary
            and comparison_summary.get("model_strategy")
        ):
            model_row = dict(comparison_summary["model_strategy"])
            model_row.setdefault("strategy_name", model_name)
            model_row.setdefault("task_type", "model")
            by_name[model_name] = model_row

        ordered_names = [model_name, *BENCHMARK_STRATEGY_ORDER[1:]]
        for strategy_name in ordered_names:
            row = by_name.get(strategy_name)
            if row is None:
                continue
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "total_return_net": _coerce_float(row.get("total_return_net")),
                    "sharpe_ratio_net": _coerce_float(row.get("sharpe_ratio_net")),
                    "max_drawdown_net": _coerce_float(row.get("max_drawdown_net")),
                    "annualized_return_net": _coerce_float(row.get("annualized_return_net")),
                    "average_daily_turnover": _coerce_float(row.get("average_daily_turnover")),
                }
            )
        return rows

    def _build_benchmark_section(self, comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not comparison_rows:
            return {
                "available": False,
                "missing_message": "未找到 comparison summary/table",
            }

        table_rows = [
            {
                "strategy_name": row["strategy_name"],
                "total_return_net": _format_optional_value(
                    row.get("total_return_net"),
                    percentage=True,
                ),
                "sharpe_ratio_net": _format_optional_value(row.get("sharpe_ratio_net")),
                "max_drawdown_net": _format_optional_value(
                    row.get("max_drawdown_net"),
                    percentage=True,
                ),
                "annualized_return_net": _format_optional_value(
                    row.get("annualized_return_net"),
                    percentage=True,
                ),
                "average_daily_turnover": _format_optional_value(
                    row.get("average_daily_turnover"),
                    percentage=True,
                ),
            }
            for row in comparison_rows
        ]

        return {
            "available": True,
            "columns": [
                "strategy_name",
                "total_return_net",
                "sharpe_ratio_net",
                "max_drawdown_net",
                "annualized_return_net",
                "average_daily_turnover",
            ],
            "table_rows": table_rows,
            "missing_items": [
                name
                for name in (
                    "simple_momentum_top_k",
                    "qqq_buy_and_hold",
                    "equal_weight_universe_buy_and_hold",
                )
                if _find_row(comparison_rows, name) is None
            ],
        }

    def _build_robustness_section(
        self,
        *,
        robustness_summary: dict[str, Any] | None,
        robustness_table: pd.DataFrame | None,
    ) -> dict[str, Any]:
        if robustness_summary is None:
            return {
                "available": False,
                "missing_message": "未找到 robustness summary",
            }

        top_config_rows: list[dict[str, Any]] = []
        if robustness_table is not None and not robustness_table.empty:
            model_rows = robustness_table.loc[
                robustness_table["strategy_name"] == "model_strategy"
            ].copy()
            if not model_rows.empty:
                model_rows = model_rows.sort_values(
                    ["sharpe_ratio_net", "total_return_net", "top_k", "holding_days"],
                    ascending=[False, False, True, True],
                    kind="stable",
                ).head(5)
                top_config_rows = [
                    {
                        "top_k": _stringify_int(row.get("top_k")),
                        "holding_days": _stringify_int(row.get("holding_days")),
                        "cost_bps_per_side": _format_optional_value(row.get("cost_bps_per_side")),
                        "sharpe_ratio_net": _format_optional_value(row.get("sharpe_ratio_net")),
                        "total_return_net": _format_optional_value(
                            row.get("total_return_net"),
                            percentage=True,
                        ),
                        "max_drawdown_net": _format_optional_value(
                            row.get("max_drawdown_net"),
                            percentage=True,
                        ),
                    }
                    for row in model_rows.to_dict(orient="records")
                ]

        return {
            "available": True,
            "fixed_grid": (
                f"top_k={robustness_summary.get('top_k_values', [])}, "
                f"holding_days={robustness_summary.get('holding_days_values', [])}, "
                f"cost_bps_per_side={robustness_summary.get('cost_bps_values', [])}"
            ),
            "best_config_by_sharpe_net": _format_config_snapshot(
                robustness_summary.get("best_config_by_sharpe_net"),
                metric_name="sharpe_ratio_net",
            ),
            "best_config_by_total_return_net": _format_config_snapshot(
                robustness_summary.get("best_config_by_total_return_net"),
                metric_name="total_return_net",
                percentage=True,
            ),
            "simple_momentum_best_sharpe_net": _format_config_snapshot(
                robustness_summary.get("simple_momentum_best_sharpe_net"),
                metric_name="sharpe_ratio_net",
            ),
            "simple_momentum_best_total_return_net": _format_config_snapshot(
                robustness_summary.get("simple_momentum_best_total_return_net"),
                metric_name="total_return_net",
                percentage=True,
            ),
            "whether_model_beats_simple_momentum_on_best_sharpe": str(
                robustness_summary.get("whether_model_beats_simple_momentum_on_best_sharpe")
            ),
            "whether_model_beats_simple_momentum_on_best_total_return": str(
                robustness_summary.get("whether_model_beats_simple_momentum_on_best_total_return")
            ),
            "robustness_notes": [
                str(note)
                for note in list(robustness_summary.get("robustness_notes", []))
            ],
            "top_config_rows": top_config_rows,
            "top_config_columns": [
                "top_k",
                "holding_days",
                "cost_bps_per_side",
                "sharpe_ratio_net",
                "total_return_net",
                "max_drawdown_net",
            ],
            "missing_items": ["robustness_table"] if robustness_table is None else [],
        }

    def _build_executive_summary(
        self,
        *,
        walkforward_summary: dict[str, Any] | None,
        backtest_summary: dict[str, Any] | None,
        comparison_rows: list[dict[str, Any]],
        robustness_summary: dict[str, Any] | None,
    ) -> list[str]:
        bullets: list[str] = []

        if backtest_summary:
            net_sharpe = _format_optional_value(backtest_summary.get("sharpe_ratio_net"))
            net_return = _format_optional_value(
                backtest_summary.get("total_return_net"),
                percentage=True,
            )
            max_drawdown = _format_optional_value(
                backtest_summary.get("max_drawdown_net"),
                percentage=True,
            )
            bullets.append(
                f"回测净 Sharpe 为 {net_sharpe}，净收益为 {net_return}，"
                f"净最大回撤为 {max_drawdown}。"
            )

        if walkforward_summary:
            metrics_summary = walkforward_summary.get("test_metrics_summary", {})
            daily_rank_ic_mean = _coerce_float(
                metrics_summary.get("daily_rank_ic_mean", {}).get("mean")
            )
            top_5_mean_future_return = _coerce_float(
                metrics_summary.get("top_5_mean_future_return", {}).get("mean")
            )
            if daily_rank_ic_mean is not None:
                if daily_rank_ic_mean > 0.02:
                    signal_text = "walk-forward 的 rank IC 明显为正"
                elif daily_rank_ic_mean >= 0:
                    signal_text = "walk-forward 的 rank IC 仅小幅为正"
                else:
                    signal_text = "walk-forward 的 rank IC 仍为负"
                bullets.append(
                    f"{signal_text}（{_format_optional_value(daily_rank_ic_mean)}）；"
                    f"`top_5_mean_future_return` 为 "
                    f"{_format_optional_value(top_5_mean_future_return, percentage=True)}。"
                )

        model_row = comparison_rows[0] if comparison_rows else None
        momentum_row = _find_row(comparison_rows, "simple_momentum_top_k")
        qqq_row = _find_row(comparison_rows, "qqq_buy_and_hold")
        equal_weight_row = _find_row(comparison_rows, "equal_weight_universe_buy_and_hold")
        if model_row and momentum_row:
            bullets.append(
                self._describe_relative_performance(
                    model_row=model_row,
                    reference_row=momentum_row,
                    reference_name="simple momentum",
                )
            )
        elif comparison_rows:
            bullets.append(
                "基准对比结果只有部分可用，因此当前对相对优势的判断仍不完整。"
            )

        if model_row and qqq_row and equal_weight_row:
            best_reference = max(
                [qqq_row, equal_weight_row],
                key=lambda row: (
                    _metric_sort_value(row.get("sharpe_ratio_net")),
                    _metric_sort_value(row.get("total_return_net")),
                ),
            )
            bullets.append(
                self._describe_relative_performance(
                    model_row=model_row,
                    reference_row=best_reference,
                    reference_name=best_reference["strategy_name"],
                )
            )

        if robustness_summary:
            notes = list(robustness_summary.get("robustness_notes", []))
            if any("narrow configuration" in note for note in notes):
                bullets.append(
                    "稳健性结果显示，该策略可能只在较窄的参数区间内有效。"
                )
            elif any("costs increase" in note for note in notes):
                bullets.append(
                    "稳健性结果显示，交易成本是当前策略的一阶风险因素。"
                )
            else:
                bullets.append(
                    "在当前默认小网格上，稳健性方向上尚算一致，"
                    "但这仍不能替代更广泛的研究验证。"
                )
        else:
            bullets.append(
                "稳健性 artifact 缺失，因此参数稳定性目前仍未验证。"
            )

        if backtest_summary:
            max_drawdown_net = _coerce_float(backtest_summary.get("max_drawdown_net"))
            average_daily_turnover = _coerce_float(
                backtest_summary.get("average_daily_turnover")
            )
            if max_drawdown_net is not None and max_drawdown_net <= -0.35:
                bullets.append(
                    "当前回撤仍然偏大，优先级应放在风险控制而不是继续增加模型复杂度。"
                )
            elif average_daily_turnover is not None and average_daily_turnover >= 0.30:
                bullets.append(
                    "当前换手偏高，交易成本敏感性值得优先关注。"
                )

        return _deduplicate_keep_order(bullets)[:5]

    def _build_key_caveats(
        self,
        *,
        walkforward_predictions: pd.DataFrame | None,
        walkforward_summary: dict[str, Any] | None,
        backtest_summary: dict[str, Any] | None,
        robustness_summary: dict[str, Any] | None,
    ) -> list[str]:
        caveats: list[str] = []

        symbol_count = 0
        if walkforward_predictions is not None and "symbol" in walkforward_predictions.columns:
            symbol_count = int(walkforward_predictions["symbol"].nunique(dropna=True))
        if symbol_count <= 10:
            caveats.append(
                "当前股票池仍然偏小，且主要来自本地已缓存股票。"
            )
        else:
            caveats.append(
                "当前股票池覆盖范围仍然来自本地 artifact，"
                "而不是更完整的生产级 universe。"
            )

        caveats.extend(
            [
                "当前特征仍以价量与技术面特征为主。",
                "交易成本仍采用简化的固定 bps 模型。",
                "目前仍未引入正式风险模型、行业中性或组合优化层。",
                "目前尚未对这些结果进行正式统计显著性检验。",
                "当前回测与执行假设仍不是生产级框架。",
            ]
        )

        fold_count = (
            int(walkforward_summary.get("fold_count", 0)) if walkforward_summary else 0
        )
        if robustness_summary is None or fold_count < 3:
            caveats.append(
                "结果可能仍然对当前样本期和较小验证网格较为敏感。"
            )

        if backtest_summary is not None:
            skipped_trade_count = int(backtest_summary.get("skipped_trade_count", 0))
            if skipped_trade_count > 0:
                caveats.append(
                    "部分交易因所需价格历史不完整而被跳过。"
                )

        return _deduplicate_keep_order(caveats)

    def _build_next_research_steps(
        self,
        *,
        walkforward_summary: dict[str, Any] | None,
        backtest_summary: dict[str, Any] | None,
        comparison_rows: list[dict[str, Any]],
        robustness_summary: dict[str, Any] | None,
    ) -> list[str]:
        steps: list[str] = []

        model_row = comparison_rows[0] if comparison_rows else None
        momentum_row = _find_row(comparison_rows, "simple_momentum_top_k")
        if model_row is None or momentum_row is None:
            steps.append(
                "先补齐缺失的 benchmark comparison artifacts，再判断模型是否有真实优势。"
            )
        else:
            model_sharpe = _metric_sort_value(model_row.get("sharpe_ratio_net"))
            momentum_sharpe = _metric_sort_value(momentum_row.get("sharpe_ratio_net"))
            if model_sharpe <= momentum_sharpe + 0.05:
                steps.append(
                    "模型相对 simple momentum 的优势仍弱，"
                    "建议先回头审视特征与标签设计，而不是继续堆模型复杂度。"
                )
            else:
                steps.append(
                    "既然相对 simple momentum 已有一定优势，"
                    "下一步更值得做的是更细的成本模型和简单风险约束。"
                )

        if robustness_summary is None:
            steps.append(
                "下一步先生成 robustness artifacts，确认当前结果不是单一参数点偶然有效。"
            )
        else:
            notes = list(robustness_summary.get("robustness_notes", []))
            if any("costs increase" in note for note in notes):
                steps.append(
                    "由于成本升高时表现会走弱，建议优先优化换手与持有期设置。"
                )
            if any("narrow configuration" in note for note in notes):
                steps.append(
                    "在相信当前最优点之前，先验证相邻 `top_k` 和 `holding_days` 配置。"
                )

        if backtest_summary is not None:
            max_drawdown_net = _coerce_float(backtest_summary.get("max_drawdown_net"))
            average_daily_turnover = _coerce_float(
                backtest_summary.get("average_daily_turnover")
            )
            if max_drawdown_net is not None and max_drawdown_net <= -0.35:
                steps.append(
                    "建议先加入简单组合约束与回撤控制，再考虑更复杂的模型。"
                )
            elif average_daily_turnover is not None and average_daily_turnover >= 0.30:
                steps.append(
                    "建议先审视换手、调仓频率和持有窗口，再扩展更多策略变体。"
                )

        if walkforward_summary is not None:
            metrics_summary = walkforward_summary.get("test_metrics_summary", {})
            daily_rank_ic_mean = _coerce_float(
                metrics_summary.get("daily_rank_ic_mean", {}).get("mean")
            )
            if daily_rank_ic_mean is not None and daily_rank_ic_mean <= 0:
                steps.append(
                    "建议先检查样本外排序信号质量，因为平均 `daily_rank_ic_mean` 仍未稳定转正。"
                )

        if len(steps) < 3:
            steps.append(
                "完成上述检查后，优先扩展更细的成本与风险诊断，而不是更大的展示界面。"
            )
        if len(steps) < 3:
            steps.append(
                "后续新增研究输出时，继续以 walk-forward 样本外 artifacts 作为统一基准。"
            )

        return _deduplicate_keep_order(steps)[:5]

    def _summarize_fold_stability(self, walkforward_folds: pd.DataFrame) -> str:
        if walkforward_folds.empty or "daily_rank_ic_mean" not in walkforward_folds.columns:
            return "当前可用 artifact 不足，无法提炼 fold 级稳定性结论。"

        fold_count = int(len(walkforward_folds))
        positive_ic_series = pd.to_numeric(
            walkforward_folds["daily_rank_ic_mean"],
            errors="coerce",
        )
        positive_ic_count = int((positive_ic_series > 0).sum())
        top_5_positive_count = 0
        if "top_5_mean_future_return" in walkforward_folds.columns:
            top_5_series = pd.to_numeric(
                walkforward_folds["top_5_mean_future_return"],
                errors="coerce",
            )
            top_5_positive_count = int(
                (top_5_series > 0).sum()
            )

        if positive_ic_count == fold_count and top_5_positive_count == fold_count:
            return "所有 fold 在 rank IC 和 top-5 收益上都保持为正。"
        if positive_ic_count >= ceil(fold_count * 0.75):
            return "大多数 fold 保持为正，但不同 fold 之间强度仍有波动。"
        if positive_ic_count == 0:
            return "所有 fold 的 rank IC 都为负，当前信号尚不稳定。"
        return "不同 fold 结果分化较大，因此稳定性仍然有限。"

    def _summarize_metric_stability(
        self,
        metrics_summary: dict[str, dict[str, Any]],
    ) -> str:
        daily_rank_ic_summary = metrics_summary.get("daily_rank_ic_mean", {})
        mean_value = _coerce_float(daily_rank_ic_summary.get("mean"))
        std_value = _coerce_float(daily_rank_ic_summary.get("std"))
        if mean_value is None or std_value is None:
            return "当前可用 artifact 不足，无法提炼 fold 级稳定性结论。"
        if mean_value > 0 and std_value <= abs(mean_value):
            return (
                "平均 fold IC 为正，但缺少更完整的 fold 数据来做更充分的稳定性判断。"
            )
        return (
            "平均 fold IC 相对波动幅度仍偏小，因此稳定性仍不确定。"
        )

    def _build_benchmark_interpretation(
        self,
        comparison_rows: list[dict[str, Any]],
    ) -> list[str]:
        model_row = comparison_rows[0] if comparison_rows else None
        if model_row is None:
            return ["comparison artifact 中缺少模型策略行。"]

        interpretation: list[str] = []
        momentum_row = _find_row(comparison_rows, "simple_momentum_top_k")
        if momentum_row is not None:
            interpretation.append(
                self._describe_relative_performance(
                    model_row=model_row,
                    reference_row=momentum_row,
                    reference_name="simple momentum",
                )
            )

        reference_candidates = [
            row
            for row in comparison_rows
            if row["strategy_name"] in {"qqq_buy_and_hold", "equal_weight_universe_buy_and_hold"}
        ]
        if reference_candidates:
            best_reference = max(
                reference_candidates,
                key=lambda row: (
                    _metric_sort_value(row.get("sharpe_ratio_net")),
                    _metric_sort_value(row.get("total_return_net")),
                ),
            )
            interpretation.append(
                self._describe_relative_performance(
                    model_row=model_row,
                    reference_row=best_reference,
                    reference_name=best_reference["strategy_name"],
                )
            )

        if momentum_row is None and not reference_candidates:
            interpretation.append(
                "基准覆盖不完整，因此当前仍难以充分判断相对表现。"
            )

        if len(interpretation) < 3:
            if momentum_row is not None:
                model_sharpe = _metric_sort_value(model_row.get("sharpe_ratio_net"))
                momentum_sharpe = _metric_sort_value(momentum_row.get("sharpe_ratio_net"))
                if model_sharpe > momentum_sharpe + 0.10:
                    interpretation.append(
                        "模型仍有继续研究价值，但前提是先确认成本和回撤敏感性。"
                    )
                elif model_sharpe >= momentum_sharpe - 0.10:
                    interpretation.append(
                        "模型目前仍接近基线策略，因此研究价值尚不算明确。"
                    )
                else:
                    interpretation.append(
                        "模型尚未明显跑赢简单基线，暂不足以支持继续增加复杂度。"
                    )

        return _deduplicate_keep_order(interpretation)[:3]

    def _describe_relative_performance(
        self,
        *,
        model_row: dict[str, Any],
        reference_row: dict[str, Any],
        reference_name: str,
    ) -> str:
        model_sharpe = _metric_sort_value(model_row.get("sharpe_ratio_net"))
        reference_sharpe = _metric_sort_value(reference_row.get("sharpe_ratio_net"))
        if model_sharpe > reference_sharpe + 0.10:
            return f"模型当前在净 Sharpe 上优于 {reference_name}。"
        if model_sharpe < reference_sharpe - 0.10:
            return f"模型当前在净 Sharpe 上落后于 {reference_name}。"
        return f"模型当前与 {reference_name} 大体接近，因此优势尚不明确。"


def build_report_service(settings: Settings | None = None) -> ReportService:
    return ReportService.from_settings(settings=settings)


def _find_row(rows: Iterable[dict[str, Any]], strategy_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("strategy_name") == strategy_name:
            return row
    return None


def _deduplicate_keep_order(values: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _metric_sort_value(value: Any) -> float:
    resolved = _coerce_float(value)
    if resolved is None:
        return float("-inf")
    return resolved


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_optional_value(value: Any, *, percentage: bool = False) -> str:
    resolved = _coerce_float(value)
    if resolved is None:
        return "暂无"
    if percentage:
        return f"{resolved:.2%}"
    return f"{resolved:.4f}"


def _format_config_snapshot(
    row: dict[str, Any] | None,
    *,
    metric_name: str,
    percentage: bool = False,
) -> str:
    if not row:
        return "暂无"
    metric_value = _format_optional_value(row.get(metric_name), percentage=percentage)
    return (
        f"top_k={_stringify_int(row.get('top_k'))}, "
        f"holding_days={_stringify_int(row.get('holding_days'))}, "
        f"cost_bps_per_side={_format_optional_value(row.get('cost_bps_per_side'))}, "
        f"{metric_name}={metric_value}"
    )


def _stringify_int(value: Any) -> str:
    resolved = _coerce_float(value)
    if resolved is None:
        return "暂无"
    return str(int(resolved))


def _is_percentage_metric(metric_name: str) -> bool:
    return "return" in metric_name


def _format_date_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _translate_report_note(note: str) -> str:
    mapping = {
        "only performs well at one narrow configuration": "仅在非常狭窄的参数配置上表现较好",
        "model remains competitive across multiple top_k values": (
            "模型在多个 top_k 设置上仍保持竞争力"
        ),
        "performance degrades materially as costs increase": "随着成本上升，表现出现明显恶化",
        "performance is directionally consistent across the small default grid": (
            "在当前默认小网格上，表现方向上基本一致"
        ),
        "no model configurations were evaluated": "没有可评估的模型配置",
        "Only one walk-forward fold was generated because unique dates are limited.": (
            "由于唯一交易日数量有限，当前只生成了一个 walk-forward fold。"
        ),
    }
    return mapping.get(note, note)
