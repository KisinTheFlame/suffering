"""High-level service for minimal walk-forward portfolio evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from suffering.backtest.benchmarks import (
    BenchmarkBacktestResult,
    build_candidate_frame_from_features,
    build_equal_weight_universe_buy_and_hold_benchmark,
    build_qqq_buy_and_hold_benchmark,
    build_simple_momentum_top_k_benchmark,
    extract_price_frames_for_symbols,
)
from suffering.backtest.metrics import compute_backtest_metrics
from suffering.backtest.portfolio import (
    PortfolioBacktestResult,
    build_top_k_cohorts,
    simulate_overlapping_portfolio,
)
from suffering.backtest.robustness import (
    build_robustness_parameter_grid,
    build_robustness_row,
    build_robustness_summary,
    build_robustness_table,
)
from suffering.backtest.signals import load_walkforward_test_signals
from suffering.backtest.storage import BacktestStorage
from suffering.config.settings import Settings, get_settings
from suffering.data.storage import DailyDataStorage
from suffering.features.storage import DailyFeatureStorage
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.models import resolve_model_name, resolve_model_task
from suffering.training.storage import TrainingStorage


@dataclass(frozen=True)
class ModelStrategyBacktestResult:
    summary: dict[str, Any]
    daily_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    skipped_trades: pd.DataFrame


class BacktestService:
    def __init__(
        self,
        storage: BacktestStorage,
        training_storage: TrainingStorage,
        daily_data_storage: DailyDataStorage,
        feature_storage: DailyFeatureStorage,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.training_storage = training_storage
        self.daily_data_storage = daily_data_storage
        self.feature_storage = feature_storage
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BacktestService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=BacktestStorage.from_settings(resolved_settings),
            training_storage=TrainingStorage.from_settings(resolved_settings),
            daily_data_storage=DailyDataStorage.from_settings(resolved_settings),
            feature_storage=DailyFeatureStorage.from_settings(resolved_settings),
            settings=resolved_settings,
        )

    def run_walkforward_backtest(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
        resolved_model_name, resolved_top_k, resolved_holding_days, resolved_cost_bps = (
            self._resolve_backtest_parameters(
                model_name=model_name,
                top_k=top_k,
                holding_days=holding_days,
                cost_bps_per_side=cost_bps_per_side,
            )
        )
        signals = load_walkforward_test_signals(
            model_name=resolved_model_name,
            training_storage=self.training_storage,
        )
        universe_symbols = sorted(signals["symbol"].dropna().astype(str).unique().tolist())
        price_frames = self._load_price_frames(universe_symbols)
        backtest_result = self._build_model_strategy_backtest_result(
            model_name=resolved_model_name,
            signals=signals,
            price_frames=price_frames,
            top_k=resolved_top_k,
            holding_days=resolved_holding_days,
            cost_bps_per_side=resolved_cost_bps,
        )

        daily_returns_path = self.storage.write_daily_returns(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            backtest_result.daily_returns,
        )
        equity_curve_path = self.storage.write_equity_curve(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            backtest_result.equity_curve,
        )
        trades_path = self.storage.write_trades(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            backtest_result.trades,
        )
        summary_with_artifacts = {
            **backtest_result.summary,
            "artifacts": {
                "summary_path": str(
                    self.storage.summary_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "daily_returns_path": str(daily_returns_path),
                "equity_curve_path": str(equity_curve_path),
                "trades_path": str(trades_path),
            },
        }
        summary_path = self.storage.write_summary(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            summary_with_artifacts,
        )
        summary_with_artifacts["artifacts"]["summary_path"] = str(summary_path)
        return summary_with_artifacts

    def read_backtest_summary(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
        resolved_model_name, resolved_top_k, resolved_holding_days, resolved_cost_bps = (
            self._resolve_backtest_parameters(
                model_name=model_name,
                top_k=top_k,
                holding_days=holding_days,
                cost_bps_per_side=cost_bps_per_side,
            )
        )

        report = self.storage.read_summary(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )
        return {
            **report,
            "artifacts": {
                "summary_path": str(
                    self.storage.summary_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "daily_returns_path": str(
                    self.storage.daily_returns_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "equity_curve_path": str(
                    self.storage.equity_curve_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "trades_path": str(
                    self.storage.trades_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
            },
        }

    def run_backtest_comparison(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
        resolved_model_name, resolved_top_k, resolved_holding_days, resolved_cost_bps = (
            self._resolve_backtest_parameters(
                model_name=model_name,
                top_k=top_k,
                holding_days=holding_days,
                cost_bps_per_side=cost_bps_per_side,
            )
        )

        model_summary = self.storage.read_summary(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )
        model_daily_returns = self.storage.read_daily_returns(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )
        model_signals = load_walkforward_test_signals(
            model_name=resolved_model_name,
            training_storage=self.training_storage,
        )
        comparison_table_rows = [
            self._build_comparison_row(
                strategy_name=resolved_model_name,
                task_type="model",
                summary=model_summary,
            )
        ]

        universe_symbols = sorted(model_signals["symbol"].dropna().astype(str).unique().tolist())
        price_frames = self._load_price_frames(universe_symbols)
        target_dates = model_daily_returns["date"]
        benchmark_results = self._build_benchmarks(
            model_signals=model_signals,
            target_dates=target_dates,
            price_frames=price_frames,
            top_k=resolved_top_k,
            holding_days=resolved_holding_days,
            cost_bps_per_side=resolved_cost_bps,
        )

        benchmark_artifacts: dict[str, dict[str, str]] = {}
        benchmark_summaries: list[dict[str, Any]] = []
        for benchmark_result in benchmark_results:
            comparison_table_rows.append(
                self._build_comparison_row(
                    strategy_name=benchmark_result.strategy_name,
                    task_type=benchmark_result.task_type,
                    summary=benchmark_result.summary,
                )
            )
            benchmark_summaries.append(benchmark_result.summary)
            benchmark_artifacts[benchmark_result.strategy_name] = {
                "daily_returns_path": str(
                    self.storage.write_comparison_daily_returns(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                        benchmark_result.strategy_name,
                        benchmark_result.daily_returns,
                    )
                ),
                "equity_curve_path": str(
                    self.storage.write_comparison_equity_curve(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                        benchmark_result.strategy_name,
                        benchmark_result.equity_curve,
                    )
                ),
                "trades_path": str(
                    self.storage.write_comparison_trades(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                        benchmark_result.strategy_name,
                        benchmark_result.trades,
                    )
                ),
            }

        comparison_table = pd.DataFrame(comparison_table_rows).sort_values(
            ["sharpe_ratio_net", "total_return_net", "strategy_name"],
            ascending=[False, False, True],
            kind="stable",
        )
        comparison_table_path = self.storage.write_comparison_table(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            comparison_table,
        )

        benchmark_by_sharpe = max(
            benchmark_summaries,
            key=lambda item: (
                _safe_float(item.get("sharpe_ratio_net")) or float("-inf"),
                _safe_float(item.get("total_return_net")) or float("-inf"),
            ),
        )
        benchmark_by_total_return = max(
            benchmark_summaries,
            key=lambda item: (
                _safe_float(item.get("total_return_net")) or float("-inf"),
                _safe_float(item.get("sharpe_ratio_net")) or float("-inf"),
            ),
        )

        summary = {
            "model_name": resolved_model_name,
            "top_k": int(resolved_top_k),
            "holding_days": int(resolved_holding_days),
            "cost_bps_per_side": resolved_cost_bps,
            "round_trip_cost_bps": resolved_cost_bps * 2.0,
            "benchmark_count": len(benchmark_results),
            "comparison_date_start": _format_date(model_daily_returns["date"].min()),
            "comparison_date_end": _format_date(model_daily_returns["date"].max()),
            "model_strategy": {
                "strategy_name": resolved_model_name,
                "task_type": "model",
                "total_return_net": _safe_float(model_summary.get("total_return_net")),
                "sharpe_ratio_net": _safe_float(model_summary.get("sharpe_ratio_net")),
                "max_drawdown_net": _safe_float(model_summary.get("max_drawdown_net")),
                "annualized_return_net": _safe_float(model_summary.get("annualized_return_net")),
                "annualized_volatility": _safe_float(model_summary.get("annualized_volatility")),
            },
            "best_benchmark_by_sharpe_net": {
                "strategy_name": benchmark_by_sharpe["strategy_name"],
                "sharpe_ratio_net": _safe_float(benchmark_by_sharpe.get("sharpe_ratio_net")),
                "total_return_net": _safe_float(benchmark_by_sharpe.get("total_return_net")),
            },
            "best_benchmark_by_total_return_net": {
                "strategy_name": benchmark_by_total_return["strategy_name"],
                "sharpe_ratio_net": _safe_float(
                    benchmark_by_total_return.get("sharpe_ratio_net")
                ),
                "total_return_net": _safe_float(
                    benchmark_by_total_return.get("total_return_net")
                ),
            },
            "strategies": comparison_table.to_dict(orient="records"),
            "benchmark_artifacts": benchmark_artifacts,
            "artifacts": {
                "summary_path": str(
                    self.storage.comparison_summary_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "table_path": str(comparison_table_path),
            },
        }
        summary_path = self.storage.write_comparison_summary(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            summary,
        )
        summary["artifacts"]["summary_path"] = str(summary_path)
        return summary

    def read_backtest_comparison(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
        resolved_model_name, resolved_top_k, resolved_holding_days, resolved_cost_bps = (
            self._resolve_backtest_parameters(
                model_name=model_name,
                top_k=top_k,
                holding_days=holding_days,
                cost_bps_per_side=cost_bps_per_side,
            )
        )
        report = self.storage.read_comparison_summary(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )
        table = self.storage.read_comparison_table(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )
        return {
            **report,
            "table_rows": table.to_dict(orient="records"),
            "artifacts": {
                **report.get("artifacts", {}),
                "summary_path": str(
                    self.storage.comparison_summary_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
                "table_path": str(
                    self.storage.comparison_table_path(
                        resolved_model_name,
                        resolved_top_k,
                        resolved_holding_days,
                        resolved_cost_bps,
                    )
                ),
            },
        }

    def run_backtest_robustness(
        self,
        model_name: str | None = None,
        top_k_values: list[int] | None = None,
        holding_days_values: list[int] | None = None,
        cost_bps_values: list[float] | None = None,
    ) -> dict[str, Any]:
        resolved_model_name = self._resolve_backtest_model_name(model_name)
        resolved_top_k_values = top_k_values or list(self.settings.default_robustness_top_k_values)
        resolved_holding_days_values = holding_days_values or list(
            self.settings.default_robustness_holding_days_values
        )
        resolved_cost_bps_values = cost_bps_values or list(
            self.settings.default_robustness_cost_bps_values
        )
        parameter_grid = build_robustness_parameter_grid(
            top_k_values=resolved_top_k_values,
            holding_days_values=resolved_holding_days_values,
            cost_bps_values=resolved_cost_bps_values,
        )

        model_signals = load_walkforward_test_signals(
            model_name=resolved_model_name,
            training_storage=self.training_storage,
        )
        universe_symbols = sorted(model_signals["symbol"].dropna().astype(str).unique().tolist())
        price_frames = self._load_price_frames(universe_symbols)
        feature_frames = self._load_feature_frames(universe_symbols)
        momentum_feature = self.settings.default_benchmark_momentum_feature
        candidate_frame = build_candidate_frame_from_features(
            model_signals=model_signals,
            feature_frames=feature_frames,
            momentum_feature=momentum_feature,
        )
        model_task_type = resolve_model_task(model_name=resolved_model_name, settings=self.settings)

        model_rows: list[dict[str, Any]] = []
        simple_momentum_rows: list[dict[str, Any]] = []
        reference_target_dates: dict[tuple[int, float], pd.Series] = {}
        for top_k_value, holding_days_value, cost_bps_value in parameter_grid:
            model_result = self._build_model_strategy_backtest_result(
                model_name=resolved_model_name,
                signals=model_signals,
                price_frames=price_frames,
                top_k=top_k_value,
                holding_days=holding_days_value,
                cost_bps_per_side=cost_bps_value,
            )
            model_rows.append(
                build_robustness_row(
                    strategy_name="model_strategy",
                    task_type=model_task_type,
                    model_name=resolved_model_name,
                    top_k=top_k_value,
                    holding_days=holding_days_value,
                    cost_bps_per_side=cost_bps_value,
                    summary=model_result.summary,
                )
            )
            reference_target_dates.setdefault(
                (holding_days_value, cost_bps_value),
                model_result.daily_returns["date"].copy(),
            )

            momentum_result = build_simple_momentum_top_k_benchmark(
                target_dates=model_result.daily_returns["date"],
                candidate_frame=candidate_frame,
                price_frames=price_frames,
                top_k=top_k_value,
                holding_days=holding_days_value,
                cost_bps_per_side=cost_bps_value,
                momentum_feature=momentum_feature,
            )
            simple_momentum_rows.append(
                build_robustness_row(
                    strategy_name=momentum_result.strategy_name,
                    task_type=momentum_result.task_type,
                    model_name=None,
                    top_k=top_k_value,
                    holding_days=holding_days_value,
                    cost_bps_per_side=cost_bps_value,
                    summary=momentum_result.summary,
                )
            )

        qqq_symbol = self.settings.default_benchmark_symbol
        qqq_price_frame = self._load_price_frames([qqq_symbol])[qqq_symbol]
        equal_weight_price_frames = extract_price_frames_for_symbols(
            price_frames=price_frames,
            symbols=sorted(price_frames.keys()),
        )
        benchmark_reference_rows: list[dict[str, Any]] = []
        for (holding_days_value, cost_bps_value), target_dates in sorted(
            reference_target_dates.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            qqq_result = build_qqq_buy_and_hold_benchmark(
                target_dates=target_dates,
                price_frame=qqq_price_frame,
                cost_bps_per_side=cost_bps_value,
            )
            benchmark_reference_rows.append(
                build_robustness_row(
                    strategy_name=qqq_result.strategy_name,
                    task_type=qqq_result.task_type,
                    model_name=None,
                    top_k=None,
                    holding_days=holding_days_value,
                    cost_bps_per_side=cost_bps_value,
                    summary=qqq_result.summary,
                )
            )

            equal_weight_result = build_equal_weight_universe_buy_and_hold_benchmark(
                target_dates=target_dates,
                price_frames=equal_weight_price_frames,
                cost_bps_per_side=cost_bps_value,
            )
            benchmark_reference_rows.append(
                build_robustness_row(
                    strategy_name=equal_weight_result.strategy_name,
                    task_type=equal_weight_result.task_type,
                    model_name=None,
                    top_k=None,
                    holding_days=holding_days_value,
                    cost_bps_per_side=cost_bps_value,
                    summary=equal_weight_result.summary,
                )
            )

        robustness_table = build_robustness_table(
            model_rows + simple_momentum_rows + benchmark_reference_rows
        )
        robustness_table_path = self.storage.write_robustness_table(
            resolved_model_name,
            robustness_table,
        )
        summary = build_robustness_summary(
            model_name=resolved_model_name,
            top_k_values=list(dict.fromkeys(item[0] for item in parameter_grid)),
            holding_days_values=list(dict.fromkeys(item[1] for item in parameter_grid)),
            cost_bps_values=list(dict.fromkeys(item[2] for item in parameter_grid)),
            model_rows=model_rows,
            simple_momentum_rows=simple_momentum_rows,
            benchmark_reference_rows=benchmark_reference_rows,
        )
        summary_with_artifacts = {
            **summary,
            "row_count": int(len(robustness_table)),
            "strategies": robustness_table.to_dict(orient="records"),
            "artifacts": {
                "summary_path": str(self.storage.robustness_summary_path(resolved_model_name)),
                "table_path": str(robustness_table_path),
            },
        }
        summary_path = self.storage.write_robustness_summary(
            resolved_model_name,
            summary_with_artifacts,
        )
        summary_with_artifacts["artifacts"]["summary_path"] = str(summary_path)
        return summary_with_artifacts

    def read_backtest_robustness(self, model_name: str | None = None) -> dict[str, Any]:
        resolved_model_name = self._resolve_backtest_model_name(model_name)
        report = self.storage.read_robustness_summary(resolved_model_name)
        table = self.storage.read_robustness_table(resolved_model_name)
        return {
            **report,
            "table_rows": table.to_dict(orient="records"),
            "artifacts": {
                **report.get("artifacts", {}),
                "summary_path": str(self.storage.robustness_summary_path(resolved_model_name)),
                "table_path": str(self.storage.robustness_table_path(resolved_model_name)),
            },
        }

    def _load_price_frames(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        price_frames: dict[str, pd.DataFrame] = {}
        missing_symbols: list[str] = []
        for symbol in symbols:
            try:
                price_frames[symbol] = self.daily_data_storage.read_daily_data(symbol)
            except FileNotFoundError:
                missing_symbols.append(symbol)
        if missing_symbols:
            missing_display = ", ".join(missing_symbols)
            missing_command = " ".join(missing_symbols)
            raise FileNotFoundError(
                "Raw daily cache not found for symbol(s): "
                f"{missing_display}. Run `suffering data-fetch {missing_command}` first."
            )
        return price_frames

    def _load_feature_frames(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        feature_frames: dict[str, pd.DataFrame] = {}
        missing_symbols: list[str] = []
        for symbol in symbols:
            try:
                feature_frames[symbol] = self.feature_storage.read_daily_features(symbol)
            except FileNotFoundError:
                missing_symbols.append(symbol)
        if missing_symbols:
            missing_display = ", ".join(missing_symbols)
            missing_command = " ".join(missing_symbols)
            raise FileNotFoundError(
                "Feature cache not found for symbol(s): "
                f"{missing_display}. Run `suffering feature-build {missing_command}` first."
            )
        return feature_frames

    def _build_benchmarks(
        self,
        model_signals: pd.DataFrame,
        target_dates: pd.Series,
        price_frames: dict[str, pd.DataFrame],
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> list[BenchmarkBacktestResult]:
        qqq_symbol = self.settings.default_benchmark_symbol
        qqq_price_frame = self._load_price_frames([qqq_symbol])[qqq_symbol]
        equal_weight_result = build_equal_weight_universe_buy_and_hold_benchmark(
            target_dates=target_dates,
            price_frames=extract_price_frames_for_symbols(
                price_frames=price_frames,
                symbols=sorted(price_frames.keys()),
            ),
            cost_bps_per_side=cost_bps_per_side,
        )
        feature_frames = self._load_feature_frames(sorted(price_frames.keys()))
        momentum_feature = self.settings.default_benchmark_momentum_feature
        candidate_frame = build_candidate_frame_from_features(
            model_signals=model_signals,
            feature_frames=feature_frames,
            momentum_feature=momentum_feature,
        )
        momentum_result = build_simple_momentum_top_k_benchmark(
            target_dates=target_dates,
            candidate_frame=candidate_frame,
            price_frames=price_frames,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
            momentum_feature=momentum_feature,
        )
        qqq_result = build_qqq_buy_and_hold_benchmark(
            target_dates=target_dates,
            price_frame=qqq_price_frame,
            cost_bps_per_side=cost_bps_per_side,
        )
        return [qqq_result, equal_weight_result, momentum_result]

    def _build_comparison_row(
        self,
        strategy_name: str,
        task_type: str,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "strategy_name": strategy_name,
            "task_type": task_type,
            "total_return_net": _safe_float(summary.get("total_return_net")),
            "sharpe_ratio_net": _safe_float(summary.get("sharpe_ratio_net")),
            "max_drawdown_net": _safe_float(summary.get("max_drawdown_net")),
            "annualized_return_net": _safe_float(summary.get("annualized_return_net")),
            "annualized_volatility": _safe_float(summary.get("annualized_volatility")),
            "average_daily_turnover": _safe_float(summary.get("average_daily_turnover")),
            "start_date": summary.get("portfolio_date_start"),
            "end_date": summary.get("portfolio_date_end"),
        }

    def _build_model_strategy_backtest_result(
        self,
        *,
        model_name: str,
        signals: pd.DataFrame,
        price_frames: dict[str, pd.DataFrame],
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> ModelStrategyBacktestResult:
        cohorts = build_top_k_cohorts(
            signals=signals,
            top_k=top_k,
            holding_days=holding_days,
        )
        portfolio_result = simulate_overlapping_portfolio(
            cohorts=cohorts,
            price_frames=price_frames,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
        summary = self._build_model_backtest_summary(
            model_name=model_name,
            signals=signals,
            cohorts=cohorts,
            portfolio_result=portfolio_result,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
        return ModelStrategyBacktestResult(
            summary=summary,
            daily_returns=portfolio_result.daily_returns,
            equity_curve=portfolio_result.equity_curve,
            trades=portfolio_result.trades,
            skipped_trades=portfolio_result.skipped_trades,
        )

    def _build_model_backtest_summary(
        self,
        *,
        model_name: str,
        signals: pd.DataFrame,
        cohorts: pd.DataFrame,
        portfolio_result: PortfolioBacktestResult,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> dict[str, Any]:
        metrics = compute_backtest_metrics(
            daily_returns=portfolio_result.daily_returns,
            trades=portfolio_result.trades,
        )
        return {
            "model_name": model_name,
            "top_k": int(top_k),
            "holding_days": int(holding_days),
            "cost_bps_per_side": float(cost_bps_per_side),
            "round_trip_cost_bps": float(cost_bps_per_side) * 2.0,
            "signal_date_start": _format_date(signals["date"].min()),
            "signal_date_end": _format_date(signals["date"].max()),
            "signal_row_count": int(len(signals)),
            "selected_row_count": int(len(cohorts)),
            "cohort_count": int(cohorts["signal_date"].nunique()),
            "executed_cohort_count": int(portfolio_result.trades["signal_date"].nunique()),
            "trade_count": int(len(portfolio_result.trades)),
            "skipped_trade_count": int(len(portfolio_result.skipped_trades)),
            "average_selected_future_return_5d": _safe_float(
                portfolio_result.trades[FUTURE_RETURN_5D_COLUMN].mean()
            ),
            "average_cohort_future_return_5d": _safe_float(
                portfolio_result.trades.groupby("signal_date")[FUTURE_RETURN_5D_COLUMN]
                .mean()
                .mean()
            ),
            **metrics,
        }

    def _resolve_backtest_model_name(self, model_name: str | None) -> str:
        resolved_model_name = model_name or self.settings.default_backtest_model
        return resolve_model_name(model_name=resolved_model_name, settings=self.settings)

    def _resolve_backtest_parameters(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> tuple[str, int, int, float]:
        resolved_model_name = self._resolve_backtest_model_name(model_name)
        resolved_top_k = top_k if top_k is not None else self.settings.default_top_k
        resolved_holding_days = (
            holding_days if holding_days is not None else self.settings.default_holding_days
        )
        resolved_cost_bps = (
            float(cost_bps_per_side)
            if cost_bps_per_side is not None
            else float(self.settings.default_cost_bps_per_side)
        )
        return (
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
        )


def build_backtest_service(settings: Settings | None = None) -> BacktestService:
    return BacktestService.from_settings(settings=settings)


def _format_date(value: pd.Timestamp | Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
