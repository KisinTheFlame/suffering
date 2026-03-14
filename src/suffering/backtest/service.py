"""High-level service for minimal walk-forward portfolio evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from suffering.backtest.metrics import compute_backtest_metrics
from suffering.backtest.portfolio import (
    build_top_k_cohorts,
    simulate_overlapping_portfolio,
)
from suffering.backtest.signals import load_walkforward_test_signals
from suffering.backtest.storage import BacktestStorage
from suffering.config.settings import Settings, get_settings
from suffering.data.storage import DailyDataStorage
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.training.models import resolve_model_name
from suffering.training.storage import TrainingStorage


class BacktestService:
    def __init__(
        self,
        storage: BacktestStorage,
        training_storage: TrainingStorage,
        daily_data_storage: DailyDataStorage,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.training_storage = training_storage
        self.daily_data_storage = daily_data_storage
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BacktestService":
        resolved_settings = settings or get_settings()
        return cls(
            storage=BacktestStorage.from_settings(resolved_settings),
            training_storage=TrainingStorage.from_settings(resolved_settings),
            daily_data_storage=DailyDataStorage.from_settings(resolved_settings),
            settings=resolved_settings,
        )

    def run_walkforward_backtest(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, Any]:
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

        signals = load_walkforward_test_signals(
            model_name=resolved_model_name,
            training_storage=self.training_storage,
        )
        cohorts = build_top_k_cohorts(
            signals=signals,
            top_k=resolved_top_k,
            holding_days=resolved_holding_days,
        )
        price_frames = self._load_price_frames(
            sorted(cohorts["symbol"].dropna().astype(str).unique().tolist())
        )
        portfolio_result = simulate_overlapping_portfolio(
            cohorts=cohorts,
            price_frames=price_frames,
            holding_days=resolved_holding_days,
            cost_bps_per_side=resolved_cost_bps,
        )
        metrics = compute_backtest_metrics(
            daily_returns=portfolio_result.daily_returns,
            trades=portfolio_result.trades,
        )

        summary = {
            "model_name": resolved_model_name,
            "top_k": int(resolved_top_k),
            "holding_days": int(resolved_holding_days),
            "cost_bps_per_side": resolved_cost_bps,
            "round_trip_cost_bps": resolved_cost_bps * 2.0,
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

        daily_returns_path = self.storage.write_daily_returns(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            portfolio_result.daily_returns,
        )
        equity_curve_path = self.storage.write_equity_curve(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            portfolio_result.equity_curve,
        )
        trades_path = self.storage.write_trades(
            resolved_model_name,
            resolved_top_k,
            resolved_holding_days,
            resolved_cost_bps,
            portfolio_result.trades,
        )
        summary_with_artifacts = {
            **summary,
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
            raise FileNotFoundError(
                "Raw daily cache not found for symbol(s): "
                f"{missing_display}. Run `suffering data-fetch {' '.join(missing_symbols)}` first."
            )
        return price_frames

    def _resolve_backtest_model_name(self, model_name: str | None) -> str:
        resolved_model_name = model_name or self.settings.default_backtest_model
        return resolve_model_name(model_name=resolved_model_name, settings=self.settings)


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
