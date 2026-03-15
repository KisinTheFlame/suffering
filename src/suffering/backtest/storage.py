"""Artifact storage helpers for minimal backtest outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from suffering.config.settings import Settings, get_settings


class BacktestStorage:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.backtests_dir = self.artifacts_dir / "backtests"
        self.comparisons_dir = self.backtests_dir / "comparisons"
        self.robustness_dir = self.backtests_dir / "robustness"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BacktestStorage":
        resolved_settings = settings or get_settings()
        return cls(artifacts_dir=resolved_settings.artifacts_dir)

    def summary_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.backtests_dir / f"{stem}_summary.json"

    def daily_returns_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.backtests_dir / f"{stem}_daily_returns.csv"

    def equity_curve_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.backtests_dir / f"{stem}_equity_curve.csv"

    def trades_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.backtests_dir / f"{stem}_trades.csv"

    def write_summary(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        summary: dict[str, Any],
    ) -> Path:
        path = self.summary_path(model_name, top_k, holding_days, cost_bps_per_side)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return path

    def read_summary(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> dict[str, Any]:
        path = self.summary_path(model_name, top_k, holding_days, cost_bps_per_side)
        if not path.exists():
            raise FileNotFoundError(
                "Backtest summary not found for "
                f"{model_name} top_k={top_k} holding_days={holding_days} "
                f"cost_bps_per_side={cost_bps_per_side}"
            )
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def write_daily_returns(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.daily_returns_path(model_name, top_k, holding_days, cost_bps_per_side),
            frame,
        )

    def read_daily_returns(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> pd.DataFrame:
        return self._read_frame(
            self.daily_returns_path(model_name, top_k, holding_days, cost_bps_per_side)
        )

    def write_equity_curve(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.equity_curve_path(model_name, top_k, holding_days, cost_bps_per_side),
            frame,
        )

    def read_equity_curve(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> pd.DataFrame:
        return self._read_frame(
            self.equity_curve_path(model_name, top_k, holding_days, cost_bps_per_side)
        )

    def write_trades(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.trades_path(model_name, top_k, holding_days, cost_bps_per_side),
            frame,
        )

    def read_trades(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> pd.DataFrame:
        return self._read_frame(
            self.trades_path(model_name, top_k, holding_days, cost_bps_per_side)
        )

    def comparison_summary_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.comparisons_dir / f"{stem}_comparison_summary.json"

    def comparison_table_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> Path:
        stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return self.comparisons_dir / f"{stem}_comparison_table.csv"

    def comparison_daily_returns_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
    ) -> Path:
        stem = self._comparison_strategy_stem(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
            strategy_name,
        )
        return self.comparisons_dir / f"{stem}_daily_returns.csv"

    def comparison_equity_curve_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
    ) -> Path:
        stem = self._comparison_strategy_stem(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
            strategy_name,
        )
        return self.comparisons_dir / f"{stem}_equity_curve.csv"

    def comparison_trades_path(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
    ) -> Path:
        stem = self._comparison_strategy_stem(
            model_name,
            top_k,
            holding_days,
            cost_bps_per_side,
            strategy_name,
        )
        return self.comparisons_dir / f"{stem}_trades.csv"

    def write_comparison_summary(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        summary: dict[str, Any],
    ) -> Path:
        path = self.comparison_summary_path(model_name, top_k, holding_days, cost_bps_per_side)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return path

    def read_comparison_summary(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> dict[str, Any]:
        path = self.comparison_summary_path(model_name, top_k, holding_days, cost_bps_per_side)
        if not path.exists():
            raise FileNotFoundError(
                "Backtest comparison summary not found for "
                f"{model_name} top_k={top_k} holding_days={holding_days} "
                f"cost_bps_per_side={cost_bps_per_side}"
            )
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def write_comparison_table(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.comparison_table_path(model_name, top_k, holding_days, cost_bps_per_side),
            frame,
        )

    def read_comparison_table(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> pd.DataFrame:
        return self._read_frame(
            self.comparison_table_path(model_name, top_k, holding_days, cost_bps_per_side)
        )

    def write_comparison_daily_returns(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.comparison_daily_returns_path(
                model_name,
                top_k,
                holding_days,
                cost_bps_per_side,
                strategy_name,
            ),
            frame,
        )

    def write_comparison_equity_curve(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.comparison_equity_curve_path(
                model_name,
                top_k,
                holding_days,
                cost_bps_per_side,
                strategy_name,
            ),
            frame,
        )

    def write_comparison_trades(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
        frame: pd.DataFrame,
    ) -> Path:
        return self._write_frame(
            self.comparison_trades_path(
                model_name,
                top_k,
                holding_days,
                cost_bps_per_side,
                strategy_name,
            ),
            frame,
        )

    def robustness_summary_path(self, model_name: str) -> Path:
        return self.robustness_dir / f"{model_name}_robustness_summary.json"

    def robustness_table_path(self, model_name: str) -> Path:
        return self.robustness_dir / f"{model_name}_robustness_table.csv"

    def write_robustness_summary(self, model_name: str, summary: dict[str, Any]) -> Path:
        path = self.robustness_summary_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return path

    def read_robustness_summary(self, model_name: str) -> dict[str, Any]:
        path = self.robustness_summary_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Backtest robustness summary not found for {model_name}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def write_robustness_table(self, model_name: str, frame: pd.DataFrame) -> Path:
        return self._write_frame(self.robustness_table_path(model_name), frame)

    def read_robustness_table(self, model_name: str) -> pd.DataFrame:
        return self._read_frame(self.robustness_table_path(model_name))

    def _write_frame(self, path: Path, frame: pd.DataFrame) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        output = frame.copy()
        for column in output.columns:
            if "date" in column:
                output[column] = pd.to_datetime(output[column]).dt.tz_localize(None)
        output.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Backtest artifact not found: {path}")
        date_columns = [column for column in pd.read_csv(path, nrows=0).columns if "date" in column]
        if not date_columns:
            return pd.read_csv(path)
        return pd.read_csv(path, parse_dates=date_columns)

    def _backtest_stem(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
    ) -> str:
        round_trip_cost_bps = float(cost_bps_per_side) * 2.0
        if round_trip_cost_bps.is_integer():
            cost_suffix = str(int(round_trip_cost_bps))
        else:
            cost_suffix = str(round_trip_cost_bps).replace(".", "p")
        return f"{model_name}_top{top_k}_h{holding_days}_cost{cost_suffix}"

    def _comparison_strategy_stem(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost_bps_per_side: float,
        strategy_name: str,
    ) -> str:
        base_stem = self._backtest_stem(model_name, top_k, holding_days, cost_bps_per_side)
        return f"{base_stem}_{strategy_name}"
