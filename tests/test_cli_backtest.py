from pathlib import Path

from suffering.cli import main


class FakeBacktestService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_paths(
        self,
        model_name: str,
        top_k: int,
        holding_days: int,
        cost: float,
    ) -> dict[str, str]:
        stem = f"{model_name}_top{top_k}_h{holding_days}_cost{int(cost * 2)}"
        summary_path = self.root_dir / f"{stem}_summary.json"
        daily_path = self.root_dir / f"{stem}_daily_returns.csv"
        equity_path = self.root_dir / f"{stem}_equity_curve.csv"
        trades_path = self.root_dir / f"{stem}_trades.csv"
        for path in (summary_path, daily_path, equity_path, trades_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return {
            "summary_path": str(summary_path),
            "daily_returns_path": str(daily_path),
            "equity_curve_path": str(equity_path),
            "trades_path": str(trades_path),
        }

    def run_walkforward_backtest(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, object]:
        resolved_model_name = model_name or "xgb_ranker"
        resolved_top_k = top_k or 5
        resolved_holding_days = holding_days or 5
        resolved_cost = cost_bps_per_side or 5.0
        return {
            "model_name": resolved_model_name,
            "top_k": resolved_top_k,
            "holding_days": resolved_holding_days,
            "cost_bps_per_side": resolved_cost,
            "round_trip_cost_bps": resolved_cost * 2,
            "signal_date_start": "2024-01-02",
            "signal_date_end": "2024-01-10",
            "portfolio_date_start": "2024-01-03",
            "portfolio_date_end": "2024-01-17",
            "trade_count": 12,
            "skipped_trade_count": 1,
            "total_return_gross": 0.12,
            "total_return_net": 0.10,
            "sharpe_ratio_gross": 1.2,
            "sharpe_ratio_net": 1.0,
            "max_drawdown_gross": -0.05,
            "max_drawdown_net": -0.06,
            "artifacts": self._artifact_paths(
                resolved_model_name,
                resolved_top_k,
                resolved_holding_days,
                resolved_cost,
            ),
        }

    def read_backtest_summary(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, object]:
        resolved_model_name = model_name or "xgb_ranker"
        if resolved_model_name == "missing":
            raise FileNotFoundError
        return self.run_walkforward_backtest(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )

    def run_backtest_comparison(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, object]:
        resolved_model_name = model_name or "xgb_ranker"
        if resolved_model_name == "missing_compare":
            raise FileNotFoundError("Backtest summary not found")

        artifacts = self._artifact_paths(
            resolved_model_name,
            top_k or 5,
            holding_days or 5,
            cost_bps_per_side or 5.0,
        )
        comparison_summary_path = self.root_dir / "comparison_summary.json"
        comparison_table_path = self.root_dir / "comparison_table.csv"
        comparison_summary_path.touch()
        comparison_table_path.touch()
        return {
            "model_name": resolved_model_name,
            "top_k": top_k or 5,
            "holding_days": holding_days or 5,
            "cost_bps_per_side": cost_bps_per_side or 5.0,
            "benchmark_count": 3,
            "comparison_date_start": "2024-01-03",
            "comparison_date_end": "2024-01-09",
            "model_strategy": {
                "total_return_net": 0.10,
                "sharpe_ratio_net": 1.0,
                "max_drawdown_net": -0.06,
            },
            "best_benchmark_by_sharpe_net": {
                "strategy_name": "simple_momentum_top_k",
                "sharpe_ratio_net": 1.1,
            },
            "best_benchmark_by_total_return_net": {
                "strategy_name": "qqq_buy_and_hold",
                "total_return_net": 0.12,
            },
            "artifacts": {
                "summary_path": str(comparison_summary_path),
                "table_path": str(comparison_table_path),
            },
            "benchmark_artifacts": {
                "simple_momentum_top_k": artifacts,
            },
        }

    def read_backtest_comparison(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        holding_days: int | None = None,
        cost_bps_per_side: float | None = None,
    ) -> dict[str, object]:
        resolved_model_name = model_name or "xgb_ranker"
        if resolved_model_name == "missing_compare":
            raise FileNotFoundError
        report = self.run_backtest_comparison(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
        return {
            **report,
            "table_rows": [
                {
                    "strategy_name": "simple_momentum_top_k",
                    "task_type": "benchmark",
                    "total_return_net": 0.11,
                    "sharpe_ratio_net": 1.1,
                    "max_drawdown_net": -0.05,
                    "annualized_return_net": 0.2,
                    "annualized_volatility": 0.18,
                    "average_daily_turnover": 0.4,
                    "start_date": "2024-01-03",
                    "end_date": "2024-01-09",
                }
            ],
        }


def test_backtest_walkforward_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_backtest_service",
        lambda settings=None: FakeBacktestService(tmp_path / "backtests"),
    )

    exit_code = main(["backtest-walkforward", "--model", "xgb_ranker", "--top-k", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_ranker" in captured.out
    assert "top_k: 3" in captured.out
    assert "gross_total_return: 0.120000" in captured.out
    assert "daily_returns_path:" in captured.out


def test_backtest_show_command_reports_missing_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_backtest_service",
        lambda settings=None: FakeBacktestService(tmp_path / "backtests"),
    )

    exit_code = main(["backtest-show", "--model", "missing"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "backtest summary not found" in captured.out


def test_backtest_compare_command_can_be_called(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_backtest_service",
        lambda settings=None: FakeBacktestService(tmp_path / "backtests"),
    )

    exit_code = main(["backtest-compare", "--model", "xgb_ranker", "--top-k", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model: xgb_ranker" in captured.out
    assert "benchmark_count: 3" in captured.out
    assert "comparison_summary_path:" in captured.out


def test_backtest_compare_show_command_reports_missing_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "suffering.cli.build_backtest_service",
        lambda settings=None: FakeBacktestService(tmp_path / "backtests"),
    )

    exit_code = main(["backtest-compare-show", "--model", "missing_compare"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "backtest comparison not found" in captured.out
