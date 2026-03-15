"""Helpers for running and collecting a full remote research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from suffering.infra.remote_workflow import build_remote_cli_command


@dataclass(frozen=True)
class RemotePipelineSpec:
    model_name: str = "xgb_ranker"
    top_k: int = 5
    holding_days: int = 5
    cost_bps_per_side: float = 5.0
    include_robustness: bool = False


def build_remote_pipeline_cli_sequences(spec: RemotePipelineSpec) -> list[list[str]]:
    cost_text = str(float(spec.cost_bps_per_side))
    commands = [
        ["train-walkforward", "--model", spec.model_name],
        [
            "backtest-walkforward",
            "--model",
            spec.model_name,
            "--top-k",
            str(spec.top_k),
            "--holding-days",
            str(spec.holding_days),
            "--cost-bps-per-side",
            cost_text,
        ],
        [
            "backtest-compare",
            "--model",
            spec.model_name,
            "--top-k",
            str(spec.top_k),
            "--holding-days",
            str(spec.holding_days),
            "--cost-bps-per-side",
            cost_text,
        ],
    ]
    if spec.include_robustness:
        commands.append(["backtest-robustness", "--model", spec.model_name])
    commands.append(
        [
            "report-generate",
            "--model",
            spec.model_name,
            "--top-k",
            str(spec.top_k),
            "--holding-days",
            str(spec.holding_days),
            "--cost-bps-per-side",
            cost_text,
        ]
    )
    return commands


def build_remote_pipeline_command(spec: RemotePipelineSpec) -> str:
    return " && ".join(
        build_remote_cli_command(command_args)
        for command_args in build_remote_pipeline_cli_sequences(spec)
    )


def build_pipeline_bundle_name(spec: RemotePipelineSpec) -> str:
    return _backtest_stem(
        spec.model_name,
        spec.top_k,
        spec.holding_days,
        spec.cost_bps_per_side,
    )


def build_pipeline_artifact_relative_paths(spec: RemotePipelineSpec) -> list[PurePosixPath]:
    stem = build_pipeline_bundle_name(spec)
    paths = [
        PurePosixPath("reports") / f"{spec.model_name}_walkforward_summary.json",
        PurePosixPath("reports") / f"{spec.model_name}_walkforward_folds.csv",
        PurePosixPath("predictions") / f"{spec.model_name}_walkforward_test_predictions.csv",
        PurePosixPath("backtests") / f"{stem}_summary.json",
        PurePosixPath("backtests") / f"{stem}_daily_returns.csv",
        PurePosixPath("backtests") / f"{stem}_equity_curve.csv",
        PurePosixPath("backtests") / f"{stem}_trades.csv",
        PurePosixPath("backtests") / "comparisons" / f"{stem}_comparison_summary.json",
        PurePosixPath("backtests") / "comparisons" / f"{stem}_comparison_table.csv",
        PurePosixPath("reports") / "research" / f"{spec.model_name}_research_report.md",
    ]
    if spec.include_robustness:
        paths.extend(
            [
                PurePosixPath("backtests")
                / "robustness"
                / f"{spec.model_name}_robustness_summary.json",
                PurePosixPath("backtests")
                / "robustness"
                / f"{spec.model_name}_robustness_table.csv",
            ]
        )
    return paths


def build_local_pipeline_output_dir(
    spec: RemotePipelineSpec,
    *,
    base_dir: Path | str = Path("artifacts") / "remote_pipeline",
) -> Path:
    return Path(base_dir) / build_pipeline_bundle_name(spec)


def _backtest_stem(
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
