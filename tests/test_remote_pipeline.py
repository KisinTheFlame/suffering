from pathlib import Path, PurePosixPath

from suffering.infra import (
    RemotePipelineSpec,
    build_local_pipeline_output_dir,
    build_pipeline_artifact_relative_paths,
    build_pipeline_bundle_name,
    build_remote_pipeline_cli_sequences,
    build_remote_pipeline_command,
)


def test_build_remote_pipeline_cli_sequences_orders_core_research_steps() -> None:
    spec = RemotePipelineSpec(
        model_name="xgb_ranker",
        top_k=5,
        holding_days=5,
        cost_bps_per_side=5.0,
    )

    commands = build_remote_pipeline_cli_sequences(spec)

    assert commands == [
        ["train-walkforward", "--model", "xgb_ranker"],
        [
            "backtest-walkforward",
            "--model",
            "xgb_ranker",
            "--top-k",
            "5",
            "--holding-days",
            "5",
            "--cost-bps-per-side",
            "5.0",
        ],
        [
            "backtest-compare",
            "--model",
            "xgb_ranker",
            "--top-k",
            "5",
            "--holding-days",
            "5",
            "--cost-bps-per-side",
            "5.0",
        ],
        [
            "report-generate",
            "--model",
            "xgb_ranker",
            "--top-k",
            "5",
            "--holding-days",
            "5",
            "--cost-bps-per-side",
            "5.0",
        ],
    ]


def test_build_remote_pipeline_command_joins_commands_with_and() -> None:
    spec = RemotePipelineSpec(model_name="xgb_ranker", top_k=3, holding_days=10)

    command = build_remote_pipeline_command(spec)

    assert "uv run suffering train-walkforward --model xgb_ranker" in command
    assert "uv run suffering backtest-walkforward --model xgb_ranker --top-k 3" in command
    assert "uv run suffering report-generate --model xgb_ranker --top-k 3" in command
    assert " && " in command


def test_build_pipeline_artifact_relative_paths_includes_report_and_backtest_outputs() -> None:
    spec = RemotePipelineSpec(
        model_name="xgb_ranker",
        top_k=5,
        holding_days=5,
        cost_bps_per_side=5.0,
        include_robustness=True,
    )

    paths = build_pipeline_artifact_relative_paths(spec)

    assert paths == [
        PurePosixPath("reports/xgb_ranker_walkforward_summary.json"),
        PurePosixPath("reports/xgb_ranker_walkforward_folds.csv"),
        PurePosixPath("predictions/xgb_ranker_walkforward_test_predictions.csv"),
        PurePosixPath("backtests/xgb_ranker_top5_h5_cost10_summary.json"),
        PurePosixPath("backtests/xgb_ranker_top5_h5_cost10_daily_returns.csv"),
        PurePosixPath("backtests/xgb_ranker_top5_h5_cost10_equity_curve.csv"),
        PurePosixPath("backtests/xgb_ranker_top5_h5_cost10_trades.csv"),
        PurePosixPath("backtests/comparisons/xgb_ranker_top5_h5_cost10_comparison_summary.json"),
        PurePosixPath("backtests/comparisons/xgb_ranker_top5_h5_cost10_comparison_table.csv"),
        PurePosixPath("reports/research/xgb_ranker_research_report.md"),
        PurePosixPath("backtests/robustness/xgb_ranker_robustness_summary.json"),
        PurePosixPath("backtests/robustness/xgb_ranker_robustness_table.csv"),
    ]


def test_build_local_pipeline_output_dir_uses_backtest_stem() -> None:
    spec = RemotePipelineSpec(
        model_name="xgb_ranker",
        top_k=5,
        holding_days=5,
        cost_bps_per_side=2.5,
    )

    output_dir = build_local_pipeline_output_dir(
        spec,
        base_dir=Path("artifacts") / "remote_pipeline",
    )

    assert build_pipeline_bundle_name(spec) == "xgb_ranker_top5_h5_cost5"
    assert output_dir == Path("artifacts/remote_pipeline/xgb_ranker_top5_h5_cost5")
