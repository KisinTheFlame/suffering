"""Minimal command line interface for the suffering project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from suffering import PROJECT_NAME, __version__
from suffering.backtest import build_backtest_service
from suffering.config.settings import get_settings
from suffering.data import build_data_service, get_default_universe
from suffering.data.universe import resolve_symbols
from suffering.features import build_feature_service
from suffering.features.definitions import FEATURE_COLUMNS
from suffering.ranking import build_ranking_service
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN
from suffering.reports import build_report_service
from suffering.training import SUPPORTED_MODEL_NAMES, build_training_service
from suffering.training.evaluate import METRIC_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROJECT_NAME,
        description="Minimal CLI for the suffering quant research skeleton.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Inspect the local project environment.")

    data_fetch_parser = subparsers.add_parser(
        "data-fetch",
        help="Fetch daily stock data and cache it locally.",
    )
    data_fetch_parser.add_argument("symbols", nargs="*", help="Ticker symbols such as AAPL MSFT")
    data_fetch_parser.add_argument(
        "--start-date",
        help="Inclusive start date, for example 2020-01-01",
    )
    data_fetch_parser.add_argument(
        "--end-date",
        help="Inclusive end date, for example 2024-12-31",
    )

    data_show_parser = subparsers.add_parser(
        "data-show",
        help="Show cached daily stock data for one symbol.",
    )
    data_show_parser.add_argument("symbol", help="Ticker symbol such as AAPL")

    feature_build_parser = subparsers.add_parser(
        "feature-build",
        help="Build daily feature tables from cached raw data.",
    )
    feature_build_parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols such as AAPL MSFT",
    )

    feature_show_parser = subparsers.add_parser(
        "feature-show",
        help="Show cached daily features for one symbol.",
    )
    feature_show_parser.add_argument("symbol", help="Ticker symbol such as AAPL")

    label_build_parser = subparsers.add_parser(
        "label-build",
        help="Build daily labels from cached raw data.",
    )
    label_build_parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols such as AAPL MSFT",
    )

    dataset_build_parser = subparsers.add_parser(
        "dataset-build",
        help="Build a cached panel dataset from feature and label caches.",
    )
    dataset_build_parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols such as AAPL MSFT",
    )
    dataset_build_parser.add_argument(
        "--dataset-name",
        help="Dataset cache name, defaulting to the configured panel dataset name.",
    )

    dataset_show_parser = subparsers.add_parser(
        "dataset-show",
        help="Show the cached panel dataset summary.",
    )
    dataset_show_parser.add_argument(
        "--dataset-name",
        help="Dataset cache name, defaulting to the configured panel dataset name.",
    )

    train_baseline_parser = subparsers.add_parser(
        "train-baseline",
        help="Train the minimal HistGradientBoostingRegressor baseline on a cached panel dataset.",
    )
    train_baseline_parser.add_argument(
        "--dataset-name",
        help="Dataset cache name, defaulting to the configured panel dataset name.",
    )
    train_baseline_parser.add_argument(
        "--model",
        dest="model",
        help=(
            "Training model name, defaulting to the configured training model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    train_baseline_parser.add_argument(
        "--model-name",
        dest="model",
        help=argparse.SUPPRESS,
    )

    train_show_parser = subparsers.add_parser(
        "train-show",
        help="Show the most recent saved baseline training report.",
    )
    train_show_parser.add_argument(
        "--model",
        dest="model",
        help=(
            "Training model name, defaulting to the configured training model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    train_show_parser.add_argument(
        "--model-name",
        dest="model",
        help=argparse.SUPPRESS,
    )

    train_walkforward_parser = subparsers.add_parser(
        "train-walkforward",
        help="Run minimal walk-forward validation on a cached panel dataset.",
    )
    train_walkforward_parser.add_argument(
        "--dataset-name",
        help="Dataset cache name, defaulting to the configured panel dataset name.",
    )
    train_walkforward_parser.add_argument(
        "--model",
        dest="model",
        help=(
            "Training model name, defaulting to the configured training model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    train_walkforward_parser.add_argument(
        "--model-name",
        dest="model",
        help=argparse.SUPPRESS,
    )

    train_walkforward_show_parser = subparsers.add_parser(
        "train-walkforward-show",
        help="Show the most recent saved walk-forward validation report.",
    )
    train_walkforward_show_parser.add_argument(
        "--model",
        dest="model",
        help=(
            "Training model name, defaulting to the configured training model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    train_walkforward_show_parser.add_argument(
        "--model-name",
        dest="model",
        help=argparse.SUPPRESS,
    )

    backtest_walkforward_parser = subparsers.add_parser(
        "backtest-walkforward",
        help="Build a minimal overlapping portfolio from walk-forward test predictions.",
    )
    backtest_walkforward_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    backtest_walkforward_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of names selected per signal date.",
    )
    backtest_walkforward_parser.add_argument(
        "--holding-days",
        type=int,
        help="Holding window in trading days.",
    )
    backtest_walkforward_parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        help="Single-side transaction cost in basis points.",
    )

    backtest_show_parser = subparsers.add_parser(
        "backtest-show",
        help="Show a saved minimal backtest summary.",
    )
    backtest_show_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    backtest_show_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of names selected per signal date.",
    )
    backtest_show_parser.add_argument(
        "--holding-days",
        type=int,
        help="Holding window in trading days.",
    )
    backtest_show_parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        help="Single-side transaction cost in basis points.",
    )

    backtest_compare_parser = subparsers.add_parser(
        "backtest-compare",
        help="Compare a saved model backtest against the minimal benchmark set.",
    )
    backtest_compare_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    backtest_compare_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of names selected per signal date.",
    )
    backtest_compare_parser.add_argument(
        "--holding-days",
        type=int,
        help="Holding window in trading days.",
    )
    backtest_compare_parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        help="Single-side transaction cost in basis points.",
    )

    backtest_compare_show_parser = subparsers.add_parser(
        "backtest-compare-show",
        help="Show a saved benchmark comparison summary and table.",
    )
    backtest_compare_show_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    backtest_compare_show_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of names selected per signal date.",
    )
    backtest_compare_show_parser.add_argument(
        "--holding-days",
        type=int,
        help="Holding window in trading days.",
    )
    backtest_compare_show_parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        help="Single-side transaction cost in basis points.",
    )

    backtest_robustness_parser = subparsers.add_parser(
        "backtest-robustness",
        help="Run a small robustness grid over backtest and benchmark configurations.",
    )
    backtest_robustness_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    backtest_robustness_parser.add_argument(
        "--top-k-values",
        help="Comma-separated top-k values, for example 3,5,10.",
    )
    backtest_robustness_parser.add_argument(
        "--holding-days-values",
        help="Comma-separated holding-day values, for example 3,5,10.",
    )
    backtest_robustness_parser.add_argument(
        "--cost-bps-values",
        help="Comma-separated single-side transaction costs, for example 0,5,10.",
    )

    backtest_robustness_show_parser = subparsers.add_parser(
        "backtest-robustness-show",
        help="Show a saved robustness summary and table.",
    )
    backtest_robustness_show_parser.add_argument(
        "--model",
        help=(
            "Backtest model name, defaulting to the configured backtest model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )

    report_generate_parser = subparsers.add_parser(
        "report-generate",
        help="Generate a minimal markdown research report from saved artifacts.",
    )
    report_generate_parser.add_argument(
        "--model",
        help=(
            "Report model name, defaulting to the configured report model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    report_generate_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of names selected per signal date.",
    )
    report_generate_parser.add_argument(
        "--holding-days",
        type=int,
        help="Holding window in trading days.",
    )
    report_generate_parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        help="Single-side transaction cost in basis points.",
    )

    report_show_parser = subparsers.add_parser(
        "report-show",
        help="Show a saved markdown research report.",
    )
    report_show_parser.add_argument(
        "--model",
        help=(
            "Report model name, defaulting to the configured report model. "
            f"Supported: {', '.join(SUPPORTED_MODEL_NAMES)}."
        ),
    )
    report_show_parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full markdown report instead of only the leading sections.",
    )
    return parser


def run_doctor() -> int:
    settings = get_settings()
    env_exists = Path(".env").exists()

    print(f"project: {PROJECT_NAME}")
    print(f"python: {sys.version.split()[0]}")
    print(f"app_env: {settings.app_env}")
    print(f"log_level: {settings.log_level}")
    print(f"data_dir: {settings.data_dir}")
    print(f"artifacts_dir: {settings.artifacts_dir}")
    print(f"default_data_provider: {settings.default_data_provider}")
    print(f"default_start_date: {settings.default_start_date}")
    print(f"default_symbols: {', '.join(get_default_universe(settings))}")
    print(f"default_backtest_model: {settings.default_backtest_model}")
    print(f"default_top_k: {settings.default_top_k}")
    print(f"default_holding_days: {settings.default_holding_days}")
    print(f"default_cost_bps_per_side: {settings.default_cost_bps_per_side}")
    print(f"default_report_model: {settings.default_report_model}")
    print(f"default_report_top_k: {settings.default_report_top_k}")
    print(f"default_report_holding_days: {settings.default_report_holding_days}")
    print(f"default_report_cost_bps_per_side: {settings.default_report_cost_bps_per_side}")
    print(
        "default_robustness_top_k_values: "
        f"{', '.join(str(item) for item in settings.default_robustness_top_k_values)}"
    )
    print(
        "default_robustness_holding_days_values: "
        f"{', '.join(str(item) for item in settings.default_robustness_holding_days_values)}"
    )
    print(
        "default_robustness_cost_bps_values: "
        f"{', '.join(str(item) for item in settings.default_robustness_cost_bps_values)}"
    )
    print(f"default_benchmark_symbol: {settings.default_benchmark_symbol}")
    print(f"default_benchmark_momentum_feature: {settings.default_benchmark_momentum_feature}")
    print(f".env detected: {'yes' if env_exists else 'no'}")
    print(
        "status: minimal data, feature, label, dataset, hist_gbr/xgb_regressor/xgb_ranker "
        "training, walk-forward validation, minimal walk-forward portfolio backtest, and "
        "benchmark comparison plus robustness analysis layers available; minimal markdown "
        "research reporting is available; formal production backtest is not implemented yet"
    )
    return 0


def run_data_fetch(symbols: list[str], start_date: str | None, end_date: str | None) -> int:
    settings = get_settings()
    service = build_data_service(settings)
    resolved_symbols = resolve_symbols(symbols, settings)
    failed_symbols: list[str] = []

    print(
        f"fetching daily data for {len(resolved_symbols)} symbol(s) "
        f"via {settings.default_data_provider} ..."
    )
    for symbol in resolved_symbols:
        try:
            frame = service.fetch_daily_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            failed_symbols.append(symbol)
            print(f"{symbol}: fetch failed: {exc}")
            continue

        path = service.storage.path_for_symbol(symbol)
        print(f"{symbol}: {len(frame)} rows cached at {path}")

    return 1 if failed_symbols else 0


def run_data_show(symbol: str) -> int:
    service = build_data_service()
    try:
        frame = service.storage.read_daily_data(symbol)
    except FileNotFoundError:
        print(f"cached data not found for {symbol}. Run `suffering data-fetch {symbol}` first.")
        return 1

    print(f"symbol: {symbol.upper()}")
    print(f"rows: {len(frame)}")
    print("head:")
    print(frame.head().to_string(index=False))
    print("tail:")
    print(frame.tail().to_string(index=False))
    return 0


def run_feature_build(symbols: list[str]) -> int:
    settings = get_settings()
    service = build_feature_service(settings)
    resolved_symbols = resolve_symbols(symbols, settings)
    failed_symbols: list[str] = []

    print(f"building daily features for {len(resolved_symbols)} symbol(s) ...")
    for symbol in resolved_symbols:
        try:
            frame = service.build_features_for_symbol(symbol)
        except FileNotFoundError as exc:
            failed_symbols.append(symbol)
            print(f"{symbol}: {exc}")
            continue

        path = service.storage.path_for_symbol(symbol)
        print(
            f"{symbol}: {len(frame)} rows, {len(FEATURE_COLUMNS)} feature columns cached at {path}"
        )

    return 1 if failed_symbols else 0


def run_feature_show(symbol: str) -> int:
    service = build_feature_service()
    try:
        frame = service.read_features(symbol)
    except FileNotFoundError:
        print(
            f"cached features not found for {symbol.upper()}. "
            f"Run `suffering feature-build {symbol.upper()}` first."
        )
        return 1

    print(f"symbol: {symbol.upper()}")
    print(f"rows: {len(frame)}")
    print(f"feature_columns: {', '.join(FEATURE_COLUMNS)}")
    print("head:")
    print(frame.head().to_string(index=False))
    print("tail:")
    print(frame.tail().to_string(index=False))
    return 0


def run_label_build(symbols: list[str]) -> int:
    settings = get_settings()
    service = build_ranking_service(settings)
    resolved_symbols = resolve_symbols(symbols, settings)
    failed_symbols: list[str] = []

    print(
        f"building daily labels ({settings.default_label_horizon_days}d) "
        f"for {len(resolved_symbols)} symbol(s) ..."
    )
    for symbol in resolved_symbols:
        try:
            frame = service.build_label_for_symbol(symbol)
        except FileNotFoundError as exc:
            failed_symbols.append(symbol)
            print(f"{symbol}: {exc}")
            continue

        path = service.storage.label_path_for_symbol(symbol)
        valid_rows = int(frame[FUTURE_RETURN_5D_COLUMN].notna().sum())
        print(f"{symbol}: {len(frame)} rows, {valid_rows} labeled rows cached at {path}")

    return 1 if failed_symbols else 0


def run_dataset_build(symbols: list[str], dataset_name: str | None) -> int:
    settings = get_settings()
    service = build_ranking_service(settings)
    resolved_dataset_name = dataset_name or settings.default_dataset_name

    print(f"building panel dataset `{resolved_dataset_name}` ...")
    try:
        frame = service.build_panel_dataset(symbols=symbols, dataset_name=resolved_dataset_name)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    path = service.storage.dataset_path(resolved_dataset_name)
    symbol_count = frame["symbol"].nunique() if "symbol" in frame.columns else 0
    if frame.empty:
        date_range = "n/a"
    else:
        date_range = f"{frame['date'].min().date()} -> {frame['date'].max().date()}"

    print(f"rows: {len(frame)}")
    print(f"columns: {len(frame.columns)}")
    print(f"date_range: {date_range}")
    print(f"symbols: {symbol_count}")
    print(f"cached at: {path}")
    return 0


def run_dataset_show(dataset_name: str | None) -> int:
    settings = get_settings()
    service = build_ranking_service(settings)
    resolved_dataset_name = dataset_name or settings.default_dataset_name

    try:
        frame = service.read_panel_dataset(dataset_name=resolved_dataset_name)
    except FileNotFoundError:
        print(
            f"cached dataset not found: {resolved_dataset_name}. "
            f"Run `suffering dataset-build` first."
        )
        return 1

    print(f"dataset: {resolved_dataset_name}")
    print(f"rows: {len(frame)}")
    print(f"columns: {len(frame.columns)}")
    print(f"has_{FUTURE_RETURN_5D_COLUMN}: {FUTURE_RETURN_5D_COLUMN in frame.columns}")
    print(f"has_{RELEVANCE_5D_5Q_COLUMN}: {RELEVANCE_5D_5Q_COLUMN in frame.columns}")
    print(f"column_names: {', '.join(frame.columns)}")
    print("head:")
    print(frame.head().to_string(index=False))
    print("tail:")
    print(frame.tail().to_string(index=False))
    return 0


def run_train_baseline(dataset_name: str | None, model_name: str | None) -> int:
    settings = get_settings()
    service = build_training_service(settings)
    resolved_dataset_name = dataset_name or settings.default_dataset_name
    resolved_model_name = model_name or settings.default_training_model

    try:
        summary = service.train_baseline(
            dataset_name=resolved_dataset_name,
            model_name=resolved_model_name,
        )
    except FileNotFoundError:
        print(
            f"cached dataset not found: {resolved_dataset_name}. "
            f"Run `suffering dataset-build --dataset-name {resolved_dataset_name}` first."
        )
        return 1
    except ValueError as exc:
        print(f"baseline training failed: {exc}")
        return 1

    print(f"dataset: {summary['dataset_name']}")
    print(f"model: {summary['model_name']}")
    print(f"task_type: {summary['task_type']}")
    print(f"rows: {summary['total_rows']}")
    print(f"feature_count: {summary['feature_count']}")
    print(f"feature_columns: {', '.join(summary['feature_columns'])}")

    for split_name in ("train", "validation", "test"):
        split_summary = summary["split_summary"][split_name]
        print(f"{split_name}_rows: {split_summary['rows']}")
        print(
            f"{split_name}_date_range: {split_summary['date_start']} -> {split_summary['date_end']}"
        )

    print("validation_metrics:")
    _print_metrics(summary["validation_metrics"])
    print("test_metrics:")
    _print_metrics(summary["test_metrics"])
    print(f"model_path: {summary['artifacts']['model_path']}")
    print(f"metrics_path: {summary['artifacts']['metrics_path']}")
    print(f"validation_predictions_path: {summary['artifacts']['validation_predictions_path']}")
    print(f"test_predictions_path: {summary['artifacts']['test_predictions_path']}")
    return 0


def run_train_show(model_name: str | None) -> int:
    settings = get_settings()
    service = build_training_service(settings)
    resolved_model_name = model_name or settings.default_training_model

    try:
        report = service.read_training_report(model_name=resolved_model_name)
    except FileNotFoundError:
        print(
            f"training report not found for model: {resolved_model_name}. "
            f"Run `suffering train-baseline` first."
        )
        return 1
    except ValueError as exc:
        print(f"training report failed: {exc}")
        return 1

    print(f"model: {report['model_name']}")
    print(f"dataset: {report['dataset_name']}")
    print(f"task_type: {report.get('task_type', 'regression')}")
    print(f"feature_count: {report['feature_count']}")

    for split_name in ("train", "validation", "test"):
        split_summary = report["split_summary"][split_name]
        print(f"{split_name}_rows: {split_summary['rows']}")
        print(
            f"{split_name}_date_range: {split_summary['date_start']} -> {split_summary['date_end']}"
        )

    print("validation_metrics:")
    _print_metrics(report["validation_metrics"])
    print("test_metrics:")
    _print_metrics(report["test_metrics"])

    artifact_paths = report["artifacts"]
    print(f"model_exists: {Path(artifact_paths['model_path']).exists()}")
    print(f"metrics_exists: {Path(artifact_paths['metrics_path']).exists()}")
    print(
        "validation_predictions_exists: "
        f"{Path(artifact_paths['validation_predictions_path']).exists()}"
    )
    print(f"test_predictions_exists: {Path(artifact_paths['test_predictions_path']).exists()}")
    return 0


def run_train_walkforward(dataset_name: str | None, model_name: str | None) -> int:
    settings = get_settings()
    service = build_training_service(settings)
    resolved_dataset_name = dataset_name or settings.default_dataset_name
    resolved_model_name = model_name or settings.default_training_model

    try:
        summary = service.train_walkforward(
            dataset_name=resolved_dataset_name,
            model_name=resolved_model_name,
        )
    except FileNotFoundError:
        print(
            f"cached dataset not found: {resolved_dataset_name}. "
            f"Run `suffering dataset-build --dataset-name {resolved_dataset_name}` first."
        )
        return 1
    except ValueError as exc:
        print(f"walk-forward validation failed: {exc}")
        return 1

    print(f"dataset: {summary['dataset_name']}")
    print(f"model: {summary['model_name']}")
    print(f"task_type: {summary['task_type']}")
    print(f"rows: {summary['total_rows']}")
    print(f"date_count: {summary['date_count']}")
    print(f"feature_count: {summary['feature_count']}")
    print(f"feature_columns: {', '.join(summary['feature_columns'])}")
    print(f"fold_count: {summary['fold_count']}")

    for fold in summary["folds"]:
        print(
            f"fold_{fold['fold_id']}_date_range: "
            f"train {fold['train_date_start']} -> {fold['train_date_end']}, "
            f"validation {fold['validation_date_start']} -> {fold['validation_date_end']}, "
            f"test {fold['test_date_start']} -> {fold['test_date_end']}"
        )

    if summary["notes"]:
        for note in summary["notes"]:
            print(f"note: {note}")

    print("walkforward_test_metric_means:")
    _print_metric_summary_means(summary["test_metrics_summary"])
    print(f"summary_path: {summary['artifacts']['summary_path']}")
    print(f"folds_path: {summary['artifacts']['folds_path']}")
    print(f"predictions_path: {summary['artifacts']['predictions_path']}")
    return 0


def run_train_walkforward_show(model_name: str | None) -> int:
    settings = get_settings()
    service = build_training_service(settings)
    resolved_model_name = model_name or settings.default_training_model

    try:
        report = service.read_walkforward_report(model_name=resolved_model_name)
    except FileNotFoundError:
        print(
            f"walk-forward report not found for model: {resolved_model_name}. "
            f"Run `suffering train-walkforward` first."
        )
        return 1
    except ValueError as exc:
        print(f"walk-forward report failed: {exc}")
        return 1

    print(f"model: {report['model_name']}")
    print(f"dataset: {report['dataset_name']}")
    print(f"task_type: {report.get('task_type', 'regression')}")
    print(f"feature_count: {report['feature_count']}")
    print(f"fold_count: {report['fold_count']}")
    if report["notes"]:
        for note in report["notes"]:
            print(f"note: {note}")

    print("walkforward_test_metric_means:")
    _print_metric_summary_means(report["test_metrics_summary"])

    artifact_paths = report["artifacts"]
    print(f"summary_exists: {Path(artifact_paths['summary_path']).exists()}")
    print(f"folds_exists: {Path(artifact_paths['folds_path']).exists()}")
    print(f"predictions_exists: {Path(artifact_paths['predictions_path']).exists()}")
    return 0


def run_backtest_walkforward(
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float | None,
) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        summary = service.run_walkforward_backtest(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Walk-forward test predictions not found" in message:
            print(
                f"{message}. Run `suffering train-walkforward --model {resolved_model_name}` first."
            )
        elif "Raw daily cache not found" in message:
            print(f"{message} If needed, run `suffering data-fetch` first.")
        else:
            print(message)
        return 1
    except ValueError as exc:
        print(f"backtest failed: {exc}")
        return 1

    print(f"model: {summary['model_name']}")
    print(f"top_k: {summary['top_k']}")
    print(f"holding_days: {summary['holding_days']}")
    print(f"cost_bps_per_side: {summary['cost_bps_per_side']}")
    print(f"round_trip_cost_bps: {summary['round_trip_cost_bps']}")
    print(f"signal_date_range: {summary['signal_date_start']} -> {summary['signal_date_end']}")
    print(
        "portfolio_date_range: "
        f"{summary['portfolio_date_start']} -> {summary['portfolio_date_end']}"
    )
    print(f"trade_count: {summary['trade_count']}")
    print(f"skipped_trade_count: {summary['skipped_trade_count']}")
    print(f"gross_total_return: {_format_metric(summary['total_return_gross'])}")
    print(f"net_total_return: {_format_metric(summary['total_return_net'])}")
    print(f"gross_sharpe: {_format_metric(summary['sharpe_ratio_gross'])}")
    print(f"net_sharpe: {_format_metric(summary['sharpe_ratio_net'])}")
    print(f"gross_max_drawdown: {_format_metric(summary['max_drawdown_gross'])}")
    print(f"net_max_drawdown: {_format_metric(summary['max_drawdown_net'])}")
    print(f"summary_path: {summary['artifacts']['summary_path']}")
    print(f"daily_returns_path: {summary['artifacts']['daily_returns_path']}")
    print(f"equity_curve_path: {summary['artifacts']['equity_curve_path']}")
    print(f"trades_path: {summary['artifacts']['trades_path']}")
    return 0


def run_backtest_show(
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float | None,
) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        summary = service.read_backtest_summary(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
    except FileNotFoundError:
        resolved_top_k = top_k if top_k is not None else settings.default_top_k
        resolved_holding_days = (
            holding_days if holding_days is not None else settings.default_holding_days
        )
        resolved_cost = (
            float(cost_bps_per_side)
            if cost_bps_per_side is not None
            else float(settings.default_cost_bps_per_side)
        )
        print(
            "backtest summary not found for "
            f"model={resolved_model_name}, top_k={resolved_top_k}, "
            f"holding_days={resolved_holding_days}, cost_bps_per_side={resolved_cost}. "
            "Run `suffering backtest-walkforward` first."
        )
        return 1
    except ValueError as exc:
        print(f"backtest summary failed: {exc}")
        return 1

    print(f"model: {summary['model_name']}")
    print(f"top_k: {summary['top_k']}")
    print(f"holding_days: {summary['holding_days']}")
    print(f"cost_bps_per_side: {summary['cost_bps_per_side']}")
    print(f"trade_count: {summary['trade_count']}")
    print(f"gross_total_return: {_format_metric(summary['total_return_gross'])}")
    print(f"net_total_return: {_format_metric(summary['total_return_net'])}")
    print(f"gross_sharpe: {_format_metric(summary['sharpe_ratio_gross'])}")
    print(f"net_sharpe: {_format_metric(summary['sharpe_ratio_net'])}")
    print(f"gross_max_drawdown: {_format_metric(summary['max_drawdown_gross'])}")
    print(f"net_max_drawdown: {_format_metric(summary['max_drawdown_net'])}")

    artifact_paths = summary["artifacts"]
    print(f"summary_exists: {Path(artifact_paths['summary_path']).exists()}")
    print(f"daily_returns_exists: {Path(artifact_paths['daily_returns_path']).exists()}")
    print(f"equity_curve_exists: {Path(artifact_paths['equity_curve_path']).exists()}")
    print(f"trades_exists: {Path(artifact_paths['trades_path']).exists()}")
    return 0


def run_backtest_compare(
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float | None,
) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        summary = service.run_backtest_comparison(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Backtest summary not found" in message or "Backtest artifact not found" in message:
            print(
                "model backtest artifacts not found for "
                f"{resolved_model_name}. "
                f"Run `suffering backtest-walkforward --model {resolved_model_name}` first."
            )
        elif "Walk-forward test predictions not found" in message:
            print(
                f"{message}. Run `suffering train-walkforward --model {resolved_model_name}` first."
            )
        elif "Feature cache not found" in message:
            print(f"{message} If needed, run `suffering feature-build` first.")
        elif "Raw daily cache not found for symbol(s): QQQ" in message:
            print(
                f"{message} Benchmark comparison does not auto-fetch QQQ. "
                "Run `suffering data-fetch QQQ "
                "--start-date 2020-01-01 --end-date 2024-12-31` first."
            )
        elif "Raw daily cache not found" in message:
            print(f"{message} If needed, run `suffering data-fetch` first.")
        else:
            print(message)
        return 1
    except ValueError as exc:
        print(f"backtest comparison failed: {exc}")
        return 1

    model_strategy = summary["model_strategy"]
    best_benchmark_by_sharpe = summary["best_benchmark_by_sharpe_net"]
    best_benchmark_by_total_return = summary["best_benchmark_by_total_return_net"]
    print(f"model: {summary['model_name']}")
    print(f"top_k: {summary['top_k']}")
    print(f"holding_days: {summary['holding_days']}")
    print(f"cost_bps_per_side: {summary['cost_bps_per_side']}")
    print(f"benchmark_count: {summary['benchmark_count']}")
    print(
        "comparison_date_range: "
        f"{summary['comparison_date_start']} -> {summary['comparison_date_end']}"
    )
    print(f"model_net_total_return: {_format_metric(model_strategy['total_return_net'])}")
    print(f"model_net_sharpe: {_format_metric(model_strategy['sharpe_ratio_net'])}")
    print(f"model_net_max_drawdown: {_format_metric(model_strategy['max_drawdown_net'])}")
    print(
        "best_benchmark_by_sharpe_net: "
        f"{best_benchmark_by_sharpe['strategy_name']} "
        f"({_format_metric(best_benchmark_by_sharpe['sharpe_ratio_net'])})"
    )
    print(
        "best_benchmark_by_total_return_net: "
        f"{best_benchmark_by_total_return['strategy_name']} "
        f"({_format_metric(best_benchmark_by_total_return['total_return_net'])})"
    )
    print(f"comparison_summary_path: {summary['artifacts']['summary_path']}")
    print(f"comparison_table_path: {summary['artifacts']['table_path']}")
    return 0


def run_backtest_compare_show(
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float | None,
) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        report = service.read_backtest_comparison(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
    except FileNotFoundError:
        resolved_top_k = top_k if top_k is not None else settings.default_top_k
        resolved_holding_days = (
            holding_days if holding_days is not None else settings.default_holding_days
        )
        resolved_cost = (
            float(cost_bps_per_side)
            if cost_bps_per_side is not None
            else float(settings.default_cost_bps_per_side)
        )
        print(
            "backtest comparison not found for "
            f"model={resolved_model_name}, top_k={resolved_top_k}, "
            f"holding_days={resolved_holding_days}, cost_bps_per_side={resolved_cost}. "
            "Run `suffering backtest-compare` first."
        )
        return 1
    except ValueError as exc:
        print(f"backtest comparison show failed: {exc}")
        return 1

    model_strategy = report["model_strategy"]
    print(f"model: {report['model_name']}")
    print(f"benchmark_count: {report['benchmark_count']}")
    print(
        "comparison_date_range: "
        f"{report['comparison_date_start']} -> {report['comparison_date_end']}"
    )
    print(f"model_net_total_return: {_format_metric(model_strategy['total_return_net'])}")
    print(f"model_net_sharpe: {_format_metric(model_strategy['sharpe_ratio_net'])}")
    print(
        "best_benchmark_by_sharpe_net: "
        f"{report['best_benchmark_by_sharpe_net']['strategy_name']} "
        f"({_format_metric(report['best_benchmark_by_sharpe_net']['sharpe_ratio_net'])})"
    )
    print(
        "best_benchmark_by_total_return_net: "
        f"{report['best_benchmark_by_total_return_net']['strategy_name']} "
        f"({_format_metric(report['best_benchmark_by_total_return_net']['total_return_net'])})"
    )

    table_frame = pd.DataFrame(report["table_rows"])
    if not table_frame.empty:
        print("comparison_table:")
        print(table_frame.to_string(index=False))

    artifact_paths = report["artifacts"]
    print(f"comparison_summary_exists: {Path(artifact_paths['summary_path']).exists()}")
    print(f"comparison_table_exists: {Path(artifact_paths['table_path']).exists()}")
    return 0


def run_backtest_robustness(
    model_name: str | None,
    top_k_values: str | None,
    holding_days_values: str | None,
    cost_bps_values: str | None,
) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        summary = service.run_backtest_robustness(
            model_name=resolved_model_name,
            top_k_values=_parse_int_values_argument(
                top_k_values,
                settings.default_robustness_top_k_values,
                argument_name="top_k_values",
            ),
            holding_days_values=_parse_int_values_argument(
                holding_days_values,
                settings.default_robustness_holding_days_values,
                argument_name="holding_days_values",
            ),
            cost_bps_values=_parse_float_values_argument(
                cost_bps_values,
                settings.default_robustness_cost_bps_values,
                argument_name="cost_bps_values",
            ),
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "Walk-forward test predictions not found" in message:
            print(
                f"{message}. Run `suffering train-walkforward --model {resolved_model_name}` first."
            )
        elif "Feature cache not found" in message:
            print(f"{message} If needed, run `suffering feature-build` first.")
        elif "Raw daily cache not found for symbol(s): QQQ" in message:
            print(
                f"{message} Robustness analysis does not auto-fetch QQQ. "
                "Run `suffering data-fetch QQQ "
                "--start-date 2020-01-01 --end-date 2024-12-31` first."
            )
        elif "Raw daily cache not found" in message:
            print(f"{message} If needed, run `suffering data-fetch` first.")
        else:
            print(message)
        return 1
    except ValueError as exc:
        print(f"robustness analysis failed: {exc}")
        return 1

    print(f"model: {summary['model_name']}")
    print(f"total_configs_evaluated: {summary['total_configs_evaluated']}")
    print(f"table_row_count: {summary['row_count']}")
    print(f"top_k_values: {', '.join(str(item) for item in summary['top_k_values'])}")
    print(f"holding_days_values: {', '.join(str(item) for item in summary['holding_days_values'])}")
    print(f"cost_bps_values: {', '.join(str(item) for item in summary['cost_bps_values'])}")
    print(
        "best_config_by_sharpe_net: "
        f"{_format_robustness_config(summary['best_config_by_sharpe_net'], 'sharpe_ratio_net')}"
    )
    print(
        "best_config_by_total_return_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                summary['best_config_by_total_return_net'],
                'total_return_net',
            )
        }"
    )
    print(
        "simple_momentum_best_sharpe_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                summary['simple_momentum_best_sharpe_net'],
                'sharpe_ratio_net',
            )
        }"
    )
    print(
        "simple_momentum_best_total_return_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                summary['simple_momentum_best_total_return_net'],
                'total_return_net',
            )
        }"
    )
    print(
        "whether_model_beats_simple_momentum_on_best_sharpe: "
        f"{summary['whether_model_beats_simple_momentum_on_best_sharpe']}"
    )
    print(
        "whether_model_beats_simple_momentum_on_best_total_return: "
        f"{summary['whether_model_beats_simple_momentum_on_best_total_return']}"
    )
    for note in summary["robustness_notes"]:
        print(f"robustness_note: {note}")
    print(f"robustness_summary_path: {summary['artifacts']['summary_path']}")
    print(f"robustness_table_path: {summary['artifacts']['table_path']}")
    return 0


def run_backtest_robustness_show(model_name: str | None) -> int:
    settings = get_settings()
    service = build_backtest_service(settings)
    resolved_model_name = model_name or settings.default_backtest_model

    try:
        report = service.read_backtest_robustness(model_name=resolved_model_name)
    except FileNotFoundError:
        print(
            f"backtest robustness report not found for model={resolved_model_name}. "
            "Run `suffering backtest-robustness` first."
        )
        return 1
    except ValueError as exc:
        print(f"backtest robustness show failed: {exc}")
        return 1

    print(f"model: {report['model_name']}")
    print(f"total_configs_evaluated: {report['total_configs_evaluated']}")
    print(f"table_row_count: {report['row_count']}")
    print(
        "best_config_by_sharpe_net: "
        f"{_format_robustness_config(report['best_config_by_sharpe_net'], 'sharpe_ratio_net')}"
    )
    print(
        "best_config_by_total_return_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                report['best_config_by_total_return_net'],
                'total_return_net',
            )
        }"
    )
    print(
        "simple_momentum_best_sharpe_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                report['simple_momentum_best_sharpe_net'],
                'sharpe_ratio_net',
            )
        }"
    )
    print(
        "simple_momentum_best_total_return_net: "
        f"{
            _format_robustness_config(  # noqa: E501
                report['simple_momentum_best_total_return_net'],
                'total_return_net',
            )
        }"
    )
    for note in report["robustness_notes"]:
        print(f"robustness_note: {note}")

    table_frame = pd.DataFrame(report["table_rows"])
    if not table_frame.empty:
        print("robustness_table_top:")
        print(table_frame.head(10).to_string(index=False))

    artifact_paths = report["artifacts"]
    print(f"robustness_summary_exists: {Path(artifact_paths['summary_path']).exists()}")
    print(f"robustness_table_exists: {Path(artifact_paths['table_path']).exists()}")
    return 0


def run_report_generate(
    model_name: str | None,
    top_k: int | None,
    holding_days: int | None,
    cost_bps_per_side: float | None,
) -> int:
    settings = get_settings()
    service = build_report_service(settings)
    resolved_model_name = model_name or settings.default_report_model

    try:
        summary = service.generate_research_report(
            model_name=resolved_model_name,
            top_k=top_k,
            holding_days=holding_days,
            cost_bps_per_side=cost_bps_per_side,
        )
    except FileNotFoundError as exc:
        print(
            f"{exc} Generate the relevant upstream artifacts first, for example "
            "`train-walkforward`, `backtest-walkforward`, `backtest-compare`, "
            "or `backtest-robustness`."
        )
        return 1
    except ValueError as exc:
        print(f"research report generation failed: {exc}")
        return 1

    print(f"model: {summary['model_name']}")
    print(f"task_type: {summary['task_type']}")
    print(
        "available_artifacts: "
        f"{', '.join(item['name'] for item in summary['available_artifacts']) or 'none'}"
    )
    print(
        "missing_artifacts: "
        f"{', '.join(item['name'] for item in summary['missing_artifacts']) or 'none'}"
    )
    print(f"available_sections: {', '.join(summary['available_sections']) or 'none'}")
    print(f"missing_sections: {', '.join(summary['missing_sections']) or 'none'}")
    print(f"report_path: {summary['report_path']}")
    return 0


def run_report_show(model_name: str | None, show_full: bool) -> int:
    settings = get_settings()
    service = build_report_service(settings)
    resolved_model_name = model_name or settings.default_report_model

    try:
        report = service.read_research_report(model_name=resolved_model_name)
    except FileNotFoundError:
        print(
            f"research report not found for model={resolved_model_name}. "
            "Run `suffering report-generate` first."
        )
        return 1
    except ValueError as exc:
        print(f"research report show failed: {exc}")
        return 1

    print(f"model: {report['model_name']}")
    print(f"report_path: {report['report_path']}")
    print("content:")
    print(_truncate_report_content(report["content"], show_full=show_full))
    if not show_full:
        print("... use `suffering report-show --full` to print the complete markdown report.")
    return 0


def _print_metrics(metrics: dict[str, float | None]) -> None:
    for name in METRIC_NAMES:
        print(f"  {name}: {_format_metric(metrics.get(name))}")


def _print_metric_summary_means(
    metrics_summary: dict[str, dict[str, float | None]],
) -> None:
    for name in METRIC_NAMES:
        metric_summary = metrics_summary.get(name, {})
        print(f"  {name}: {_format_metric(metric_summary.get('mean'))}")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_robustness_config(
    row: dict[str, object] | None,
    metric_name: str,
) -> str:
    if not row:
        return "n/a"

    return (
        f"strategy={row.get('strategy_name')}, "
        f"model_name={row.get('model_name') or 'n/a'}, "
        f"top_k={row.get('top_k')}, "
        f"holding_days={row.get('holding_days')}, "
        f"cost_bps_per_side={row.get('cost_bps_per_side')}, "
        f"{metric_name}={_format_metric(_coerce_optional_float(row.get(metric_name)))}"
    )


def _parse_int_values_argument(
    raw_value: str | None,
    default_values: list[int],
    *,
    argument_name: str,
) -> list[int]:
    if raw_value is None:
        return list(default_values)
    parsed_values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not parsed_values:
        raise ValueError(f"{argument_name} must not be empty")
    return [int(item) for item in parsed_values]


def _parse_float_values_argument(
    raw_value: str | None,
    default_values: list[float],
    *,
    argument_name: str,
) -> list[float]:
    if raw_value is None:
        return list(default_values)
    parsed_values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not parsed_values:
        raise ValueError(f"{argument_name} must not be empty")
    return [float(item) for item in parsed_values]


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _truncate_report_content(content: str, *, show_full: bool, max_lines: int = 80) -> str:
    if show_full:
        return content
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    return "\n".join(lines[:max_lines])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor()
    if args.command == "data-fetch":
        return run_data_fetch(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    if args.command == "data-show":
        return run_data_show(symbol=args.symbol)
    if args.command == "feature-build":
        return run_feature_build(symbols=args.symbols)
    if args.command == "feature-show":
        return run_feature_show(symbol=args.symbol)
    if args.command == "label-build":
        return run_label_build(symbols=args.symbols)
    if args.command == "dataset-build":
        return run_dataset_build(symbols=args.symbols, dataset_name=args.dataset_name)
    if args.command == "dataset-show":
        return run_dataset_show(dataset_name=args.dataset_name)
    if args.command == "train-baseline":
        return run_train_baseline(dataset_name=args.dataset_name, model_name=args.model)
    if args.command == "train-show":
        return run_train_show(model_name=args.model)
    if args.command == "train-walkforward":
        return run_train_walkforward(dataset_name=args.dataset_name, model_name=args.model)
    if args.command == "train-walkforward-show":
        return run_train_walkforward_show(model_name=args.model)
    if args.command == "backtest-walkforward":
        return run_backtest_walkforward(
            model_name=args.model,
            top_k=args.top_k,
            holding_days=args.holding_days,
            cost_bps_per_side=args.cost_bps_per_side,
        )
    if args.command == "backtest-show":
        return run_backtest_show(
            model_name=args.model,
            top_k=args.top_k,
            holding_days=args.holding_days,
            cost_bps_per_side=args.cost_bps_per_side,
        )
    if args.command == "backtest-compare":
        return run_backtest_compare(
            model_name=args.model,
            top_k=args.top_k,
            holding_days=args.holding_days,
            cost_bps_per_side=args.cost_bps_per_side,
        )
    if args.command == "backtest-compare-show":
        return run_backtest_compare_show(
            model_name=args.model,
            top_k=args.top_k,
            holding_days=args.holding_days,
            cost_bps_per_side=args.cost_bps_per_side,
        )
    if args.command == "backtest-robustness":
        return run_backtest_robustness(
            model_name=args.model,
            top_k_values=args.top_k_values,
            holding_days_values=args.holding_days_values,
            cost_bps_values=args.cost_bps_values,
        )
    if args.command == "backtest-robustness-show":
        return run_backtest_robustness_show(model_name=args.model)
    if args.command == "report-generate":
        return run_report_generate(
            model_name=args.model,
            top_k=args.top_k,
            holding_days=args.holding_days,
            cost_bps_per_side=args.cost_bps_per_side,
        )
    if args.command == "report-show":
        return run_report_show(model_name=args.model, show_full=args.full)

    print("Welcome to suffering.")
    print("This is the initial quant research project skeleton.")
    print("Run `suffering doctor` to inspect the current environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
