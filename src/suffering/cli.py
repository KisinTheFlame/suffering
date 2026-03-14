"""Minimal command line interface for the suffering project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from suffering import PROJECT_NAME, __version__
from suffering.config.settings import get_settings
from suffering.data import build_data_service, get_default_universe
from suffering.data.universe import resolve_symbols
from suffering.features import build_feature_service
from suffering.features.definitions import FEATURE_COLUMNS
from suffering.ranking import build_ranking_service
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN
from suffering.ranking.panel import RELEVANCE_5D_5Q_COLUMN


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
    print(f".env detected: {'yes' if env_exists else 'no'}")
    print(
        "status: minimal data, feature, label, and dataset layers available; "
        "training/backtest are not implemented yet"
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

    print("Welcome to suffering.")
    print("This is the initial quant research project skeleton.")
    print("Run `suffering doctor` to inspect the current environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
