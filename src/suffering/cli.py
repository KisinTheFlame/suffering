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
        "status: minimal data and feature layers available; "
        "label/training/backtest are not implemented yet"
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

    print("Welcome to suffering.")
    print("This is the initial quant research project skeleton.")
    print("Run `suffering doctor` to inspect the current environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
