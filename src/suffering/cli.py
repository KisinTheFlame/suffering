"""Minimal command line interface for the suffering project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from suffering import PROJECT_NAME, __version__
from suffering.config.settings import get_settings


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
    print(f".env detected: {'yes' if env_exists else 'no'}")
    print("status: project skeleton initialized; data/training/backtest are not implemented yet")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor()

    print("Welcome to suffering.")
    print("This is the initial quant research project skeleton.")
    print("Run `suffering doctor` to inspect the current environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

