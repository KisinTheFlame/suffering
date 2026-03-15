"""Run the full research pipeline on the current Nasdaq-100 static universe.

This script intentionally uses a static snapshot of the current Nasdaq-100
constituents, then runs the existing local pipeline over 2018-01-01 to
2025-12-31 without introducing point-in-time constituent changes.
"""

from __future__ import annotations

import time
from pathlib import Path

from suffering.backtest import build_backtest_service
from suffering.config.settings import Settings
from suffering.data import build_data_service
from suffering.features import build_feature_service
from suffering.ranking import build_ranking_service
from suffering.reports import build_report_service
from suffering.training import build_training_service

CURRENT_NASDAQ_100_SYMBOLS = [
    "ADBE",
    "AMD",
    "ABNB",
    "ALNY",
    "GOOGL",
    "GOOG",
    "AMZN",
    "AEP",
    "AMGN",
    "ADI",
    "AAPL",
    "AMAT",
    "APP",
    "ARM",
    "ASML",
    "TEAM",
    "ADSK",
    "ADP",
    "AXON",
    "BKR",
    "BKNG",
    "AVGO",
    "CDNS",
    "CHTR",
    "CTAS",
    "CSCO",
    "CCEP",
    "CTSH",
    "CMCSA",
    "CEG",
    "CPRT",
    "CSGP",
    "COST",
    "CRWD",
    "CSX",
    "DDOG",
    "DXCM",
    "FANG",
    "DASH",
    "EA",
    "EXC",
    "FAST",
    "FER",
    "FTNT",
    "GEHC",
    "GILD",
    "HON",
    "IDXX",
    "INSM",
    "INTC",
    "INTU",
    "ISRG",
    "KDP",
    "KLAC",
    "KHC",
    "LRCX",
    "LIN",
    "MAR",
    "MRVL",
    "MELI",
    "META",
    "MCHP",
    "MU",
    "MSFT",
    "MSTR",
    "MDLZ",
    "MPWR",
    "MNST",
    "NFLX",
    "NVDA",
    "NXPI",
    "ORLY",
    "ODFL",
    "PCAR",
    "PLTR",
    "PANW",
    "PAYX",
    "PYPL",
    "PDD",
    "PEP",
    "QCOM",
    "REGN",
    "ROP",
    "ROST",
    "STX",
    "SHOP",
    "SBUX",
    "SNPS",
    "TMUS",
    "TTWO",
    "TSLA",
    "TXN",
    "TRI",
    "VRSK",
    "VRTX",
    "WMT",
    "WBD",
    "WDC",
    "WDAY",
    "XEL",
    "ZS",
]
BENCHMARK_SYMBOLS = ["QQQ"]
FETCH_SYMBOLS = CURRENT_NASDAQ_100_SYMBOLS + BENCHMARK_SYMBOLS

RUN_NAME = "nasdaq100_current_static_2018_2025"
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
DATASET_NAME = "nasdaq100_current_static_panel_5d"
MODEL_NAME = "xgb_ranker"
TOP_K = 5
HOLDING_DAYS = 5
COST_BPS_PER_SIDE = 5.0
FETCH_RETRIES = 3


def build_run_settings() -> Settings:
    data_dir = Path("data") / RUN_NAME
    artifacts_dir = Path("artifacts") / RUN_NAME
    return Settings(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        default_start_date=START_DATE,
        default_end_date=END_DATE,
        default_symbols=list(CURRENT_NASDAQ_100_SYMBOLS),
        default_dataset_name=DATASET_NAME,
        default_training_model=MODEL_NAME,
        default_backtest_model=MODEL_NAME,
        default_report_model=MODEL_NAME,
        default_top_k=TOP_K,
        default_holding_days=HOLDING_DAYS,
        default_cost_bps_per_side=COST_BPS_PER_SIDE,
        default_report_top_k=TOP_K,
        default_report_holding_days=HOLDING_DAYS,
        default_report_cost_bps_per_side=COST_BPS_PER_SIDE,
    )


def main() -> int:
    settings = build_run_settings()
    data_service = build_data_service(settings)
    feature_service = build_feature_service(settings)
    ranking_service = build_ranking_service(settings)
    training_service = build_training_service(settings)
    backtest_service = build_backtest_service(settings)
    report_service = build_report_service(settings)

    print(f"run_name: {RUN_NAME}")
    print(
        f"universe_size: {len(CURRENT_NASDAQ_100_SYMBOLS)} "
        f"(static current Nasdaq-100 constituents)"
    )
    print(f"date_range: {START_DATE} -> {END_DATE}")
    print(f"data_dir: {settings.data_dir}")
    print(f"artifacts_dir: {settings.artifacts_dir}")

    failed_symbols = fetch_raw_data(data_service)
    if failed_symbols:
        failed_display = ", ".join(failed_symbols)
        raise RuntimeError(f"data fetch failed for symbol(s): {failed_display}")

    build_features(feature_service)
    build_labels(ranking_service)

    dataset_frame = ranking_service.build_panel_dataset(
        symbols=CURRENT_NASDAQ_100_SYMBOLS,
        dataset_name=DATASET_NAME,
    )
    print(
        f"dataset_rows: {len(dataset_frame)} | "
        f"dataset_path: {ranking_service.storage.dataset_path(DATASET_NAME)}"
    )

    walkforward_summary = training_service.train_walkforward(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
    )
    print(
        f"walkforward_fold_count: {walkforward_summary['fold_count']} | "
        f"summary_path: {walkforward_summary['artifacts']['summary_path']}"
    )

    backtest_summary = backtest_service.run_walkforward_backtest(
        model_name=MODEL_NAME,
        top_k=TOP_K,
        holding_days=HOLDING_DAYS,
        cost_bps_per_side=COST_BPS_PER_SIDE,
    )
    print(
        f"backtest_net_sharpe: {backtest_summary['sharpe_ratio_net']:.4f} | "
        f"summary_path: {backtest_summary['artifacts']['summary_path']}"
    )

    comparison_summary = backtest_service.run_backtest_comparison(
        model_name=MODEL_NAME,
        top_k=TOP_K,
        holding_days=HOLDING_DAYS,
        cost_bps_per_side=COST_BPS_PER_SIDE,
    )
    print(
        f"comparison_benchmark_count: {comparison_summary['benchmark_count']} | "
        f"summary_path: {comparison_summary['artifacts']['summary_path']}"
    )

    robustness_summary = backtest_service.run_backtest_robustness(model_name=MODEL_NAME)
    print(
        f"robustness_config_count: {robustness_summary['total_configs_evaluated']} | "
        f"summary_path: {robustness_summary['artifacts']['summary_path']}"
    )

    report_summary = report_service.generate_research_report(
        model_name=MODEL_NAME,
        top_k=TOP_K,
        holding_days=HOLDING_DAYS,
        cost_bps_per_side=COST_BPS_PER_SIDE,
    )
    print(f"report_path: {report_summary['report_path']}")
    return 0


def fetch_raw_data(data_service) -> list[str]:
    failed_symbols: list[str] = []
    total = len(FETCH_SYMBOLS)
    print(f"fetching_raw_data: {total} symbol(s)")
    for index, symbol in enumerate(FETCH_SYMBOLS, start=1):
        if data_service.storage.exists(symbol):
            cached_frame = data_service.storage.read_daily_data(symbol)
            print(f"[fetch {index}/{total}] {symbol}: using cached {len(cached_frame)} rows")
            continue

        success = False
        last_error: Exception | None = None
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                frame = data_service.fetch_daily_data(
                    symbol=symbol,
                    start_date=START_DATE,
                    end_date=END_DATE,
                )
            except Exception as exc:
                last_error = exc
                print(
                    f"[fetch {index}/{total}] {symbol} attempt {attempt}/{FETCH_RETRIES} failed: "
                    f"{exc}"
                )
                time.sleep(float(attempt))
                continue

            print(f"[fetch {index}/{total}] {symbol}: {len(frame)} rows")
            success = True
            break

        if not success:
            failed_symbols.append(symbol)
            if last_error is not None:
                print(f"[fetch {index}/{total}] {symbol}: giving up after retries ({last_error})")
    return failed_symbols


def build_features(feature_service) -> None:
    total = len(CURRENT_NASDAQ_100_SYMBOLS)
    print(f"building_features: {total} symbol(s)")
    for index, symbol in enumerate(CURRENT_NASDAQ_100_SYMBOLS, start=1):
        frame = feature_service.build_features_for_symbol(symbol)
        print(f"[feature {index}/{total}] {symbol}: {len(frame)} rows")


def build_labels(ranking_service) -> None:
    total = len(CURRENT_NASDAQ_100_SYMBOLS)
    print(f"building_labels: {total} symbol(s)")
    for index, symbol in enumerate(CURRENT_NASDAQ_100_SYMBOLS, start=1):
        frame = ranking_service.build_label_for_symbol(symbol)
        print(f"[label {index}/{total}] {symbol}: {len(frame)} rows")


if __name__ == "__main__":
    raise SystemExit(main())
