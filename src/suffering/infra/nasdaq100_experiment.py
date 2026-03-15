"""Shared configuration for the static current Nasdaq-100 main experiment."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from suffering.config.settings import Settings

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


def build_nasdaq100_current_static_settings() -> Settings:
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


def remote_artifacts_relative_root() -> PurePosixPath:
    return PurePosixPath("artifacts") / RUN_NAME


def local_artifacts_output_dir(
    base_dir: Path | str = Path("artifacts") / "remote_experiments",
) -> Path:
    return Path(base_dir) / RUN_NAME


def report_relative_path() -> PurePosixPath:
    return PurePosixPath("reports") / "research" / f"{MODEL_NAME}_research_report.md"
