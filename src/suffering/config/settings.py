"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    default_data_provider: str = "yfinance"
    default_start_date: str = "2020-01-01"
    default_end_date: str | None = None
    default_symbols: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    default_label_horizon_days: int = 5
    default_dataset_name: str = "panel_5d"
    default_train_ratio: float = 0.6
    default_validation_ratio: float = 0.2
    default_test_ratio: float = 0.2
    walkforward_step_ratio: float = 0.2
    walkforward_min_folds: int = 1
    random_seed: int = 7
    default_training_model: str = "hist_gbr"
    default_backtest_model: str = "xgb_ranker"
    default_top_k: int = 5
    default_holding_days: int = 5
    default_cost_bps_per_side: float = 5.0
    default_report_model: str = "xgb_ranker"
    default_report_top_k: int = 5
    default_report_holding_days: int = 5
    default_report_cost_bps_per_side: float = 5.0
    default_robustness_top_k_values: list[int] = [3, 5, 10]
    default_robustness_holding_days_values: list[int] = [3, 5, 10]
    default_robustness_cost_bps_values: list[float] = [0.0, 5.0, 10.0]
    default_benchmark_symbol: str = "QQQ"
    default_benchmark_momentum_feature: str = "return_20d"
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_device: str = "cpu"
    xgb_ranker_n_estimators: int = 100
    xgb_ranker_max_depth: int = 4
    xgb_ranker_learning_rate: float = 0.05
    xgb_ranker_subsample: float = 0.8
    xgb_ranker_colsample_bytree: float = 0.8
    xgb_ranker_device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    @field_validator("default_symbols", mode="before")
    @classmethod
    def parse_default_symbols(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "default_robustness_top_k_values",
        "default_robustness_holding_days_values",
        mode="before",
    )
    @classmethod
    def parse_int_list_values(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator("default_robustness_cost_bps_values", mode="before")
    @classmethod
    def parse_float_list_values(cls, value: object) -> object:
        if isinstance(value, str):
            return [float(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "default_train_ratio",
        "default_validation_ratio",
        "default_test_ratio",
        "walkforward_step_ratio",
        "xgb_learning_rate",
        "xgb_subsample",
        "xgb_colsample_bytree",
        "xgb_ranker_learning_rate",
        "xgb_ranker_subsample",
        "xgb_ranker_colsample_bytree",
    )
    @classmethod
    def validate_ratio_range(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            raise ValueError("split ratios must be between 0 and 1")
        return value

    @field_validator("walkforward_min_folds")
    @classmethod
    def validate_walkforward_min_folds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("walkforward_min_folds must be at least 1")
        return value

    @field_validator(
        "default_top_k",
        "default_holding_days",
        "default_report_top_k",
        "default_report_holding_days",
    )
    @classmethod
    def validate_positive_defaults(cls, value: int) -> int:
        if value < 1:
            raise ValueError("backtest integer defaults must be at least 1")
        return value

    @field_validator("default_cost_bps_per_side", "default_report_cost_bps_per_side")
    @classmethod
    def validate_non_negative_cost_defaults(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cost defaults must be non-negative")
        return value

    @field_validator(
        "default_robustness_top_k_values",
        "default_robustness_holding_days_values",
    )
    @classmethod
    def validate_positive_int_list(cls, value: list[int]) -> list[int]:
        if not value or any(item < 1 for item in value):
            raise ValueError("robustness integer defaults must be positive and non-empty")
        return value

    @field_validator("default_robustness_cost_bps_values")
    @classmethod
    def validate_non_negative_float_list(cls, value: list[float]) -> list[float]:
        if not value or any(item < 0 for item in value):
            raise ValueError("robustness cost defaults must be non-negative and non-empty")
        return value

    @field_validator("xgb_device", "xgb_ranker_device")
    @classmethod
    def validate_xgb_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("xgboost device setting must not be empty")
        if normalized != "cpu" and not normalized.startswith("cuda"):
            raise ValueError("xgboost device must be `cpu` or start with `cuda`")
        return normalized

    @field_validator(
        "xgb_n_estimators",
        "xgb_max_depth",
        "xgb_ranker_n_estimators",
        "xgb_ranker_max_depth",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("xgboost integer settings must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_split_ratio_sum(self) -> "Settings":
        ratio_sum = (
            self.default_train_ratio + self.default_validation_ratio + self.default_test_ratio
        )
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(
                "default_train_ratio + default_validation_ratio + default_test_ratio must equal 1.0"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
