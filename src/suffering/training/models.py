"""Minimal model factory helpers for regression training."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from suffering.config.settings import Settings, get_settings

SUPPORTED_MODEL_NAMES = ("hist_gbr", "xgb_regressor")


def validate_model_name(model_name: str) -> str:
    if model_name not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(
            f"Unsupported training model: {model_name}. Supported models: {supported}"
        )
    return model_name


def resolve_model_name(
    model_name: str | None = None,
    settings: Settings | None = None,
) -> str:
    resolved_settings = settings or get_settings()
    return validate_model_name(model_name or resolved_settings.default_training_model)


def build_regressor(
    model_name: str,
    settings: Settings | None = None,
    random_state: int | None = None,
) -> Any:
    resolved_settings = settings or get_settings()
    resolved_model_name = validate_model_name(model_name)
    resolved_random_state = (
        resolved_settings.random_seed if random_state is None else random_state
    )

    if resolved_model_name == "hist_gbr":
        return HistGradientBoostingRegressor(random_state=resolved_random_state)

    return XGBRegressor(
        n_estimators=resolved_settings.xgb_n_estimators,
        max_depth=resolved_settings.xgb_max_depth,
        learning_rate=resolved_settings.xgb_learning_rate,
        subsample=resolved_settings.xgb_subsample,
        colsample_bytree=resolved_settings.xgb_colsample_bytree,
        random_state=resolved_random_state,
        tree_method="hist",
        objective="reg:squarederror",
    )
