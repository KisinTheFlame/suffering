"""Minimal model factory helpers for regression training."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRanker, XGBRegressor

from suffering.config.settings import Settings, get_settings

REGRESSION_MODEL_NAMES = ("hist_gbr", "xgb_regressor")
RANKING_MODEL_NAMES = ("xgb_ranker",)
SUPPORTED_MODEL_NAMES = REGRESSION_MODEL_NAMES + RANKING_MODEL_NAMES


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


def resolve_model_task(
    model_name: str | None = None,
    settings: Settings | None = None,
) -> str:
    resolved_model_name = resolve_model_name(model_name=model_name, settings=settings)
    if resolved_model_name in RANKING_MODEL_NAMES:
        return "ranking"
    return "regression"


def build_regressor(
    model_name: str,
    settings: Settings | None = None,
    random_state: int | None = None,
) -> Any:
    resolved_settings = settings or get_settings()
    resolved_model_name = validate_model_name(model_name)
    if resolved_model_name not in REGRESSION_MODEL_NAMES:
        raise ValueError(f"Model {resolved_model_name} does not support regression training")
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


def build_ranker(
    model_name: str,
    settings: Settings | None = None,
    random_state: int | None = None,
) -> Any:
    resolved_settings = settings or get_settings()
    resolved_model_name = validate_model_name(model_name)
    if resolved_model_name not in RANKING_MODEL_NAMES:
        raise ValueError(f"Model {resolved_model_name} does not support ranking training")

    resolved_random_state = (
        resolved_settings.random_seed if random_state is None else random_state
    )
    return XGBRanker(
        objective="rank:ndcg",
        n_estimators=resolved_settings.xgb_ranker_n_estimators,
        max_depth=resolved_settings.xgb_ranker_max_depth,
        learning_rate=resolved_settings.xgb_ranker_learning_rate,
        subsample=resolved_settings.xgb_ranker_subsample,
        colsample_bytree=resolved_settings.xgb_ranker_colsample_bytree,
        random_state=resolved_random_state,
        tree_method="hist",
    )
