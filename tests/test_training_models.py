from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRanker, XGBRegressor

from suffering.config.settings import Settings
from suffering.training.models import (
    SUPPORTED_MODEL_NAMES,
    build_ranker,
    build_regressor,
    resolve_model_name,
    resolve_model_task,
    validate_model_name,
)


def test_build_regressor_and_ranker_support_supported_models() -> None:
    settings = Settings(xgb_n_estimators=12, xgb_ranker_n_estimators=9)

    hist_model = build_regressor("hist_gbr", settings=settings, random_state=11)
    xgb_model = build_regressor("xgb_regressor", settings=settings, random_state=11)
    rank_model = build_ranker("xgb_ranker", settings=settings, random_state=11)

    assert SUPPORTED_MODEL_NAMES == ("hist_gbr", "xgb_regressor", "xgb_ranker")
    assert isinstance(hist_model, HistGradientBoostingRegressor)
    assert isinstance(xgb_model, XGBRegressor)
    assert isinstance(rank_model, XGBRanker)
    assert hist_model.random_state == 11
    assert xgb_model.get_params()["n_estimators"] == 12
    assert xgb_model.get_params()["tree_method"] == "hist"
    assert rank_model.get_params()["n_estimators"] == 9
    assert rank_model.get_params()["objective"] == "rank:ndcg"


def test_resolve_model_name_uses_configured_default() -> None:
    settings = Settings(default_training_model="xgb_regressor")

    assert resolve_model_name(settings=settings) == "xgb_regressor"


def test_validate_model_name_reports_unknown_model_clearly() -> None:
    try:
        validate_model_name("unknown_model")
    except ValueError as exc:
        assert "Unsupported training model: unknown_model" in str(exc)
        assert "hist_gbr, xgb_regressor, xgb_ranker" in str(exc)
    else:
        raise AssertionError("expected unknown training model to raise ValueError")


def test_resolve_model_task_reports_ranking_for_xgb_ranker() -> None:
    assert resolve_model_task("hist_gbr") == "regression"
    assert resolve_model_task("xgb_ranker") == "ranking"
