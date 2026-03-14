from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from suffering.config.settings import Settings
from suffering.training.models import (
    SUPPORTED_MODEL_NAMES,
    build_regressor,
    resolve_model_name,
    validate_model_name,
)


def test_build_regressor_supports_hist_gbr_and_xgb_regressor() -> None:
    settings = Settings(xgb_n_estimators=12)

    hist_model = build_regressor("hist_gbr", settings=settings, random_state=11)
    xgb_model = build_regressor("xgb_regressor", settings=settings, random_state=11)

    assert SUPPORTED_MODEL_NAMES == ("hist_gbr", "xgb_regressor")
    assert isinstance(hist_model, HistGradientBoostingRegressor)
    assert isinstance(xgb_model, XGBRegressor)
    assert hist_model.random_state == 11
    assert xgb_model.get_params()["n_estimators"] == 12
    assert xgb_model.get_params()["tree_method"] == "hist"


def test_resolve_model_name_uses_configured_default() -> None:
    settings = Settings(default_training_model="xgb_regressor")

    assert resolve_model_name(settings=settings) == "xgb_regressor"


def test_validate_model_name_reports_unknown_model_clearly() -> None:
    try:
        validate_model_name("unknown_model")
    except ValueError as exc:
        assert "Unsupported training model: unknown_model" in str(exc)
        assert "hist_gbr, xgb_regressor" in str(exc)
    else:
        raise AssertionError("expected unknown training model to raise ValueError")
