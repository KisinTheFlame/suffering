"""Prediction helpers that keep XGBoost inference on the configured device when possible."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd


def predict_with_model(
    model: Any,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Run inference, preferring GPU-native XGBoost prediction when configured and available."""
    if _should_use_cuda_prediction(model):
        cupy = _load_cupy()
        if cupy is not None:
            feature_matrix = cupy.asarray(feature_frame.to_numpy(dtype="float32", copy=False))
            predictions = model.get_booster().inplace_predict(feature_matrix)
            return np.asarray(cupy.asnumpy(predictions))

    return np.asarray(model.predict(feature_frame))


def _should_use_cuda_prediction(model: Any) -> bool:
    if not hasattr(model, "get_booster"):
        return False

    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        return False

    device = str(get_params().get("device", "")).strip().lower()
    return device.startswith("cuda")


def _load_cupy() -> Any | None:
    try:
        return import_module("cupy")
    except ImportError:
        return None
