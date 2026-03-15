import sys

import numpy as np
import pandas as pd

from suffering.training.prediction import predict_with_model


class FakeCPUModel:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        return feature_frame.sum(axis=1).to_numpy()


class FakeBooster:
    def __init__(self) -> None:
        self.inplace_predict_calls = 0
        self.last_input = None

    def inplace_predict(self, feature_matrix: object) -> object:
        self.inplace_predict_calls += 1
        self.last_input = feature_matrix
        return feature_matrix


class FakeGPUModel:
    def __init__(self) -> None:
        self.predict_calls = 0
        self.booster = FakeBooster()

    def get_params(self) -> dict[str, str]:
        return {"device": "cuda"}

    def get_booster(self) -> FakeBooster:
        return self.booster

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        return feature_frame.sum(axis=1).to_numpy()


class FakeCuPyArray:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)


class FakeCuPyModule:
    @staticmethod
    def asarray(value: object) -> FakeCuPyArray:
        return FakeCuPyArray(value)

    @staticmethod
    def asnumpy(value: object) -> np.ndarray:
        if isinstance(value, FakeCuPyArray):
            return value.value
        return np.asarray(value)


def test_predict_with_model_uses_model_predict_for_cpu_models() -> None:
    feature_frame = pd.DataFrame({"feature_alpha": [1.0, 2.0], "feature_beta": [3.0, 4.0]})
    model = FakeCPUModel()

    predictions = predict_with_model(model=model, feature_frame=feature_frame)

    assert model.predict_calls == 1
    assert predictions.tolist() == [4.0, 6.0]


def test_predict_with_model_uses_cupy_backed_inplace_predict_for_cuda_xgboost(
    monkeypatch,
) -> None:
    feature_frame = pd.DataFrame({"feature_alpha": [1.0, 2.0], "feature_beta": [3.0, 4.0]})
    model = FakeGPUModel()
    monkeypatch.setitem(sys.modules, "cupy", FakeCuPyModule())

    predictions = predict_with_model(model=model, feature_frame=feature_frame)

    assert model.predict_calls == 0
    assert model.booster.inplace_predict_calls == 1
    assert isinstance(model.booster.last_input, FakeCuPyArray)
    assert predictions.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_predict_with_model_falls_back_to_cpu_when_cupy_is_unavailable(monkeypatch) -> None:
    feature_frame = pd.DataFrame({"feature_alpha": [1.0], "feature_beta": [2.0]})
    model = FakeGPUModel()
    monkeypatch.delitem(sys.modules, "cupy", raising=False)

    predictions = predict_with_model(model=model, feature_frame=feature_frame)

    assert model.predict_calls == 1
    assert model.booster.inplace_predict_calls == 0
    assert predictions.tolist() == [3.0]
