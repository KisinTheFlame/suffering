from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings
from suffering.data.storage import DailyDataStorage
from suffering.features.service import FeatureService


def build_sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=25, freq="B"),
            "open": [100.0 + index for index in range(25)],
            "high": [101.0 + index for index in range(25)],
            "low": [99.0 + index for index in range(25)],
            "close": [100.5 + index for index in range(25)],
            "adj_close": [100.3 + index for index in range(25)],
            "volume": [1_000 + index * 10 for index in range(25)],
            "symbol": ["AAPL"] * 25,
        }
    )


def test_feature_service_builds_features_from_cached_raw_data(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    raw_storage = DailyDataStorage(data_dir=tmp_path)
    raw_storage.write_daily_data("AAPL", build_sample_raw_frame())

    service = FeatureService.from_settings(settings)
    feature_frame = service.build_features_for_symbol("AAPL")

    assert service.storage.exists("AAPL")
    assert len(feature_frame) == 25
    assert "avg_volume_20d" in feature_frame.columns


def test_feature_service_requires_existing_raw_cache(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    service = FeatureService.from_settings(settings)

    try:
        service.build_features_for_symbol("AAPL")
    except FileNotFoundError as exc:
        assert "suffering data-fetch AAPL" in str(exc)
    else:
        raise AssertionError("expected missing raw cache to raise FileNotFoundError")
