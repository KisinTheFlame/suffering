from pathlib import Path

import pandas as pd

from suffering.config.settings import Settings
from suffering.data.storage import DailyDataStorage
from suffering.features.service import FeatureService
from suffering.ranking.service import RankingService


def build_sample_raw_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=25, freq="B"),
            "open": [100.0 + index for index in range(25)],
            "high": [101.0 + index for index in range(25)],
            "low": [99.0 + index for index in range(25)],
            "close": [100.5 + index for index in range(25)],
            "adj_close": [100.3 + index for index in range(25)],
            "volume": [1_000 + index * 10 for index in range(25)],
            "symbol": [symbol] * 25,
        }
    )


def test_ranking_service_builds_labels_and_panel_dataset_from_cached_data(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, default_symbols=["AAPL", "MSFT"])
    raw_storage = DailyDataStorage(data_dir=tmp_path)
    raw_storage.write_daily_data("AAPL", build_sample_raw_frame("AAPL"))
    raw_storage.write_daily_data("MSFT", build_sample_raw_frame("MSFT"))

    feature_service = FeatureService.from_settings(settings)
    feature_service.build_features_for_symbol("AAPL")
    feature_service.build_features_for_symbol("MSFT")

    service = RankingService.from_settings(settings)
    service.build_label_for_symbol("AAPL")
    service.build_label_for_symbol("MSFT")
    panel_frame = service.build_panel_dataset()

    assert service.storage.label_exists("AAPL")
    assert service.storage.label_exists("MSFT")
    assert service.storage.dataset_exists(settings.default_dataset_name)
    assert not panel_frame.empty
    assert "future_return_5d" in panel_frame.columns
    assert "relevance_5d_5q" in panel_frame.columns


def test_ranking_service_requires_existing_feature_cache_for_dataset_build(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, default_symbols=["AAPL"])
    raw_storage = DailyDataStorage(data_dir=tmp_path)
    raw_storage.write_daily_data("AAPL", build_sample_raw_frame("AAPL"))

    service = RankingService.from_settings(settings)
    service.build_label_for_symbol("AAPL")

    try:
        service.build_panel_dataset()
    except FileNotFoundError as exc:
        assert "suffering feature-build AAPL" in str(exc)
    else:
        raise AssertionError("expected missing feature cache to raise FileNotFoundError")
