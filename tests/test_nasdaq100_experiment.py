from pathlib import Path, PurePosixPath

from suffering.infra import (
    CURRENT_NASDAQ_100_SYMBOLS,
    DATASET_NAME,
    END_DATE,
    FETCH_SYMBOLS,
    RUN_NAME,
    START_DATE,
    build_nasdaq100_current_static_settings,
    local_artifacts_output_dir,
    remote_artifacts_relative_root,
    report_relative_path,
)


def test_nasdaq100_experiment_settings_match_expected_main_run() -> None:
    settings = build_nasdaq100_current_static_settings()

    assert settings.data_dir == Path("data") / RUN_NAME
    assert settings.artifacts_dir == Path("artifacts") / RUN_NAME
    assert settings.default_dataset_name == DATASET_NAME
    assert settings.default_start_date == START_DATE
    assert settings.default_end_date == END_DATE
    assert settings.default_symbols == CURRENT_NASDAQ_100_SYMBOLS


def test_nasdaq100_experiment_paths_point_to_remote_artifact_bundle() -> None:
    assert remote_artifacts_relative_root() == PurePosixPath("artifacts") / RUN_NAME
    assert local_artifacts_output_dir() == Path("artifacts") / "remote_experiments" / RUN_NAME
    assert report_relative_path() == (
        PurePosixPath("reports") / "research" / "xgb_ranker_research_report.md"
    )


def test_nasdaq100_experiment_fetch_symbols_include_benchmark() -> None:
    assert "QQQ" in FETCH_SYMBOLS
    assert len(FETCH_SYMBOLS) == len(CURRENT_NASDAQ_100_SYMBOLS) + 1
