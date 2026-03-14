from suffering import PROJECT_NAME
from suffering.cli import main
from suffering.config.settings import Settings, get_settings


def test_package_can_be_imported() -> None:
    assert PROJECT_NAME == "suffering"


def test_settings_can_be_created() -> None:
    settings = Settings()

    assert settings.app_env
    assert settings.data_dir
    assert get_settings().artifacts_dir
    assert settings.default_data_provider == "yfinance"


def test_cli_main_can_be_called(capsys) -> None:
    exit_code = main(["doctor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "project: suffering" in captured.out
    assert "minimal data and feature layers available" in captured.out
