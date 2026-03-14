from suffering.config.settings import Settings
from suffering.data.universe import get_default_universe, resolve_symbols


def test_default_universe_uses_settings_symbols() -> None:
    settings = Settings(default_symbols=["aapl", "msft", "nvda"])

    symbols = get_default_universe(settings)

    assert symbols == ["AAPL", "MSFT", "NVDA"]


def test_resolve_symbols_prefers_explicit_input() -> None:
    settings = Settings(default_symbols=["AAPL"])

    symbols = resolve_symbols(["meta", "googl"], settings)

    assert symbols == ["META", "GOOGL"]
