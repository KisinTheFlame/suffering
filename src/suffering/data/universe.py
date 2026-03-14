"""Small and replaceable symbol universe helpers."""

from __future__ import annotations

from collections.abc import Iterable

from suffering.config.settings import Settings, get_settings
from suffering.data.models import normalize_symbols

STATIC_DEFAULT_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]


def get_default_universe(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    if resolved_settings.default_symbols:
        return normalize_symbols(resolved_settings.default_symbols)
    return list(STATIC_DEFAULT_UNIVERSE)


def resolve_symbols(symbols: Iterable[str] | None, settings: Settings | None = None) -> list[str]:
    if symbols:
        return normalize_symbols(symbols)
    return get_default_universe(settings=settings)
