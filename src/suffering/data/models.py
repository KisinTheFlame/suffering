"""Shared constants for the minimal daily data layer."""

from __future__ import annotations

from collections.abc import Iterable

DATE_COLUMN = "date"
SYMBOL_COLUMN = "symbol"

DAILY_PRICE_COLUMNS = [
    DATE_COLUMN,
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    SYMBOL_COLUMN,
]


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    return [normalize_symbol(symbol) for symbol in symbols if symbol.strip()]
