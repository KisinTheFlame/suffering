"""Minimal data access helpers for market data workflows."""

from suffering.data.service import DailyDataService, build_data_service
from suffering.data.universe import get_default_universe

__all__ = ["DailyDataService", "build_data_service", "get_default_universe"]
