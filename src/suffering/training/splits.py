"""Single time-ordered split helpers for panel datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN


@dataclass(frozen=True)
class PanelDatasetSplit:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame


def build_frame_date_summary(frame: pd.DataFrame) -> dict[str, int | str | None]:
    if frame.empty:
        return {"rows": 0, "date_count": 0, "date_start": None, "date_end": None}

    dates = pd.to_datetime(frame[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    return {
        "rows": int(len(frame)),
        "date_count": int(dates.nunique()),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
    }


def split_panel_dataset_by_date(
    frame: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> PanelDatasetSplit:
    """Split a panel dataset by ordered trading dates without shuffling rows."""
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")

    normalized = frame.copy()
    normalized[DATE_COLUMN] = (
        pd.to_datetime(normalized[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    )
    unique_dates = pd.Index(sorted(normalized[DATE_COLUMN].dropna().unique()))
    if len(unique_dates) < 3:
        raise ValueError("At least 3 unique dates are required for train/validation/test splits")

    train_end, validation_end = _resolve_split_boundaries(
        date_count=len(unique_dates),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]
    _validate_ordered_dates(
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
    )

    return PanelDatasetSplit(
        train_frame=_slice_frame_by_dates(normalized, train_dates),
        validation_frame=_slice_frame_by_dates(normalized, validation_dates),
        test_frame=_slice_frame_by_dates(normalized, test_dates),
    )


def _resolve_split_boundaries(
    date_count: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    train_end = max(1, int(date_count * train_ratio))
    validation_end = max(train_end + 1, int(date_count * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, date_count - 1)

    if train_end >= validation_end:
        train_end = max(1, validation_end - 1)
    if validation_end >= date_count:
        validation_end = date_count - 1

    return train_end, validation_end


def _validate_ordered_dates(
    train_dates: pd.Index,
    validation_dates: pd.Index,
    test_dates: pd.Index,
) -> None:
    if len(train_dates) == 0 or len(validation_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Each split must contain at least one unique date")
    if len(train_dates.intersection(validation_dates)) > 0:
        raise ValueError("Train and validation dates overlap")
    if len(train_dates.intersection(test_dates)) > 0:
        raise ValueError("Train and test dates overlap")
    if len(validation_dates.intersection(test_dates)) > 0:
        raise ValueError("Validation and test dates overlap")
    if not (train_dates.max() < validation_dates.min() < test_dates.min()):
        raise ValueError("Split dates must be strictly increasing")


def _slice_frame_by_dates(frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    return (
        frame.loc[frame[DATE_COLUMN].isin(dates)]
        .sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable")
        .reset_index(drop=True)
    )
