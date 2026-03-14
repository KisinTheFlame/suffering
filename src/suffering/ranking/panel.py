"""Panel dataset assembly helpers for ranking research."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from suffering.data.models import DATE_COLUMN, SYMBOL_COLUMN
from suffering.ranking.labels import FUTURE_RETURN_5D_COLUMN

RELEVANCE_5D_5Q_COLUMN = "relevance_5d_5q"
PANEL_INDEX_COLUMNS = [DATE_COLUMN, SYMBOL_COLUMN]


def build_daily_panel_dataset(
    feature_frames: Sequence[pd.DataFrame],
    label_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Merge cached features and labels into a minimal panel dataset."""
    features = _concat_feature_frames(feature_frames)
    labels = _concat_label_frames(label_frames)

    feature_columns = [
        column
        for column in features.columns
        if column not in PANEL_INDEX_COLUMNS
    ]

    if features.empty or labels.empty:
        return pd.DataFrame(
            columns=[
                DATE_COLUMN,
                SYMBOL_COLUMN,
                *feature_columns,
                FUTURE_RETURN_5D_COLUMN,
                RELEVANCE_5D_5Q_COLUMN,
            ]
        )

    merged = features.merge(
        labels.loc[:, [DATE_COLUMN, SYMBOL_COLUMN, FUTURE_RETURN_5D_COLUMN]],
        on=[DATE_COLUMN, SYMBOL_COLUMN],
        how="inner",
        sort=False,
    )

    # Supervised samples must have a realized future return; feature NaN values stay untouched.
    merged = merged.loc[merged[FUTURE_RETURN_5D_COLUMN].notna()].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                DATE_COLUMN,
                SYMBOL_COLUMN,
                *feature_columns,
                FUTURE_RETURN_5D_COLUMN,
                RELEVANCE_5D_5Q_COLUMN,
            ]
        )

    merged = merged.sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable").reset_index(drop=True)
    merged[RELEVANCE_5D_5Q_COLUMN] = _build_relevance_labels(merged)

    return merged.loc[
        :,
        [
            DATE_COLUMN,
            SYMBOL_COLUMN,
            *feature_columns,
            FUTURE_RETURN_5D_COLUMN,
            RELEVANCE_5D_5Q_COLUMN,
        ],
    ]


def _concat_feature_frames(feature_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not feature_frames:
        return pd.DataFrame(columns=PANEL_INDEX_COLUMNS)

    _validate_frames(feature_frames, required_columns=PANEL_INDEX_COLUMNS, frame_type="feature")
    features = pd.concat(feature_frames, ignore_index=True)
    features[DATE_COLUMN] = (
        pd.to_datetime(features[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    )
    features[SYMBOL_COLUMN] = features[SYMBOL_COLUMN].astype(str).str.strip().str.upper()
    return features.sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable").reset_index(drop=True)


def _concat_label_frames(label_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not label_frames:
        return pd.DataFrame(columns=[*PANEL_INDEX_COLUMNS, FUTURE_RETURN_5D_COLUMN])

    _validate_frames(
        label_frames,
        required_columns=[*PANEL_INDEX_COLUMNS, FUTURE_RETURN_5D_COLUMN],
        frame_type="label",
    )
    labels = pd.concat(label_frames, ignore_index=True)
    labels[DATE_COLUMN] = pd.to_datetime(labels[DATE_COLUMN]).dt.tz_localize(None).dt.normalize()
    labels[SYMBOL_COLUMN] = labels[SYMBOL_COLUMN].astype(str).str.strip().str.upper()
    return labels.sort_values([DATE_COLUMN, SYMBOL_COLUMN], kind="stable").reset_index(drop=True)


def _validate_frames(
    frames: Sequence[pd.DataFrame],
    required_columns: Sequence[str],
    frame_type: str,
) -> None:
    for frame in frames:
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Missing expected columns for {frame_type} panel assembly: {missing_text}"
            )


def _build_relevance_labels(frame: pd.DataFrame, bucket_count: int = 5) -> pd.Series:
    relevance = pd.Series(index=frame.index, dtype="int64")

    for _, group in frame.groupby(DATE_COLUMN, sort=False):
        ranked_group = group.sort_values(
            [FUTURE_RETURN_5D_COLUMN, SYMBOL_COLUMN],
            kind="stable",
        )
        symbol_count = len(ranked_group)
        effective_bucket_count = min(bucket_count, symbol_count)
        bucket_labels = (
            (pd.RangeIndex(symbol_count) * effective_bucket_count) // symbol_count
        ).astype("int64")
        relevance.loc[ranked_group.index] = bucket_labels

    return relevance.astype("int64")
