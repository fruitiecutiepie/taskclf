"""Join features with label spans and split into train/val by day."""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd

from taskclf.core.defaults import DEFAULT_TRAIN_SPLIT_RATIO
from taskclf.core.types import LabelSpan


def assign_labels_to_buckets(
    features_df: pd.DataFrame,
    label_spans: Sequence[LabelSpan],
) -> pd.DataFrame:
    """Assign a ``label`` column to *features_df* from covering *label_spans*.

    For each feature row, the first span whose ``[start_ts, end_ts)``
    interval contains the row's ``bucket_start_ts`` wins.  Rows with no
    covering span are dropped.

    Args:
        features_df: Feature DataFrame with a ``bucket_start_ts`` column.
        label_spans: Label spans to match against feature timestamps.

    Returns:
        A copy of *features_df* with an added ``label`` column, containing
        only the rows that had a covering span.
    """
    if not label_spans:
        result = features_df.copy()
        result["label"] = None
        return result.dropna(subset=["label"]).reset_index(drop=True)

    labels_df = pd.DataFrame(
        [{"start_ts": s.start_ts, "end_ts": s.end_ts, "label": s.label} for s in label_spans]
    )

    assigned: list[str | None] = [None] * len(features_df)
    ts_values = features_df["bucket_start_ts"].values

    for idx, ts in enumerate(ts_values):
        ts_pd = pd.Timestamp(ts)
        mask = (labels_df["start_ts"] <= ts_pd) & (ts_pd < labels_df["end_ts"])
        matches = labels_df.loc[mask, "label"]
        if not matches.empty:
            assigned[idx] = matches.iloc[0]

    result = features_df.copy()
    result["label"] = assigned
    return result.dropna(subset=["label"]).reset_index(drop=True)


def split_by_day(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val by calendar day.

    The last unique day becomes the validation set.  If there is only one
    day, fall back to an 80/20 chronological split and emit a warning.

    Args:
        df: Labeled feature DataFrame with a ``bucket_start_ts`` column.

    Returns:
        A ``(train_df, val_df)`` tuple of DataFrames.
    """
    df = df.sort_values("bucket_start_ts").reset_index(drop=True)
    days = df["bucket_start_ts"].dt.date.unique()

    if len(days) < 2:
        warnings.warn(
            "Only one day of data — using 80/20 chronological split instead of by-day.",
            stacklevel=2,
        )
        split_idx = int(len(df) * DEFAULT_TRAIN_SPLIT_RATIO)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    val_day = days[-1]
    is_val = df["bucket_start_ts"].dt.date == val_day
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
