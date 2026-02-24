"""Join features with label spans and split by time."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Sequence

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


def _deterministic_holdout_users(
    users: list[str],
    fraction: float,
    seed: str = "taskclf-holdout",
) -> list[str]:
    """Select a deterministic subset of users for holdout.

    Uses a hash-based ordering so the selection is reproducible without
    a random seed, and stable when new users are added.
    """
    if fraction <= 0 or not users:
        return []
    scored = sorted(
        users,
        key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest(),
    )
    k = max(1, int(len(scored) * fraction))
    return scored[:k]


def split_by_time(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_user_fraction: float = 0.0,
) -> dict[str, Any]:
    """Three-way chronological split with optional cross-user holdout.

    For each non-holdout user the rows are sorted by ``bucket_start_ts``
    and split chronologically into train / val / test by the given ratios.
    Holdout users (if any) have *all* their data placed in the test set to
    evaluate cold-start generalization.

    Args:
        df: Labeled feature DataFrame.  Must contain ``bucket_start_ts``
            and ``user_id`` columns.
        train_ratio: Fraction of each user's chronological data for
            training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).  The remainder
            goes to the test set.
        holdout_user_fraction: Fraction of unique users to hold out
            entirely for the test set (default 0 = no holdout).

    Returns:
        A dict with keys ``"train"``, ``"val"``, ``"test"`` (each a list
        of integer indices into *df*), and ``"holdout_users"`` (list of
        held-out user_id strings).

    Raises:
        ValueError: If ``user_id`` column is missing or ratios are invalid.
    """
    if "user_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'user_id' column")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    all_users = sorted(df["user_id"].unique().tolist())

    holdout_users = _deterministic_holdout_users(all_users, holdout_user_fraction)
    holdout_set = set(holdout_users)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for uid, group in df.groupby("user_id", sort=False):
        group = group.sort_values("bucket_start_ts")
        indices = group.index.tolist()

        if uid in holdout_set:
            test_idx.extend(indices)
            continue

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx.extend(indices[:train_end])
        val_idx.extend(indices[train_end:val_end])
        test_idx.extend(indices[val_end:])

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "holdout_users": holdout_users,
    }
