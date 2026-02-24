"""Time-based dataset splitting utilities."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


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
