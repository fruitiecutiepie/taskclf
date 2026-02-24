"""Tests for time-based dataset splitting.

Covers: TC-EVAL-001 (time-based split).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from taskclf.train.dataset import split_by_time


def _make_multi_user_df(
    users: list[str], rows_per_user: int = 20,
) -> pd.DataFrame:
    """Build a DataFrame with multiple users and chronological timestamps."""
    records = []
    for uid in users:
        for i in range(rows_per_user):
            records.append({
                "user_id": uid,
                "bucket_start_ts": dt.datetime(2025, 6, 15, 9, 0) + dt.timedelta(minutes=i),
                "x": i,
            })
    return pd.DataFrame(records)


class TestSplitByTime:
    def test_three_way_split(self) -> None:
        df = _make_multi_user_df(["u1"], rows_per_user=20)
        result = split_by_time(df)
        assert set(result.keys()) == {"train", "val", "test", "holdout_users"}
        all_idx = set(result["train"] + result["val"] + result["test"])
        assert all_idx == set(range(len(df)))

    def test_no_overlap(self) -> None:
        df = _make_multi_user_df(["u1", "u2"], rows_per_user=30)
        result = split_by_time(df)
        train_set = set(result["train"])
        val_set = set(result["val"])
        test_set = set(result["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_chronological_ordering(self) -> None:
        """Train timestamps must precede val, which precede test."""
        df = _make_multi_user_df(["u1"], rows_per_user=100)
        result = split_by_time(df)
        if result["train"] and result["val"]:
            max_train_ts = df.loc[result["train"], "bucket_start_ts"].max()
            min_val_ts = df.loc[result["val"], "bucket_start_ts"].min()
            assert max_train_ts <= min_val_ts
        if result["val"] and result["test"]:
            max_val_ts = df.loc[result["val"], "bucket_start_ts"].max()
            min_test_ts = df.loc[result["test"], "bucket_start_ts"].min()
            assert max_val_ts <= min_test_ts

    def test_holdout_users_excluded_from_train_val(self) -> None:
        df = _make_multi_user_df([f"u{i}" for i in range(10)], rows_per_user=20)
        result = split_by_time(df, holdout_user_fraction=0.3)
        assert len(result["holdout_users"]) >= 1

        holdout_set = set(result["holdout_users"])
        train_users = set(df.loc[result["train"], "user_id"].unique())
        val_users = set(df.loc[result["val"], "user_id"].unique())
        assert holdout_set.isdisjoint(train_users)
        assert holdout_set.isdisjoint(val_users)

    def test_deterministic(self) -> None:
        df = _make_multi_user_df([f"u{i}" for i in range(5)], rows_per_user=20)
        r1 = split_by_time(df, holdout_user_fraction=0.2)
        r2 = split_by_time(df, holdout_user_fraction=0.2)
        assert r1["train"] == r2["train"]
        assert r1["holdout_users"] == r2["holdout_users"]

    def test_missing_user_id_raises(self) -> None:
        df = pd.DataFrame({
            "bucket_start_ts": [dt.datetime(2025, 6, 15, 10, 0)],
            "x": [1],
        })
        with pytest.raises(ValueError, match="user_id"):
            split_by_time(df)

    def test_ratios_exceeded_raises(self) -> None:
        df = _make_multi_user_df(["u1"])
        with pytest.raises(ValueError, match="train_ratio"):
            split_by_time(df, train_ratio=0.8, val_ratio=0.3)
