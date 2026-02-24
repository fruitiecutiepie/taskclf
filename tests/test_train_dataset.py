"""Tests for dataset joining (label assignment) and time-based splitting.

Covers: TC-LABEL-001 (bucket label assignment), TC-EVAL-001 (time-based split).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest

from taskclf.core.types import LabelSpan
from taskclf.train.dataset import assign_labels_to_buckets, split_by_day, split_by_time


def _make_features_df(timestamps: list[dt.datetime]) -> pd.DataFrame:
    """Build a minimal features DataFrame with the given bucket_start_ts values."""
    return pd.DataFrame({"bucket_start_ts": timestamps, "value": range(len(timestamps))})


class TestAssignLabelsToBuckets:
    def test_assigns_correct_label_when_inside_span(self) -> None:
        """TC-LABEL-001: bucket_start_ts inside [start_ts, end_ts) gets the label."""
        ts = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 55),
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="Build",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 1
        assert result.iloc[0]["label"] == "Build"

    def test_drops_rows_outside_any_span(self) -> None:
        ts_inside = dt.datetime(2025, 6, 15, 10, 0)
        ts_outside = dt.datetime(2025, 6, 15, 14, 0)
        features = _make_features_df([ts_inside, ts_outside])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 55),
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="Build",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 1

    def test_first_matching_span_wins(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 55),
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="Build",
                provenance="manual",
            ),
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 50),
                end_ts=dt.datetime(2025, 6, 15, 10, 10),
                label="Write",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert result.iloc[0]["label"] == "Build"

    def test_boundary_start_is_inclusive(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=ts,
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="BreakIdle",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 1
        assert result.iloc[0]["label"] == "BreakIdle"

    def test_boundary_end_is_exclusive(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 5)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 10, 0),
                end_ts=ts,
                label="Build",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 0

    def test_empty_spans_drops_all(self) -> None:
        features = _make_features_df([dt.datetime(2025, 6, 15, 10, 0)])
        result = assign_labels_to_buckets(features, [])
        assert len(result) == 0


class TestSplitByDay:
    def test_last_day_becomes_val(self) -> None:
        """TC-EVAL-001: validation set is the last calendar day."""
        timestamps = [
            dt.datetime(2025, 6, 14, h, 0) for h in range(9, 17)
        ] + [
            dt.datetime(2025, 6, 15, h, 0) for h in range(9, 17)
        ]
        df = pd.DataFrame({"bucket_start_ts": timestamps, "x": range(len(timestamps))})
        train, val = split_by_day(df)

        train_dates = set(train["bucket_start_ts"].dt.date)
        val_dates = set(val["bucket_start_ts"].dt.date)

        assert val_dates == {dt.date(2025, 6, 15)}
        assert train_dates == {dt.date(2025, 6, 14)}

    def test_no_date_overlap(self) -> None:
        """TC-EVAL-001 (extended): train and val must not share any dates."""
        timestamps = [
            dt.datetime(2025, 6, d, 10, 0) for d in range(10, 16)
        ]
        df = pd.DataFrame({"bucket_start_ts": timestamps, "x": range(len(timestamps))})
        train, val = split_by_day(df)

        train_dates = set(train["bucket_start_ts"].dt.date)
        val_dates = set(val["bucket_start_ts"].dt.date)
        assert train_dates.isdisjoint(val_dates)

    def test_single_day_fallback(self) -> None:
        timestamps = [dt.datetime(2025, 6, 15, h, 0) for h in range(9, 17)]
        df = pd.DataFrame({"bucket_start_ts": timestamps, "x": range(len(timestamps))})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            train, val = split_by_day(df)
            assert any("80/20" in str(warning.message) for warning in w)

        assert len(train) + len(val) == len(df)
        assert len(train) > len(val)


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
