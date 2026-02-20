"""Tests for dataset joining (label assignment) and time-based splitting.

Covers: TC-LABEL-001 (bucket label assignment), TC-EVAL-001 (time-based split).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest

from taskclf.core.types import LabelSpan
from taskclf.train.dataset import assign_labels_to_buckets, split_by_day


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
                label="coding",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 1
        assert result.iloc[0]["label"] == "coding"

    def test_drops_rows_outside_any_span(self) -> None:
        ts_inside = dt.datetime(2025, 6, 15, 10, 0)
        ts_outside = dt.datetime(2025, 6, 15, 14, 0)
        features = _make_features_df([ts_inside, ts_outside])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 55),
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="coding",
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
                label="coding",
                provenance="manual",
            ),
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 9, 50),
                end_ts=dt.datetime(2025, 6, 15, 10, 10),
                label="writing_docs",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert result.iloc[0]["label"] == "coding"

    def test_boundary_start_is_inclusive(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=ts,
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="break_idle",
                provenance="manual",
            ),
        ]
        result = assign_labels_to_buckets(features, spans)
        assert len(result) == 1
        assert result.iloc[0]["label"] == "break_idle"

    def test_boundary_end_is_exclusive(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 5)
        features = _make_features_df([ts])
        spans = [
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 10, 0),
                end_ts=ts,
                label="coding",
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
