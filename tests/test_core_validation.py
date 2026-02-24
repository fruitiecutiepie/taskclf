"""Tests for core.validation: range, missing rate, timestamp, and distribution checks."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from taskclf.core.validation import (
    Severity,
    ValidationReport,
    validate_feature_dataframe,
)


def _base_row(
    ts: dt.datetime | None = None,
    user_id: str = "u1",
    session_id: str = "s1",
) -> dict:
    """Minimal valid row for building test DataFrames."""
    ts = ts or dt.datetime(2025, 6, 15, 10, 0)
    return {
        "user_id": user_id,
        "device_id": None,
        "session_id": session_id,
        "bucket_start_ts": ts,
        "bucket_end_ts": ts + dt.timedelta(seconds=60),
        "schema_version": "v1",
        "schema_hash": "abc",
        "source_ids": ["test"],
        "app_id": "com.test.App",
        "app_category": "editor",
        "window_title_hash": "hash1",
        "is_browser": False,
        "is_editor": True,
        "is_terminal": False,
        "app_switch_count_last_5m": 2,
        "app_foreground_time_ratio": 0.8,
        "app_change_count": 1,
        "keys_per_min": 60.0,
        "backspace_ratio": 0.05,
        "shortcut_rate": 0.1,
        "clicks_per_min": 5.0,
        "scroll_events_per_min": 2.0,
        "mouse_distance": 300.0,
        "active_seconds_keyboard": 40.0,
        "active_seconds_mouse": 30.0,
        "active_seconds_any": 45.0,
        "max_idle_run_seconds": 10.0,
        "event_density": 2.0,
        "hour_of_day": 10,
        "day_of_week": 6,
        "session_length_so_far": 30.0,
    }


def _make_df(rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [_base_row()])


class TestEmptyDataFrame:
    def test_warns_on_empty(self) -> None:
        report = validate_feature_dataframe(pd.DataFrame())
        assert len(report.warnings) == 1
        assert report.ok


class TestNonNullable:
    def test_catches_null_user_id(self) -> None:
        row = _base_row()
        row["user_id"] = None
        report = validate_feature_dataframe(_make_df([row]))
        errors = [f for f in report.errors if f.check == "non_nullable" and f.column == "user_id"]
        assert len(errors) == 1

    def test_passes_valid(self) -> None:
        report = validate_feature_dataframe(_make_df())
        non_null_errors = [f for f in report.errors if f.check == "non_nullable"]
        assert len(non_null_errors) == 0


class TestMissingRate:
    def test_exceeds_threshold(self) -> None:
        rows = [_base_row(ts=dt.datetime(2025, 6, 15, 10, i)) for i in range(10)]
        for r in rows[:8]:
            r["keys_per_min"] = None
        report = validate_feature_dataframe(_make_df(rows), max_missing_rate=0.5)
        miss_errors = [f for f in report.errors if f.check == "missing_rate" and f.column == "keys_per_min"]
        assert len(miss_errors) == 1

    def test_within_threshold(self) -> None:
        rows = [_base_row(ts=dt.datetime(2025, 6, 15, 10, i)) for i in range(10)]
        for r in rows[:3]:
            r["keys_per_min"] = None
        report = validate_feature_dataframe(_make_df(rows), max_missing_rate=0.5)
        miss_errors = [f for f in report.errors if f.check == "missing_rate" and f.column == "keys_per_min"]
        assert len(miss_errors) == 0


class TestRangeChecks:
    def test_hour_out_of_range(self) -> None:
        row = _base_row()
        row["hour_of_day"] = 25
        report = validate_feature_dataframe(_make_df([row]))
        range_errors = [f for f in report.errors if f.check == "range_max" and f.column == "hour_of_day"]
        assert len(range_errors) == 1

    def test_negative_keys_per_min(self) -> None:
        row = _base_row()
        row["keys_per_min"] = -5.0
        report = validate_feature_dataframe(_make_df([row]))
        range_errors = [f for f in report.errors if f.check == "range_min" and f.column == "keys_per_min"]
        assert len(range_errors) == 1

    def test_foreground_ratio_above_one(self) -> None:
        row = _base_row()
        row["app_foreground_time_ratio"] = 1.5
        report = validate_feature_dataframe(_make_df([row]))
        range_errors = [f for f in report.errors if f.check == "range_max" and f.column == "app_foreground_time_ratio"]
        assert len(range_errors) == 1

    def test_valid_ranges_pass(self) -> None:
        report = validate_feature_dataframe(_make_df())
        range_errors = [f for f in report.errors if f.check.startswith("range_")]
        assert len(range_errors) == 0


class TestBucketEndConsistency:
    def test_mismatch_detected(self) -> None:
        row = _base_row()
        row["bucket_end_ts"] = row["bucket_start_ts"] + dt.timedelta(seconds=90)
        report = validate_feature_dataframe(_make_df([row]))
        errors = [f for f in report.errors if f.check == "bucket_end_consistency"]
        assert len(errors) == 1

    def test_correct_end_passes(self) -> None:
        report = validate_feature_dataframe(_make_df())
        errors = [f for f in report.errors if f.check == "bucket_end_consistency"]
        assert len(errors) == 0


class TestMonotonicTimestamps:
    def test_non_monotonic_detected(self) -> None:
        rows = [
            _base_row(ts=dt.datetime(2025, 6, 15, 10, 2)),
            _base_row(ts=dt.datetime(2025, 6, 15, 10, 1)),
        ]
        report = validate_feature_dataframe(_make_df(rows))
        errors = [f for f in report.errors if f.check == "monotonic_timestamps"]
        assert len(errors) == 1

    def test_monotonic_passes(self) -> None:
        rows = [
            _base_row(ts=dt.datetime(2025, 6, 15, 10, i))
            for i in range(5)
        ]
        report = validate_feature_dataframe(_make_df(rows))
        errors = [f for f in report.errors if f.check == "monotonic_timestamps"]
        assert len(errors) == 0


class TestDistributionWarnings:
    def test_constant_column_warns(self) -> None:
        rows = [_base_row(ts=dt.datetime(2025, 6, 15, 10, i)) for i in range(10)]
        for r in rows:
            r["keys_per_min"] = 42.0
        report = validate_feature_dataframe(_make_df(rows))
        warns = [f for f in report.warnings if f.check == "constant_column" and f.column == "keys_per_min"]
        assert len(warns) == 1


class TestClassBalance:
    def test_imbalanced_class_warns(self) -> None:
        base = dt.datetime(2025, 6, 15, 10, 0)
        rows = [_base_row(ts=base + dt.timedelta(minutes=i)) for i in range(100)]
        for r in rows[:97]:
            r["label"] = "Build"
        for r in rows[97:]:
            r["label"] = "BreakIdle"
        report = validate_feature_dataframe(_make_df(rows))
        warns = [f for f in report.warnings if f.check == "class_balance"]
        assert len(warns) >= 1

    def test_no_label_column_ok(self) -> None:
        report = validate_feature_dataframe(_make_df())
        warns = [f for f in report.warnings if f.check == "class_balance"]
        assert len(warns) == 0


class TestOverallReport:
    def test_valid_data_is_ok(self) -> None:
        row = _base_row()
        row["device_id"] = "dev-01"
        report = validate_feature_dataframe(_make_df([row]))
        assert report.ok, [f.message for f in report.errors]
