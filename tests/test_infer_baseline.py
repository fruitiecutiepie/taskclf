"""Tests for the rule-based baseline classifier.

Covers: rule correctness, priority ordering, acceptance gates, integration
with smoothing/segmentization, and reject-rate computation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from taskclf.core.defaults import (
    BASELINE_IDLE_ACTIVE_THRESHOLD,
    BASELINE_IDLE_RUN_THRESHOLD,
    BASELINE_KEYS_HIGH,
    BASELINE_KEYS_LOW,
    BASELINE_SCROLL_HIGH,
    BASELINE_SHORTCUT_HIGH,
    MIXED_UNKNOWN,
)
from taskclf.core.metrics import compare_baselines, reject_rate
from taskclf.core.types import CoreLabel
from taskclf.infer.baseline import (
    classify_single_row,
    predict_baseline,
    run_baseline_inference,
)


def _make_row(**overrides) -> pd.Series:
    """Build a single feature row as a pandas Series with sensible defaults."""
    base = {
        "bucket_start_ts": dt.datetime(2025, 6, 15, 10, 0),
        "bucket_end_ts": dt.datetime(2025, 6, 15, 10, 1),
        "user_id": "test-user",
        "session_id": "s1",
        "app_id": "com.example.App",
        "app_category": "other",
        "is_browser": False,
        "is_editor": False,
        "is_terminal": False,
        "keys_per_min": 20.0,
        "backspace_ratio": 0.05,
        "shortcut_rate": 0.5,
        "clicks_per_min": 5.0,
        "scroll_events_per_min": 1.0,
        "mouse_distance": 500.0,
        "active_seconds_keyboard": 30.0,
        "active_seconds_mouse": 20.0,
        "active_seconds_any": 40.0,
        "max_idle_run_seconds": 10.0,
        "event_density": 2.0,
        "app_switch_count_last_5m": 3,
        "app_foreground_time_ratio": 0.8,
        "app_change_count": 1,
        "hour_of_day": 10,
        "day_of_week": 2,
        "session_length_so_far": 60.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of override dicts."""
    return pd.DataFrame([_make_row(**r) for r in rows])


# ── Rule 1: BreakIdle ──────────────────────────────────────────────────────

class TestBreakIdleRule:
    def test_null_active_seconds(self) -> None:
        row = _make_row(active_seconds_any=None)
        assert classify_single_row(row) == CoreLabel.BreakIdle

    def test_low_active_seconds(self) -> None:
        row = _make_row(active_seconds_any=2.0)
        assert classify_single_row(row) == CoreLabel.BreakIdle

    def test_exactly_at_threshold_is_not_idle(self) -> None:
        row = _make_row(active_seconds_any=BASELINE_IDLE_ACTIVE_THRESHOLD)
        assert classify_single_row(row) != CoreLabel.BreakIdle

    def test_high_idle_run(self) -> None:
        row = _make_row(
            active_seconds_any=30.0,
            max_idle_run_seconds=55.0,
        )
        assert classify_single_row(row) == CoreLabel.BreakIdle

    def test_nan_active_seconds(self) -> None:
        row = _make_row(active_seconds_any=float("nan"))
        assert classify_single_row(row) == CoreLabel.BreakIdle


# ── Rule 2: ReadResearch ───────────────────────────────────────────────────

class TestReadResearchRule:
    def test_browser_high_scroll_low_keys(self) -> None:
        row = _make_row(
            is_browser=True,
            scroll_events_per_min=8.0,
            keys_per_min=3.0,
        )
        assert classify_single_row(row) == CoreLabel.ReadResearch

    def test_browser_low_scroll_is_not_read(self) -> None:
        row = _make_row(
            is_browser=True,
            scroll_events_per_min=1.0,
            keys_per_min=3.0,
        )
        assert classify_single_row(row) != CoreLabel.ReadResearch

    def test_browser_high_keys_is_not_read(self) -> None:
        row = _make_row(
            is_browser=True,
            scroll_events_per_min=8.0,
            keys_per_min=40.0,
        )
        assert classify_single_row(row) != CoreLabel.ReadResearch

    def test_non_browser_is_not_read(self) -> None:
        row = _make_row(
            is_browser=False,
            scroll_events_per_min=8.0,
            keys_per_min=3.0,
        )
        assert classify_single_row(row) != CoreLabel.ReadResearch


# ── Rule 3: Build ─────────────────────────────────────────────────────────

class TestBuildRule:
    def test_editor_high_keys_high_shortcuts(self) -> None:
        row = _make_row(
            is_editor=True,
            keys_per_min=50.0,
            shortcut_rate=3.0,
        )
        assert classify_single_row(row) == CoreLabel.Build

    def test_terminal_high_keys_high_shortcuts(self) -> None:
        row = _make_row(
            is_terminal=True,
            keys_per_min=50.0,
            shortcut_rate=3.0,
        )
        assert classify_single_row(row) == CoreLabel.Build

    def test_editor_low_keys_is_not_build(self) -> None:
        row = _make_row(
            is_editor=True,
            keys_per_min=10.0,
            shortcut_rate=3.0,
        )
        assert classify_single_row(row) != CoreLabel.Build

    def test_editor_low_shortcuts_is_not_build(self) -> None:
        row = _make_row(
            is_editor=True,
            keys_per_min=50.0,
            shortcut_rate=0.3,
        )
        assert classify_single_row(row) != CoreLabel.Build

    def test_non_editor_non_terminal_is_not_build(self) -> None:
        row = _make_row(
            is_editor=False,
            is_terminal=False,
            keys_per_min=50.0,
            shortcut_rate=3.0,
        )
        assert classify_single_row(row) != CoreLabel.Build


# ── Rule 4: fallback ──────────────────────────────────────────────────────

class TestFallback:
    def test_no_rule_matches_gives_mixed_unknown(self) -> None:
        row = _make_row()
        assert classify_single_row(row) == MIXED_UNKNOWN

    def test_chat_app_gives_mixed_unknown(self) -> None:
        row = _make_row(app_category="chat", keys_per_min=15.0)
        assert classify_single_row(row) == MIXED_UNKNOWN


# ── Priority ordering ─────────────────────────────────────────────────────

class TestPriority:
    def test_idle_browser_is_break_not_read(self) -> None:
        """An idle browser window must be BreakIdle, not ReadResearch."""
        row = _make_row(
            is_browser=True,
            scroll_events_per_min=8.0,
            keys_per_min=3.0,
            active_seconds_any=1.0,
        )
        assert classify_single_row(row) == CoreLabel.BreakIdle

    def test_idle_editor_is_break_not_build(self) -> None:
        """An idle editor must be BreakIdle, not Build."""
        row = _make_row(
            is_editor=True,
            keys_per_min=50.0,
            shortcut_rate=3.0,
            active_seconds_any=0.0,
        )
        assert classify_single_row(row) == CoreLabel.BreakIdle


# ── Acceptance gate ────────────────────────────────────────────────────────

class TestAcceptanceGate:
    def test_break_idle_never_classified_as_build(self) -> None:
        """Rows with near-zero activity must never produce Build."""
        idle_rows = _make_df([
            {"active_seconds_any": 0.0, "is_editor": True, "keys_per_min": 50.0, "shortcut_rate": 3.0},
            {"active_seconds_any": None, "is_terminal": True, "keys_per_min": 60.0, "shortcut_rate": 5.0},
            {"active_seconds_any": 2.0, "is_editor": True, "keys_per_min": 80.0, "shortcut_rate": 4.0},
        ])
        labels = predict_baseline(idle_rows)
        for lbl in labels:
            assert lbl != CoreLabel.Build
            assert lbl != CoreLabel.Write

    def test_break_idle_never_classified_as_write(self) -> None:
        idle_rows = _make_df([
            {"active_seconds_any": 0.0},
            {"active_seconds_any": None},
            {"active_seconds_any": 1.0},
        ])
        labels = predict_baseline(idle_rows)
        for lbl in labels:
            assert lbl != CoreLabel.Write


# ── Batch predict_baseline ─────────────────────────────────────────────────

class TestPredictBaseline:
    def test_returns_one_label_per_row(self) -> None:
        df = _make_df([{}, {}, {}])
        labels = predict_baseline(df)
        assert len(labels) == len(df)

    def test_empty_df_returns_empty(self) -> None:
        df = pd.DataFrame()
        labels = predict_baseline(df)
        assert labels == []

    def test_mixed_rules(self) -> None:
        df = _make_df([
            {"active_seconds_any": 0.0},
            {"is_browser": True, "scroll_events_per_min": 8.0, "keys_per_min": 3.0},
            {"is_editor": True, "keys_per_min": 50.0, "shortcut_rate": 3.0},
            {},
        ])
        labels = predict_baseline(df)
        assert labels == [
            CoreLabel.BreakIdle,
            CoreLabel.ReadResearch,
            CoreLabel.Build,
            MIXED_UNKNOWN,
        ]


# ── Integration: run_baseline_inference ────────────────────────────────────

class TestRunBaselineInference:
    def test_produces_smoothed_labels_and_segments(self) -> None:
        base_ts = dt.datetime(2025, 6, 15, 10, 0)
        rows = []
        for i in range(10):
            rows.append({
                "bucket_start_ts": base_ts + dt.timedelta(minutes=i),
                "bucket_end_ts": base_ts + dt.timedelta(minutes=i + 1),
                "active_seconds_any": 0.0 if i < 3 else 30.0,
                "is_editor": True if 3 <= i < 7 else False,
                "keys_per_min": 50.0 if 3 <= i < 7 else 5.0,
                "shortcut_rate": 3.0 if 3 <= i < 7 else 0.1,
            })
        df = _make_df(rows)

        smoothed, segments = run_baseline_inference(df)
        assert len(smoothed) == 10
        assert len(segments) >= 1
        for seg in segments:
            assert seg.start_ts < seg.end_ts
            assert seg.bucket_count >= 1

    def test_all_idle_produces_single_segment(self) -> None:
        base_ts = dt.datetime(2025, 6, 15, 10, 0)
        rows = [
            {"bucket_start_ts": base_ts + dt.timedelta(minutes=i),
             "bucket_end_ts": base_ts + dt.timedelta(minutes=i + 1),
             "active_seconds_any": 0.0}
            for i in range(5)
        ]
        df = _make_df(rows)
        smoothed, segments = run_baseline_inference(df)
        assert all(lbl == CoreLabel.BreakIdle for lbl in smoothed)
        assert len(segments) == 1
        assert segments[0].label == CoreLabel.BreakIdle


# ── Reject rate + comparison metrics ───────────────────────────────────────

class TestMetrics:
    def test_reject_rate_empty(self) -> None:
        assert reject_rate([]) == 0.0

    def test_reject_rate_all_reject(self) -> None:
        assert reject_rate([MIXED_UNKNOWN, MIXED_UNKNOWN]) == 1.0

    def test_reject_rate_none_reject(self) -> None:
        assert reject_rate(["Build", "Debug"]) == 0.0

    def test_reject_rate_partial(self) -> None:
        labels = ["Build", MIXED_UNKNOWN, "Debug", MIXED_UNKNOWN]
        assert reject_rate(labels) == pytest.approx(0.5)

    def test_compare_baselines_produces_per_method_metrics(self) -> None:
        y_true = ["Build", "BreakIdle", "ReadResearch", "Build"]
        preds = {
            "baseline": ["Build", "BreakIdle", MIXED_UNKNOWN, MIXED_UNKNOWN],
            "oracle": y_true,
        }
        label_names = sorted(set(y_true))
        result = compare_baselines(y_true, preds, label_names)

        assert "baseline" in result
        assert "oracle" in result
        assert result["oracle"]["macro_f1"] >= result["baseline"]["macro_f1"]
        assert result["baseline"]["reject_rate"] > 0
        assert result["oracle"]["reject_rate"] == 0.0
