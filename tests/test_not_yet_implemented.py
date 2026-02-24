"""Placeholder tests for features not yet implemented.

Each test is skipped with a reason pointing to the stub module that needs
implementation.  Remove the skip marker once the feature is built.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# TC-CORE-005: raw title opt-in policy (config gating)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once title_policy config is implemented in core/types.py")
def test_tc_core_005_raw_title_opt_in() -> None:
    """TC-CORE-005: allow raw title only when config title_policy=raw_opt_in."""


# ---------------------------------------------------------------------------
# TC-FEAT-*: advanced feature computation (features/windows.py, features/text.py stubs)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once features/windows.py rolling-window aggregations are implemented")
def test_tc_feat_002_app_switch_count_last_5m() -> None:
    """TC-FEAT-002: app switch counts in last 5 minutes match expected."""


@pytest.mark.skip(reason="TODO: remove .skip once features/build.py handles real idle events")
def test_tc_feat_003_idle_segments_produce_zero_active() -> None:
    """TC-FEAT-003: idle segments produce active_seconds=0 and correct idle flags."""


@pytest.mark.skip(reason="TODO: remove .skip once features/text.py title featurization is implemented")
def test_tc_feat_004_title_featurization_uses_hash_only() -> None:
    """TC-FEAT-004: window title featurization uses hash/tokenization only."""


@pytest.mark.skip(reason="TODO: remove .skip once features/windows.py rolling windows are implemented")
def test_tc_feat_005_rolling_window_start_of_day() -> None:
    """TC-FEAT-005: rolling-window features are consistent at start-of-day."""


# ---------------------------------------------------------------------------
# TC-INF-*: smoothing and segmentization
# ---------------------------------------------------------------------------

def test_tc_inf_001_rolling_majority_reduces_spikes() -> None:
    """TC-INF-001: rolling majority smoothing reduces short spikes."""
    from taskclf.infer.smooth import rolling_majority

    labels = ["Build", "Build", "BreakIdle", "Build", "Build"]
    smoothed = rolling_majority(labels, window=3)
    assert smoothed[2] == "Build"
    assert len(smoothed) == len(labels)


def test_tc_inf_002_segmentization_merges_adjacent() -> None:
    """TC-INF-002: segmentization merges adjacent identical labels."""
    from datetime import datetime, timedelta

    from taskclf.infer.smooth import segmentize

    base = datetime(2025, 6, 15, 10, 0, 0)
    bucket_starts = [base + timedelta(minutes=i) for i in range(5)]
    labels = ["Build", "Build", "Build", "Write", "Write"]

    segs = segmentize(bucket_starts, labels)
    assert len(segs) == 2
    assert segs[0].label == "Build"
    assert segs[0].bucket_count == 3
    assert segs[1].label == "Write"
    assert segs[1].bucket_count == 2


def test_tc_inf_003_segments_ordered_nonoverlapping_full_coverage() -> None:
    """TC-INF-003: segments are strictly ordered, non-overlapping, cover all predicted buckets."""
    from datetime import datetime, timedelta

    from taskclf.infer.smooth import segmentize

    base = datetime(2025, 6, 15, 10, 0, 0)
    n = 10
    bucket_starts = [base + timedelta(minutes=i) for i in range(n)]
    labels = ["Build"] * 3 + ["BreakIdle"] * 2 + ["Write"] * 5

    segs = segmentize(bucket_starts, labels)

    for i in range(len(segs) - 1):
        assert segs[i].end_ts <= segs[i + 1].start_ts
        assert segs[i].end_ts == segs[i + 1].start_ts

    total_buckets = sum(s.bucket_count for s in segs)
    assert total_buckets == n

    assert segs[0].start_ts == bucket_starts[0]
    assert segs[-1].end_ts == bucket_starts[-1] + timedelta(minutes=1)


def test_tc_inf_004_segment_durations_match_bucket_counts() -> None:
    """TC-INF-004: segment durations match bucket_count * bucket_size."""
    from datetime import datetime, timedelta

    from taskclf.infer.smooth import segmentize

    base = datetime(2025, 6, 15, 10, 0, 0)
    bucket_starts = [base + timedelta(minutes=i) for i in range(6)]
    labels = ["Build", "Build", "Build", "BreakIdle", "BreakIdle", "BreakIdle"]
    bucket_seconds = 60

    segs = segmentize(bucket_starts, labels, bucket_seconds=bucket_seconds)
    for seg in segs:
        expected_duration = timedelta(seconds=seg.bucket_count * bucket_seconds)
        assert seg.end_ts - seg.start_ts == expected_duration


# ---------------------------------------------------------------------------
# TC-INT-001..003: adapter ingest integration (adapters are stubs)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once adapters/activitywatch/client.py is implemented")
def test_tc_int_001_ingest_produces_normalized_events() -> None:
    """TC-INT-001: ingest fixture AW export produces normalized events with expected fields."""


@pytest.mark.skip(reason="TODO: remove .skip once adapters/activitywatch/mapping.py is implemented")
def test_tc_int_002_unknown_app_ids_normalized() -> None:
    """TC-INT-002: unknown app ids are normalized to app_id='unknown' with provenance retained."""


@pytest.mark.skip(reason="TODO: remove .skip once adapters/activitywatch/mapping.py is implemented")
def test_tc_int_003_titles_hashed_during_normalization() -> None:
    """TC-INT-003: window titles are hashed/tokenized during normalization if required."""


# ---------------------------------------------------------------------------
# TC-INT-030..031: report generation (report modules are stubs)
# ---------------------------------------------------------------------------

def test_tc_int_030_daily_report_totals_match_active_time() -> None:
    """TC-INT-030: daily report totals sum to total active time (within tolerance)."""
    from datetime import datetime, timedelta

    from taskclf.infer.smooth import Segment
    from taskclf.report.daily import build_daily_report

    base = datetime(2025, 6, 15, 10, 0, 0)
    segments = [
        Segment(start_ts=base, end_ts=base + timedelta(minutes=3), label="Build", bucket_count=3),
        Segment(start_ts=base + timedelta(minutes=3), end_ts=base + timedelta(minutes=5), label="Write", bucket_count=2),
        Segment(start_ts=base + timedelta(minutes=5), end_ts=base + timedelta(minutes=10), label="BreakIdle", bucket_count=5),
    ]

    report = build_daily_report(segments, bucket_seconds=60)

    total_buckets = sum(s.bucket_count for s in segments)
    expected_minutes = total_buckets * 60 / 60.0
    assert abs(report.total_minutes - expected_minutes) < 0.01

    breakdown_sum = sum(report.breakdown.values())
    assert abs(breakdown_sum - report.total_minutes) < 0.01

    assert report.breakdown["Build"] == 3.0
    assert report.breakdown["Write"] == 2.0
    assert report.breakdown["BreakIdle"] == 5.0
    assert report.segments_count == 3


def test_tc_int_031_report_does_not_leak_raw_titles(tmp_path) -> None:
    """TC-INT-031: report does not leak raw titles or sensitive data."""
    import json
    from datetime import datetime, timedelta
    from pathlib import Path

    from taskclf.infer.smooth import Segment
    from taskclf.report.daily import build_daily_report
    from taskclf.report.export import export_report_json

    base = datetime(2025, 6, 15, 10, 0, 0)
    segments = [
        Segment(start_ts=base, end_ts=base + timedelta(minutes=5), label="Build", bucket_count=5),
        Segment(start_ts=base + timedelta(minutes=5), end_ts=base + timedelta(minutes=10), label="BreakIdle", bucket_count=5),
    ]

    report = build_daily_report(segments)
    out_path = tmp_path / "report.json"
    export_report_json(report, out_path)

    raw = out_path.read_text()
    for forbidden in ("raw_keystrokes", "window_title_raw", "clipboard_content", "clipboard"):
        assert forbidden not in raw, f"Sensitive field {forbidden!r} found in report JSON"

    data = json.loads(raw)
    assert "date" in data
    assert "breakdown" in data
    assert "total_minutes" in data
