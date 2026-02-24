"""Tests for report generation (daily summaries and JSON export)."""

from __future__ import annotations


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
