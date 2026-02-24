"""Tests for time-bucket alignment: align_to_bucket.

Covers: TC-TIME-001 through TC-TIME-004.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from taskclf.core.time import align_to_bucket


def test_tc_time_001_bucket_alignment() -> None:
    """TC-TIME-001: e.g. 12:00:37 -> 12:00:00 for 60s buckets."""
    ts = datetime(2025, 6, 15, 12, 0, 37)
    assert align_to_bucket(ts) == datetime(2025, 6, 15, 12, 0, 0)

    ts2 = datetime(2025, 6, 15, 12, 0, 59)
    assert align_to_bucket(ts2) == datetime(2025, 6, 15, 12, 0, 0)

    ts3 = datetime(2025, 6, 15, 12, 1, 1)
    assert align_to_bucket(ts3) == datetime(2025, 6, 15, 12, 1, 0)


def test_tc_time_002_boundary_on_exact_bucket() -> None:
    """TC-TIME-002: timestamp exactly on boundary stays stable."""
    exact = datetime(2025, 6, 15, 12, 0, 0)
    assert align_to_bucket(exact) == exact

    exact2 = datetime(2025, 6, 15, 0, 0, 0)
    assert align_to_bucket(exact2) == exact2


def test_tc_time_003_day_rollover() -> None:
    """TC-TIME-003: 23:59:30 aligns to 23:59:00, not next day."""
    ts = datetime(2025, 6, 15, 23, 59, 30)
    assert align_to_bucket(ts) == datetime(2025, 6, 15, 23, 59, 0)


def test_tc_time_004_dst_transition() -> None:
    """TC-TIME-004: DST-aware timestamps convert to UTC, no duplicate buckets."""
    eastern_std = timezone(timedelta(hours=-5))
    eastern_dst = timezone(timedelta(hours=-4))

    ts_std = datetime(2025, 3, 9, 1, 30, 15, tzinfo=eastern_std)   # 06:30:15 UTC
    ts_dst = datetime(2025, 3, 9, 3, 30, 15, tzinfo=eastern_dst)   # 07:30:15 UTC

    aligned_std = align_to_bucket(ts_std)
    aligned_dst = align_to_bucket(ts_dst)

    assert aligned_std.tzinfo is None
    assert aligned_dst.tzinfo is None
    assert aligned_std != aligned_dst
