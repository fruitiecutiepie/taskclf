"""Tests for time-bucket alignment and range generation.

Covers: TC-TIME-001 through TC-TIME-010.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from taskclf.core.time import align_to_bucket, generate_bucket_range


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


# ---------------------------------------------------------------------------
# generate_bucket_range
# ---------------------------------------------------------------------------


def test_tc_time_005_basic_range() -> None:
    """TC-TIME-005: 10:00 to 10:05 with 60s buckets produces 5 buckets."""
    start = datetime(2025, 6, 15, 10, 0, 0)
    end = datetime(2025, 6, 15, 10, 5, 0)
    buckets = generate_bucket_range(start, end)
    assert len(buckets) == 5
    assert buckets[0] == datetime(2025, 6, 15, 10, 0, 0)
    assert buckets[-1] == datetime(2025, 6, 15, 10, 4, 0)


def test_tc_time_006_start_equals_end() -> None:
    """TC-TIME-006: start == end produces empty list (exclusive end)."""
    ts = datetime(2025, 6, 15, 10, 0, 0)
    assert generate_bucket_range(ts, ts) == []


def test_tc_time_007_start_after_end() -> None:
    """TC-TIME-007: start > end produces empty list."""
    start = datetime(2025, 6, 15, 10, 5, 0)
    end = datetime(2025, 6, 15, 10, 0, 0)
    assert generate_bucket_range(start, end) == []


def test_tc_time_008_unaligned_inputs() -> None:
    """TC-TIME-008: unaligned start/end are aligned before ranging."""
    start = datetime(2025, 6, 15, 10, 0, 37)
    end = datetime(2025, 6, 15, 10, 3, 12)
    buckets = generate_bucket_range(start, end)
    assert buckets[0] == datetime(2025, 6, 15, 10, 0, 0)
    assert buckets[-1] == datetime(2025, 6, 15, 10, 2, 0)
    assert len(buckets) == 3


def test_tc_time_009_timezone_aware_inputs() -> None:
    """TC-TIME-009: timezone-aware inputs are converted to naive UTC."""
    eastern = timezone(timedelta(hours=-5))
    start = datetime(2025, 6, 15, 10, 0, 0, tzinfo=eastern)  # 15:00 UTC
    end = datetime(2025, 6, 15, 10, 3, 0, tzinfo=eastern)    # 15:03 UTC
    buckets = generate_bucket_range(start, end)
    assert len(buckets) == 3
    for b in buckets:
        assert b.tzinfo is None


def test_tc_time_010_custom_bucket_seconds() -> None:
    """TC-TIME-010: custom bucket_seconds=300 produces 5-min buckets."""
    start = datetime(2025, 6, 15, 10, 0, 0)
    end = datetime(2025, 6, 15, 10, 15, 0)
    buckets = generate_bucket_range(start, end, bucket_seconds=300)
    assert len(buckets) == 3
    assert buckets[1] - buckets[0] == timedelta(seconds=300)
