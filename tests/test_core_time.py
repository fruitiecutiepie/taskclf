"""Tests for time-bucket alignment, range generation, and UTC helpers.

Covers: TC-TIME-001 through TC-TIME-010, TC-TIME-011 through TC-TIME-016.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from taskclf.core.time import (
    align_to_bucket,
    ts_utc_aware_get,
    generate_bucket_range,
    to_naive_utc,
)

_UTC = timezone.utc


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Shorthand for ``datetime(..., tzinfo=timezone.utc)``."""
    return datetime(year, month, day, hour, minute, second, tzinfo=_UTC)


def test_tc_time_001_bucket_alignment() -> None:
    """TC-TIME-001: e.g. 12:00:37 -> 12:00:00 for 60s buckets."""
    ts = datetime(2025, 6, 15, 12, 0, 37)
    assert align_to_bucket(ts) == _utc(2025, 6, 15, 12, 0, 0)

    ts2 = datetime(2025, 6, 15, 12, 0, 59)
    assert align_to_bucket(ts2) == _utc(2025, 6, 15, 12, 0, 0)

    ts3 = datetime(2025, 6, 15, 12, 1, 1)
    assert align_to_bucket(ts3) == _utc(2025, 6, 15, 12, 1, 0)


def test_tc_time_002_boundary_on_exact_bucket() -> None:
    """TC-TIME-002: timestamp exactly on boundary stays stable."""
    exact = datetime(2025, 6, 15, 12, 0, 0)
    assert align_to_bucket(exact) == _utc(2025, 6, 15, 12, 0, 0)

    exact2 = datetime(2025, 6, 15, 0, 0, 0)
    assert align_to_bucket(exact2) == _utc(2025, 6, 15, 0, 0, 0)


def test_tc_time_003_day_rollover() -> None:
    """TC-TIME-003: 23:59:30 aligns to 23:59:00, not next day."""
    ts = datetime(2025, 6, 15, 23, 59, 30)
    assert align_to_bucket(ts) == _utc(2025, 6, 15, 23, 59, 0)


def test_tc_time_004_dst_transition() -> None:
    """TC-TIME-004: DST-aware timestamps convert to UTC, no duplicate buckets."""
    eastern_std = timezone(timedelta(hours=-5))
    eastern_dst = timezone(timedelta(hours=-4))

    ts_std = datetime(2025, 3, 9, 1, 30, 15, tzinfo=eastern_std)  # 06:30:15 UTC
    ts_dst = datetime(2025, 3, 9, 3, 30, 15, tzinfo=eastern_dst)  # 07:30:15 UTC

    aligned_std = align_to_bucket(ts_std)
    aligned_dst = align_to_bucket(ts_dst)

    assert aligned_std.tzinfo is _UTC
    assert aligned_dst.tzinfo is _UTC
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
    assert buckets[0] == _utc(2025, 6, 15, 10, 0, 0)
    assert buckets[-1] == _utc(2025, 6, 15, 10, 4, 0)


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
    assert buckets[0] == _utc(2025, 6, 15, 10, 0, 0)
    assert buckets[-1] == _utc(2025, 6, 15, 10, 2, 0)
    assert len(buckets) == 3


def test_tc_time_009_timezone_aware_inputs() -> None:
    """TC-TIME-009: timezone-aware inputs produce aware UTC results."""
    eastern = timezone(timedelta(hours=-5))
    start = datetime(2025, 6, 15, 10, 0, 0, tzinfo=eastern)  # 15:00 UTC
    end = datetime(2025, 6, 15, 10, 3, 0, tzinfo=eastern)  # 15:03 UTC
    buckets = generate_bucket_range(start, end)
    assert len(buckets) == 3
    for b in buckets:
        assert b.tzinfo is _UTC


def test_tc_time_010_custom_bucket_seconds() -> None:
    """TC-TIME-010: custom bucket_seconds=300 produces 5-min buckets."""
    start = datetime(2025, 6, 15, 10, 0, 0)
    end = datetime(2025, 6, 15, 10, 15, 0)
    buckets = generate_bucket_range(start, end, bucket_seconds=300)
    assert len(buckets) == 3
    assert buckets[1] - buckets[0] == timedelta(seconds=300)


# ---------------------------------------------------------------------------
# to_naive_utc
# ---------------------------------------------------------------------------


def test_tc_time_011_to_naive_utc_passthrough() -> None:
    """TC-TIME-011: naive datetime passes through unchanged."""
    naive = datetime(2026, 3, 1, 12, 0, 0)
    result = to_naive_utc(naive)
    assert result == naive
    assert result.tzinfo is None


def test_tc_time_012_to_naive_utc_aware_utc() -> None:
    """TC-TIME-012: aware UTC datetime is stripped of tzinfo."""
    aware = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = to_naive_utc(aware)
    assert result == datetime(2026, 3, 1, 12, 0, 0)
    assert result.tzinfo is None


def test_tc_time_013_to_naive_utc_aware_non_utc() -> None:
    """TC-TIME-013: aware non-UTC datetime is converted to UTC then stripped."""
    eastern = timezone(timedelta(hours=-5))
    aware = datetime(2026, 3, 1, 7, 0, 0, tzinfo=eastern)  # 12:00 UTC
    result = to_naive_utc(aware)
    assert result == datetime(2026, 3, 1, 12, 0, 0)
    assert result.tzinfo is None


# ---------------------------------------------------------------------------
# ts_utc_aware_get
# ---------------------------------------------------------------------------


def test_tc_time_014_ts_utc_aware_get_naive_tagged() -> None:
    """TC-TIME-014: naive datetime is tagged as UTC."""
    naive = datetime(2026, 3, 1, 12, 0, 0)
    result = ts_utc_aware_get(naive)
    assert result.tzinfo is _UTC
    assert result == _utc(2026, 3, 1, 12, 0, 0)


def test_tc_time_015_ts_utc_aware_get_utc_preserved() -> None:
    """TC-TIME-015: aware UTC datetime is returned unchanged."""
    aware = _utc(2026, 3, 1, 12, 0, 0)
    result = ts_utc_aware_get(aware)
    assert result is aware
    assert result.tzinfo is _UTC


def test_tc_time_016_ts_utc_aware_get_non_utc_converted() -> None:
    """TC-TIME-016: aware non-UTC datetime is converted to UTC."""
    eastern = timezone(timedelta(hours=-5))
    aware = datetime(2026, 3, 1, 7, 0, 0, tzinfo=eastern)  # 12:00 UTC
    result = ts_utc_aware_get(aware)
    assert result.tzinfo is _UTC
    assert result == _utc(2026, 3, 1, 12, 0, 0)
