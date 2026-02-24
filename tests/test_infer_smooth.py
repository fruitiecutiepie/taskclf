"""Tests for inference smoothing and segmentization (infer/smooth.py).

Covers:
- Rolling majority smoothing reduces short spikes
- Segmentization merges adjacent identical labels
- Segments are ordered, non-overlapping, and cover all buckets
- Segment durations match bucket_count * bucket_size
"""

from __future__ import annotations

from datetime import datetime, timedelta

from taskclf.infer.smooth import rolling_majority, segmentize


class TestRollingMajority:
    def test_reduces_spikes(self) -> None:
        """TC-INF-001: rolling majority smoothing reduces short spikes."""
        labels = ["Build", "Build", "BreakIdle", "Build", "Build"]
        smoothed = rolling_majority(labels, window=3)
        assert smoothed[2] == "Build"
        assert len(smoothed) == len(labels)


class TestSegmentize:
    def test_merges_adjacent(self) -> None:
        """TC-INF-002: segmentization merges adjacent identical labels."""
        base = datetime(2025, 6, 15, 10, 0, 0)
        bucket_starts = [base + timedelta(minutes=i) for i in range(5)]
        labels = ["Build", "Build", "Build", "Write", "Write"]

        segs = segmentize(bucket_starts, labels)
        assert len(segs) == 2
        assert segs[0].label == "Build"
        assert segs[0].bucket_count == 3
        assert segs[1].label == "Write"
        assert segs[1].bucket_count == 2

    def test_ordered_nonoverlapping_full_coverage(self) -> None:
        """TC-INF-003: segments are strictly ordered, non-overlapping, cover all predicted buckets."""
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

    def test_durations_match_bucket_counts(self) -> None:
        """TC-INF-004: segment durations match bucket_count * bucket_size."""
        base = datetime(2025, 6, 15, 10, 0, 0)
        bucket_starts = [base + timedelta(minutes=i) for i in range(6)]
        labels = ["Build", "Build", "Build", "BreakIdle", "BreakIdle", "BreakIdle"]
        bucket_seconds = 60

        segs = segmentize(bucket_starts, labels, bucket_seconds=bucket_seconds)
        for seg in segs:
            expected_duration = timedelta(seconds=seg.bucket_count * bucket_seconds)
            assert seg.end_ts - seg.start_ts == expected_duration
