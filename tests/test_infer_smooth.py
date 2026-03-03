"""Tests for inference smoothing and segmentization (infer/smooth.py).

Covers:
- Rolling majority smoothing reduces short spikes
- Rolling majority edge cases (empty, single, all-identical, tie-breaking, window=1)
- Segmentization merges adjacent identical labels
- Segments are ordered, non-overlapping, and cover all buckets
- Segment durations match bucket_count * bucket_size
- Segmentize edge cases (mismatched lengths, empty, single bucket)
- Flap rate computation
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from taskclf.infer.smooth import flap_rate, rolling_majority, segmentize


class TestRollingMajority:
    def test_reduces_spikes(self) -> None:
        """TC-INF-001: rolling majority smoothing reduces short spikes."""
        labels = ["Build", "Build", "BreakIdle", "Build", "Build"]
        smoothed = rolling_majority(labels, window=3)
        assert smoothed[2] == "Build"
        assert len(smoothed) == len(labels)

    def test_empty_list(self) -> None:
        """TC-SMOOTH-007: empty input returns empty output."""
        assert rolling_majority([]) == []

    def test_single_element(self) -> None:
        """TC-SMOOTH-008: single element returned as-is."""
        assert rolling_majority(["Build"]) == ["Build"]

    def test_all_identical(self) -> None:
        """TC-SMOOTH-009: all identical labels remain unchanged."""
        labels = ["Write"] * 10
        assert rolling_majority(labels) == labels

    def test_tie_keeps_original(self) -> None:
        """TC-SMOOTH-010: when counts tie, original label preserved."""
        labels = ["X", "A", "B"]
        smoothed = rolling_majority(labels, window=3)
        assert smoothed[1] == "A"

    def test_window_one_no_smoothing(self) -> None:
        """TC-SMOOTH-011: window=1 produces output identical to input."""
        labels = ["Build", "Write", "BreakIdle", "Build"]
        assert rolling_majority(labels, window=1) == labels


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

    def test_mismatched_lengths_raises(self) -> None:
        """TC-SMOOTH-012: mismatched bucket_starts/labels raises ValueError."""
        base = datetime(2025, 6, 15, 10, 0, 0)
        with pytest.raises(ValueError, match="same length"):
            segmentize(
                [base, base + timedelta(minutes=1)],
                ["Build"],
            )

    def test_empty_inputs(self) -> None:
        """TC-SMOOTH-013: empty inputs return empty list."""
        assert segmentize([], []) == []

    def test_single_bucket(self) -> None:
        """TC-SMOOTH-014: single bucket produces one segment with bucket_count=1."""
        base = datetime(2025, 6, 15, 10, 0, 0)
        segs = segmentize([base], ["Build"])
        assert len(segs) == 1
        assert segs[0].bucket_count == 1
        assert segs[0].label == "Build"
        assert segs[0].start_ts == base
        assert segs[0].end_ts == base + timedelta(seconds=60)


class TestFlapRate:
    def test_all_same(self) -> None:
        """TC-SMOOTH-001: all same labels → 0.0."""
        assert flap_rate(["Build"] * 5) == 0.0

    def test_all_different(self) -> None:
        """TC-SMOOTH-002: all different labels → (n-1)/n."""
        result = flap_rate(["A", "B", "C", "D"])
        assert result == pytest.approx(3 / 4)

    def test_single_element(self) -> None:
        """TC-SMOOTH-003: single element → 0.0."""
        assert flap_rate(["Build"]) == 0.0

    def test_empty_sequence(self) -> None:
        """TC-SMOOTH-004: empty sequence → 0.0."""
        assert flap_rate([]) == 0.0

    def test_alternating(self) -> None:
        """TC-SMOOTH-005: alternating labels → (n-1)/n."""
        result = flap_rate(["A", "B", "A", "B", "A"])
        assert result == pytest.approx(4 / 5)

    def test_two_runs(self) -> None:
        """TC-SMOOTH-006: two runs → 1/n."""
        result = flap_rate(["A", "A", "A", "B", "B"])
        assert result == pytest.approx(1 / 5)
