"""Tests for session boundary detection.

Covers:
- Single contiguous session (no gaps)
- Multiple sessions split by idle gap
- Custom idle_gap_seconds threshold
- Empty event list
- session_start_for_bucket binary search
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.features.sessions import detect_session_boundaries, session_start_for_bucket


def _ev(ts: dt.datetime, duration: float = 30.0) -> AWEvent:
    return AWEvent(
        timestamp=ts,
        duration_seconds=duration,
        app_id="org.mozilla.firefox",
        window_title_hash="abc123",
        is_browser=True,
        is_editor=False,
        is_terminal=False,
        app_category="browser",
    )


class TestDetectSessionBoundaries:
    def test_empty_events(self) -> None:
        assert detect_session_boundaries([]) == []

    def test_single_event(self) -> None:
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        starts = detect_session_boundaries([_ev(ts)])
        assert starts == [ts]

    def test_contiguous_events_one_session(self) -> None:
        """Events with gaps smaller than idle threshold stay in one session."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=30.0),
            _ev(dt.datetime(2026, 2, 23, 10, 1, 0), duration=30.0),
            _ev(dt.datetime(2026, 2, 23, 10, 2, 0), duration=30.0),
        ]
        starts = detect_session_boundaries(events)
        assert len(starts) == 1
        assert starts[0] == events[0].timestamp

    def test_large_gap_splits_session(self) -> None:
        """A 10-minute gap (> 5 min default) creates two sessions."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=30.0),
            _ev(dt.datetime(2026, 2, 23, 10, 1, 0), duration=30.0),
            # 10-minute gap after event 2 ends at 10:01:30
            _ev(dt.datetime(2026, 2, 23, 10, 12, 0), duration=30.0),
            _ev(dt.datetime(2026, 2, 23, 10, 13, 0), duration=30.0),
        ]
        starts = detect_session_boundaries(events)
        assert len(starts) == 2
        assert starts[0] == events[0].timestamp
        assert starts[1] == events[2].timestamp

    def test_gap_just_under_threshold_stays_one_session(self) -> None:
        """A gap of exactly 4 minutes (< 5 min default) does not split."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=60.0),
            # ends at 10:01:00; next starts at 10:05:00 → gap = 4 min
            _ev(dt.datetime(2026, 2, 23, 10, 5, 0), duration=60.0),
        ]
        starts = detect_session_boundaries(events)
        assert len(starts) == 1

    def test_gap_exactly_at_threshold_splits(self) -> None:
        """A gap equal to idle_gap_seconds starts a new session."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=0.0),
            # ends at 10:00:00; next starts at 10:05:00 → gap = 5 min = 300s
            _ev(dt.datetime(2026, 2, 23, 10, 5, 0), duration=0.0),
        ]
        starts = detect_session_boundaries(events, idle_gap_seconds=300.0)
        assert len(starts) == 2

    def test_custom_threshold(self) -> None:
        """A 2-minute gap splits when threshold is 60s."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=10.0),
            # ends at 10:00:10; next starts at 10:02:00 → gap = 110s > 60s
            _ev(dt.datetime(2026, 2, 23, 10, 2, 0), duration=10.0),
        ]
        starts = detect_session_boundaries(events, idle_gap_seconds=60.0)
        assert len(starts) == 2

    def test_multiple_sessions(self) -> None:
        """Three sessions separated by large gaps."""
        events = [
            _ev(dt.datetime(2026, 2, 23, 9, 0, 0), duration=10.0),
            _ev(dt.datetime(2026, 2, 23, 9, 0, 30), duration=10.0),
            # gap > 5 min
            _ev(dt.datetime(2026, 2, 23, 10, 0, 0), duration=10.0),
            # gap > 5 min
            _ev(dt.datetime(2026, 2, 23, 11, 0, 0), duration=10.0),
            _ev(dt.datetime(2026, 2, 23, 11, 0, 30), duration=10.0),
        ]
        starts = detect_session_boundaries(events)
        assert len(starts) == 3
        assert starts == [
            events[0].timestamp,
            events[2].timestamp,
            events[3].timestamp,
        ]


class TestSessionStartForBucket:
    def test_bucket_in_first_session(self) -> None:
        starts = [dt.datetime(2026, 2, 23, 9, 0, 0)]
        result = session_start_for_bucket(dt.datetime(2026, 2, 23, 9, 5, 0), starts)
        assert result == starts[0]

    def test_bucket_in_second_session(self) -> None:
        starts = [
            dt.datetime(2026, 2, 23, 9, 0, 0),
            dt.datetime(2026, 2, 23, 11, 0, 0),
        ]
        result = session_start_for_bucket(dt.datetime(2026, 2, 23, 11, 5, 0), starts)
        assert result == starts[1]

    def test_bucket_exactly_at_session_start(self) -> None:
        starts = [
            dt.datetime(2026, 2, 23, 9, 0, 0),
            dt.datetime(2026, 2, 23, 11, 0, 0),
        ]
        result = session_start_for_bucket(dt.datetime(2026, 2, 23, 11, 0, 0), starts)
        assert result == starts[1]

    def test_bucket_between_sessions(self) -> None:
        starts = [
            dt.datetime(2026, 2, 23, 9, 0, 0),
            dt.datetime(2026, 2, 23, 11, 0, 0),
        ]
        result = session_start_for_bucket(dt.datetime(2026, 2, 23, 10, 30, 0), starts)
        assert result == starts[0]
