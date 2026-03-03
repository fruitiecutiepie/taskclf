"""Tests for rolling-window feature helpers (features/windows.py).

Covers:
- app_switch_count_in_window with multiple apps
- Single-app and empty-event edge cases
- Start-of-day behavior
- Custom window_minutes and bucket_seconds parameters
- Events entirely before / after the window
- compute_rolling_app_switches across multiple buckets
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.features.windows import (
    app_switch_count_in_window,
    compute_rolling_app_switches,
)


def _ev(ts: dt.datetime, app: str) -> AWEvent:
    return AWEvent(
        timestamp=ts,
        duration_seconds=30.0,
        app_id=app,
        window_title_hash="aabbccddee00",
        is_browser=False,
        is_editor=False,
        is_terminal=False,
        app_category="other",
    )


class TestAppSwitchCountInWindow:
    def test_multiple_switches(self) -> None:
        """TC-FEAT-002: app switch counts in last 5 minutes match expected."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(minutes=1), "app.two"),
            _ev(base + dt.timedelta(minutes=2), "app.three"),
            _ev(base + dt.timedelta(minutes=3), "app.one"),
            _ev(base + dt.timedelta(minutes=4), "app.four"),
        ]

        count = app_switch_count_in_window(events, base + dt.timedelta(minutes=4))
        assert count == 3  # 4 unique apps - 1

        count_first = app_switch_count_in_window(events, base)
        assert count_first == 0  # only one app in [base-5m, base+60s)

    def test_start_of_day(self) -> None:
        """TC-FEAT-005: rolling-window features are consistent at start-of-day."""
        base = dt.datetime(2026, 2, 23, 9, 0, 0)

        single = [_ev(base, "app.one")]
        assert app_switch_count_in_window(single, base) == 0

        two_apps = [_ev(base, "app.one"), _ev(base + dt.timedelta(seconds=30), "app.two")]
        assert app_switch_count_in_window(two_apps, base) == 1

        assert app_switch_count_in_window([], base) == 0

    def test_events_entirely_before_window(self) -> None:
        """TC-FEAT-WIN-007: events entirely before the window returns 0."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base - dt.timedelta(minutes=20), "app.one"),
            _ev(base - dt.timedelta(minutes=15), "app.two"),
        ]
        assert app_switch_count_in_window(events, base) == 0

    def test_events_entirely_after_window(self) -> None:
        """TC-FEAT-WIN-008: events entirely after the window returns 0."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base + dt.timedelta(minutes=5), "app.one"),
            _ev(base + dt.timedelta(minutes=10), "app.two"),
        ]
        assert app_switch_count_in_window(events, base) == 0

    def test_custom_window_minutes(self) -> None:
        """TC-FEAT-WIN-extra: custom window_minutes captures more events."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base - dt.timedelta(minutes=10), "app.one"),
            _ev(base - dt.timedelta(minutes=3), "app.two"),
            _ev(base, "app.three"),
        ]
        assert app_switch_count_in_window(events, base, window_minutes=5) == 1
        assert app_switch_count_in_window(events, base, window_minutes=15) == 2

    def test_custom_bucket_seconds(self) -> None:
        """TC-FEAT-WIN-extra: custom bucket_seconds widens the window end."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(seconds=90), "app.two"),
        ]
        assert app_switch_count_in_window(events, base, bucket_seconds=60) == 0
        assert app_switch_count_in_window(events, base, bucket_seconds=300) == 1


class TestComputeRollingAppSwitches:
    def test_three_buckets_four_apps(self) -> None:
        """TC-FEAT-WIN-001: 3 buckets with 4 apps spread across them."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(minutes=1), "app.two"),
            _ev(base + dt.timedelta(minutes=2), "app.three"),
            _ev(base + dt.timedelta(minutes=3), "app.four"),
        ]
        buckets = [base + dt.timedelta(minutes=i) for i in range(3)]
        result = compute_rolling_app_switches(events, buckets)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_single_bucket(self) -> None:
        """TC-FEAT-WIN-002: single bucket returns list with one element."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_ev(base, "app.one"), _ev(base + dt.timedelta(seconds=30), "app.two")]
        result = compute_rolling_app_switches(events, [base])
        assert result == [1]

    def test_empty_buckets(self) -> None:
        """TC-FEAT-WIN-003: empty sorted_buckets returns empty list."""
        events = [_ev(dt.datetime(2026, 2, 23, 10, 0, 0), "app.one")]
        assert compute_rolling_app_switches(events, []) == []

    def test_empty_events(self) -> None:
        """TC-FEAT-WIN-004: empty events returns all-zero list."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        buckets = [base + dt.timedelta(minutes=i) for i in range(3)]
        assert compute_rolling_app_switches([], buckets) == [0, 0, 0]

    def test_custom_window_minutes(self) -> None:
        """TC-FEAT-WIN-005: wider window captures more switches."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base - dt.timedelta(minutes=10), "app.one"),
            _ev(base - dt.timedelta(minutes=3), "app.two"),
            _ev(base, "app.three"),
        ]
        narrow = compute_rolling_app_switches(events, [base], window_minutes=5)
        wide = compute_rolling_app_switches(events, [base], window_minutes=15)
        assert wide[0] > narrow[0]

    def test_custom_bucket_seconds(self) -> None:
        """TC-FEAT-WIN-006: custom bucket_seconds adjusts window end."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(seconds=200), "app.two"),
        ]
        short = compute_rolling_app_switches(events, [base], bucket_seconds=60)
        long = compute_rolling_app_switches(events, [base], bucket_seconds=300)
        assert short[0] == 0
        assert long[0] == 1
