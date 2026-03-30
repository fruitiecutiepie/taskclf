"""Tests for rolling-window feature helpers (features/windows.py).

Covers:
- app_switch_count_in_window with multiple apps
- Single-app and empty-event edge cases
- Start-of-day behavior
- Custom window_minutes and bucket_seconds parameters
- Events entirely before / after the window
- compute_rolling_app_switches across multiple buckets
- TC-FEAT-WIN-UTC-*: aware-UTC timestamp handling
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.features.windows import (
    app_entropy_in_window,
    app_switch_count_in_window,
    compute_rolling_app_switches,
    top2_app_concentration_in_window,
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

        two_apps = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(seconds=30), "app.two"),
        ]
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


# ---------------------------------------------------------------------------
# Aware-UTC timestamp tests (Phase 4 migration)
# ---------------------------------------------------------------------------

_UTC = dt.timezone.utc


class TestAppSwitchCountAwareUtc:
    """TC-FEAT-WIN-UTC-*: verify window logic works with aware-UTC timestamps."""

    def test_aware_utc_events_and_bucket(self) -> None:
        """TC-FEAT-WIN-UTC-001: all aware-UTC timestamps work identically to naive."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0, tzinfo=_UTC)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(minutes=1), "app.two"),
            _ev(base + dt.timedelta(minutes=2), "app.three"),
        ]
        count = app_switch_count_in_window(events, base + dt.timedelta(minutes=2))
        assert count == 2

    def test_aware_matches_naive_result(self) -> None:
        """TC-FEAT-WIN-UTC-002: aware and naive inputs produce identical counts."""
        naive_base = dt.datetime(2026, 2, 23, 10, 0, 0)
        aware_base = dt.datetime(2026, 2, 23, 10, 0, 0, tzinfo=_UTC)

        naive_events = [
            _ev(naive_base, "app.one"),
            _ev(naive_base + dt.timedelta(minutes=1), "app.two"),
            _ev(naive_base + dt.timedelta(minutes=3), "app.three"),
        ]
        aware_events = [
            _ev(aware_base, "app.one"),
            _ev(aware_base + dt.timedelta(minutes=1), "app.two"),
            _ev(aware_base + dt.timedelta(minutes=3), "app.three"),
        ]

        naive_count = app_switch_count_in_window(
            naive_events, naive_base + dt.timedelta(minutes=3)
        )
        aware_count = app_switch_count_in_window(
            aware_events, aware_base + dt.timedelta(minutes=3)
        )
        assert naive_count == aware_count

    def test_mixed_naive_events_aware_bucket(self) -> None:
        """TC-FEAT-WIN-UTC-003: naive events with aware bucket still count correctly."""
        naive_base = dt.datetime(2026, 2, 23, 10, 0, 0)
        aware_bucket = dt.datetime(2026, 2, 23, 10, 2, 0, tzinfo=_UTC)

        events = [
            _ev(naive_base, "app.one"),
            _ev(naive_base + dt.timedelta(minutes=1), "app.two"),
            _ev(naive_base + dt.timedelta(minutes=2), "app.three"),
        ]
        count = app_switch_count_in_window(events, aware_bucket)
        assert count == 2

    def test_non_utc_offset_events_normalized(self) -> None:
        """TC-FEAT-WIN-UTC-004: events with non-UTC offset are normalized correctly."""
        utc_plus_5 = dt.timezone(dt.timedelta(hours=5))
        base_local = dt.datetime(2026, 2, 23, 15, 0, 0, tzinfo=utc_plus_5)
        base_utc = dt.datetime(2026, 2, 23, 10, 0, 0, tzinfo=_UTC)

        events = [
            _ev(base_local, "app.one"),
            _ev(base_local + dt.timedelta(seconds=30), "app.two"),
        ]
        count = app_switch_count_in_window(events, base_utc)
        assert count == 1

    def test_rolling_with_aware_utc(self) -> None:
        """TC-FEAT-WIN-UTC-005: compute_rolling_app_switches works with aware timestamps."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0, tzinfo=_UTC)
        events = [
            _ev(base, "app.one"),
            _ev(base + dt.timedelta(seconds=30), "app.two"),
        ]
        result = compute_rolling_app_switches(events, [base])
        assert result == [1]


# ---------------------------------------------------------------------------
# P6-001: app_entropy_5m / app_entropy_15m (Shannon entropy)
# ---------------------------------------------------------------------------


def _ev_dur(ts: dt.datetime, app: str, duration: float = 30.0) -> AWEvent:
    """Helper creating an event with a specific duration."""
    return AWEvent(
        timestamp=ts,
        duration_seconds=duration,
        app_id=app,
        window_title_hash="aabbccddee00",
        is_browser=False,
        is_editor=False,
        is_terminal=False,
        app_category="other",
    )


class TestAppEntropyInWindow:
    """P6-001: Shannon entropy of app distribution in a look-back window."""

    def test_single_app_zero_entropy(self) -> None:
        """One app in the window yields entropy 0.0."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.one", 30.0),
        ]
        result = app_entropy_in_window(events, base)
        assert result == 0.0

    def test_two_equal_apps_one_bit(self) -> None:
        """Two apps with equal duration yields entropy 1.0 bit."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.two", 30.0),
        ]
        result = app_entropy_in_window(events, base)
        assert result == 1.0

    def test_three_unequal_apps(self) -> None:
        """Three apps with unequal durations produce expected entropy."""
        import math

        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.two", 20.0),
            _ev_dur(base + dt.timedelta(seconds=50), "app.three", 10.0),
        ]
        result = app_entropy_in_window(events, base)
        assert result is not None

        total = 60.0
        p1, p2, p3 = 30 / total, 20 / total, 10 / total
        expected = -(p1 * math.log2(p1) + p2 * math.log2(p2) + p3 * math.log2(p3))
        assert result == round(expected, 4)

    def test_no_events_returns_none(self) -> None:
        """Empty event list yields None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        result = app_entropy_in_window([], base)
        assert result is None

    def test_events_outside_window_returns_none(self) -> None:
        """Events entirely outside the window yield None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base - dt.timedelta(minutes=20), "app.one", 30.0),
        ]
        result = app_entropy_in_window(events, base)
        assert result is None

    def test_custom_window_minutes(self) -> None:
        """Wider window captures more events, potentially changing entropy."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base - dt.timedelta(minutes=10), "app.one", 30.0),
            _ev_dur(base, "app.two", 30.0),
        ]
        narrow = app_entropy_in_window(events, base, window_minutes=5)
        wide = app_entropy_in_window(events, base, window_minutes=15)

        assert narrow is not None
        assert narrow == 0.0
        assert wide is not None
        assert wide == 1.0


# ---------------------------------------------------------------------------
# P6-001: top2_app_concentration_15m helper
# ---------------------------------------------------------------------------


class TestTop2AppConcentrationInWindow:
    """P6-001: Helper for the `top2_app_concentration_15m` feature."""

    def test_single_app_returns_one(self) -> None:
        """One app in the window yields concentration 1.0."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.one", 30.0),
        ]
        result = top2_app_concentration_in_window(events, base)
        assert result == 1.0

    def test_two_equal_apps_returns_one(self) -> None:
        """Two apps with equal duration yields concentration 1.0."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.two", 30.0),
        ]
        result = top2_app_concentration_in_window(events, base)
        assert result == 1.0

    def test_three_equal_apps(self) -> None:
        """Three equal apps: top 2 get 2/3 of total time."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 20.0),
            _ev_dur(base + dt.timedelta(seconds=20), "app.two", 20.0),
            _ev_dur(base + dt.timedelta(seconds=40), "app.three", 20.0),
        ]
        result = top2_app_concentration_in_window(events, base)
        assert result is not None
        assert result == round(40.0 / 60.0, 4)

    def test_three_unequal_apps(self) -> None:
        """Three unequal apps: top 2 share is (30+20)/60."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base, "app.one", 30.0),
            _ev_dur(base + dt.timedelta(seconds=30), "app.two", 20.0),
            _ev_dur(base + dt.timedelta(seconds=50), "app.three", 10.0),
        ]
        result = top2_app_concentration_in_window(events, base)
        assert result is not None
        assert result == round(50.0 / 60.0, 4)

    def test_no_events_returns_none(self) -> None:
        """Empty event list yields None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        result = top2_app_concentration_in_window([], base)
        assert result is None

    def test_events_outside_window_returns_none(self) -> None:
        """Events entirely outside the window yield None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base - dt.timedelta(minutes=20), "app.one", 30.0),
        ]
        result = top2_app_concentration_in_window(events, base)
        assert result is None

    def test_custom_window_minutes(self) -> None:
        """Wider window captures more events, changing concentration."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _ev_dur(base - dt.timedelta(minutes=10), "app.one", 30.0),
            _ev_dur(base, "app.two", 30.0),
        ]
        narrow = top2_app_concentration_in_window(events, base, window_minutes=5)
        wide = top2_app_concentration_in_window(events, base, window_minutes=15)

        assert narrow is not None
        assert narrow == 1.0
        assert wide is not None
        assert wide == 1.0
