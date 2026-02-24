"""Tests for rolling-window feature helpers (features/windows.py).

Covers:
- app_switch_count_in_window with multiple apps
- Single-app and empty-event edge cases
- Start-of-day behavior
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.features.windows import app_switch_count_in_window


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
