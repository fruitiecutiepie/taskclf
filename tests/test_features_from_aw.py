"""Tests for building FeatureRows from normalized AW events.

Covers:
- Bucketing: events are grouped into 60s buckets
- Dominant app selection per bucket
- App switch count computation
- Temporal fields (hour_of_day, day_of_week, session_length_so_far)
- Keyboard/mouse fields are None (AW window watcher only)
- Schema metadata is correct
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.core.schema import FeatureSchemaV1
from taskclf.features.build import build_features_from_aw_events


def _make_event(
    ts: dt.datetime,
    duration: float = 30.0,
    app_id: str = "org.mozilla.firefox",
    title_hash: str = "abc123def456",
    is_browser: bool = True,
    is_editor: bool = False,
    is_terminal: bool = False,
) -> AWEvent:
    return AWEvent(
        timestamp=ts,
        duration_seconds=duration,
        app_id=app_id,
        window_title_hash=title_hash,
        is_browser=is_browser,
        is_editor=is_editor,
        is_terminal=is_terminal,
    )


class TestBuildFeaturesFromAWEvents:
    def test_empty_events(self) -> None:
        assert build_features_from_aw_events([]) == []

    def test_single_event_produces_one_row(self) -> None:
        ts = dt.datetime(2026, 2, 23, 10, 0, 30)
        events = [_make_event(ts)]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 1

    def test_bucket_start_aligned(self) -> None:
        ts = dt.datetime(2026, 2, 23, 10, 0, 45)
        events = [_make_event(ts)]
        rows = build_features_from_aw_events(events)
        assert rows[0].bucket_start_ts == dt.datetime(2026, 2, 23, 10, 0, 0)

    def test_multiple_events_same_bucket(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, duration=20.0, app_id="org.mozilla.firefox"),
            _make_event(base + dt.timedelta(seconds=20), duration=40.0, app_id="com.microsoft.VSCode",
                        is_browser=False, is_editor=True),
        ]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 1
        assert rows[0].app_id == "com.microsoft.VSCode"
        assert rows[0].is_editor is True

    def test_events_in_different_buckets(self) -> None:
        events = [
            _make_event(dt.datetime(2026, 2, 23, 10, 0, 0)),
            _make_event(dt.datetime(2026, 2, 23, 10, 1, 0)),
            _make_event(dt.datetime(2026, 2, 23, 10, 2, 0)),
        ]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 3
        assert rows[0].bucket_start_ts < rows[1].bucket_start_ts < rows[2].bucket_start_ts

    def test_schema_metadata(self) -> None:
        events = [_make_event(dt.datetime(2026, 2, 23, 10, 0, 0))]
        rows = build_features_from_aw_events(events)
        assert rows[0].schema_version == FeatureSchemaV1.VERSION
        assert rows[0].schema_hash == FeatureSchemaV1.SCHEMA_HASH
        assert rows[0].source_ids == ["aw-watcher-window"]

    def test_keyboard_mouse_fields_are_none(self) -> None:
        events = [_make_event(dt.datetime(2026, 2, 23, 10, 0, 0))]
        rows = build_features_from_aw_events(events)
        row = rows[0]
        assert row.keys_per_min is None
        assert row.backspace_ratio is None
        assert row.shortcut_rate is None
        assert row.clicks_per_min is None
        assert row.scroll_events_per_min is None
        assert row.mouse_distance is None

    def test_temporal_fields(self) -> None:
        ts = dt.datetime(2026, 2, 23, 14, 30, 0)  # Monday
        events = [_make_event(ts)]
        rows = build_features_from_aw_events(events)
        assert rows[0].hour_of_day == 14
        assert rows[0].day_of_week == ts.weekday()

    def test_session_length_increases(self) -> None:
        events = [
            _make_event(dt.datetime(2026, 2, 23, 10, 0, 0)),
            _make_event(dt.datetime(2026, 2, 23, 10, 5, 0)),
            _make_event(dt.datetime(2026, 2, 23, 10, 10, 0)),
        ]
        rows = build_features_from_aw_events(events)
        assert rows[0].session_length_so_far == 0.0
        assert rows[1].session_length_so_far == 5.0
        assert rows[2].session_length_so_far == 10.0

    def test_app_switch_count(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, app_id="app.one"),
            _make_event(base + dt.timedelta(minutes=1), app_id="app.two"),
            _make_event(base + dt.timedelta(minutes=2), app_id="app.three"),
            _make_event(base + dt.timedelta(minutes=3), app_id="app.one"),
        ]
        rows = build_features_from_aw_events(events)
        last_row = rows[-1]
        assert last_row.app_switch_count_last_5m >= 2

    def test_dominant_app_by_duration(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, duration=10.0, app_id="app.short"),
            _make_event(base + dt.timedelta(seconds=10), duration=50.0, app_id="app.long"),
        ]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 1
        assert rows[0].app_id == "app.long"
