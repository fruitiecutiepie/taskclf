"""Tests for building FeatureRows from normalized AW events.

Covers:
- Bucketing: events are grouped into 60s buckets
- Dominant app selection per bucket
- App switch count computation
- Temporal fields (hour_of_day, day_of_week, session_length_so_far)
- Keyboard/mouse fields are None (AW window watcher only)
- Keyboard/mouse fields populated when input_events provided
- Schema metadata is correct
"""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.types import AWEvent, AWInputEvent
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
    app_category: str = "browser",
) -> AWEvent:
    return AWEvent(
        timestamp=ts,
        duration_seconds=duration,
        app_id=app_id,
        window_title_hash=title_hash,
        is_browser=is_browser,
        is_editor=is_editor,
        is_terminal=is_terminal,
        app_category=app_category,
    )


def _make_input_event(
    ts: dt.datetime,
    duration: float = 5.0,
    presses: int = 10,
    clicks: int = 2,
    delta_x: int = 100,
    delta_y: int = 50,
    scroll_x: int = 0,
    scroll_y: int = 3,
) -> AWInputEvent:
    return AWInputEvent(
        timestamp=ts,
        duration_seconds=duration,
        presses=presses,
        clicks=clicks,
        delta_x=delta_x,
        delta_y=delta_y,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
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
                        is_browser=False, is_editor=True, app_category="editor"),
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

    def test_session_resets_after_idle_gap(self) -> None:
        """A gap > idle_gap_seconds resets session_length_so_far to 0."""
        events = [
            _make_event(dt.datetime(2026, 2, 23, 10, 0, 0), duration=10.0),
            _make_event(dt.datetime(2026, 2, 23, 10, 1, 0), duration=10.0),
            # 10-minute gap (>> 5 min default threshold)
            _make_event(dt.datetime(2026, 2, 23, 10, 12, 0), duration=10.0),
            _make_event(dt.datetime(2026, 2, 23, 10, 13, 0), duration=10.0),
        ]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 4
        assert rows[0].session_length_so_far == 0.0
        assert rows[1].session_length_so_far == 1.0
        assert rows[2].session_length_so_far == 0.0
        assert rows[3].session_length_so_far == 1.0

    def test_explicit_session_start_overrides_detection(self) -> None:
        """When session_start is provided, it is used for all buckets."""
        forced_start = dt.datetime(2026, 2, 23, 9, 50, 0)
        events = [
            _make_event(dt.datetime(2026, 2, 23, 10, 0, 0)),
            # large gap that would normally split sessions
            _make_event(dt.datetime(2026, 2, 23, 10, 20, 0)),
        ]
        rows = build_features_from_aw_events(events, session_start=forced_start)
        assert rows[0].session_length_so_far == 10.0
        assert rows[1].session_length_so_far == 30.0

    def test_dominant_app_by_duration(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, duration=10.0, app_id="app.short"),
            _make_event(base + dt.timedelta(seconds=10), duration=50.0, app_id="app.long"),
        ]
        rows = build_features_from_aw_events(events)
        assert len(rows) == 1
        assert rows[0].app_id == "app.long"


class TestBuildFeaturesWithInputEvents:
    """Verify keyboard/mouse features are populated from AWInputEvent data."""

    def test_input_events_populate_features(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        input_events = [
            _make_input_event(base, presses=30, clicks=6, delta_x=200, delta_y=100,
                              scroll_x=2, scroll_y=4),
            _make_input_event(base + dt.timedelta(seconds=5), presses=30, clicks=6,
                              delta_x=100, delta_y=50, scroll_x=0, scroll_y=2),
        ]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert len(rows) == 1
        row = rows[0]
        # 60 presses / 1 min = 60.0
        assert row.keys_per_min == 60.0
        # 12 clicks / 1 min = 12.0
        assert row.clicks_per_min == 12.0
        # (2+4+0+2) = 8 scroll / 1 min = 8.0
        assert row.scroll_events_per_min == 8.0
        # (200+100+100+50) = 450 px
        assert row.mouse_distance == 450.0

    def test_no_input_events_leaves_none(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        rows = build_features_from_aw_events(window_events, input_events=None)
        row = rows[0]
        assert row.keys_per_min is None
        assert row.clicks_per_min is None
        assert row.scroll_events_per_min is None
        assert row.mouse_distance is None

    def test_backspace_ratio_and_shortcut_rate_remain_none(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        input_events = [_make_input_event(base, presses=50)]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert rows[0].backspace_ratio is None
        assert rows[0].shortcut_rate is None

    def test_source_ids_include_input_watcher(self) -> None:
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        input_events = [_make_input_event(base)]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert "aw-watcher-input" in rows[0].source_ids
        assert "aw-watcher-window" in rows[0].source_ids

    def test_partial_coverage_only_fills_matched_buckets(self) -> None:
        """Input events in only one bucket; the other bucket stays None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [
            _make_event(base),
            _make_event(base + dt.timedelta(minutes=1)),
        ]
        input_events = [_make_input_event(base, presses=20, clicks=4)]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert len(rows) == 2
        assert rows[0].keys_per_min == 20.0
        assert rows[0].clicks_per_min == 4.0
        assert rows[1].keys_per_min is None
        assert rows[1].clicks_per_min is None

    def test_zero_input_produces_zero_features(self) -> None:
        """All-zero input events should produce 0.0, not None."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        input_events = [
            _make_input_event(base, presses=0, clicks=0, delta_x=0, delta_y=0,
                              scroll_x=0, scroll_y=0),
        ]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        row = rows[0]
        assert row.keys_per_min == 0.0
        assert row.clicks_per_min == 0.0
        assert row.scroll_events_per_min == 0.0
        assert row.mouse_distance == 0.0


class TestNewFeatureUpgrades:
    """Tests for TODO 12 feature upgrades (items 37-40)."""

    def test_domain_category_non_browser(self) -> None:
        """Non-browser apps get domain_category='non_browser'."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts, is_browser=False, app_category="editor")]
        rows = build_features_from_aw_events(events)
        assert rows[0].domain_category == "non_browser"

    def test_domain_category_browser_unknown(self) -> None:
        """Browser apps without URL info get domain_category='unknown'."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts, is_browser=True, app_category="browser")]
        rows = build_features_from_aw_events(events)
        assert rows[0].domain_category == "unknown"

    def test_window_title_bucket_range(self) -> None:
        """window_title_bucket is in [0, 255]."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts)]
        rows = build_features_from_aw_events(events)
        assert 0 <= rows[0].window_title_bucket <= 255

    def test_window_title_bucket_deterministic(self) -> None:
        """Same title hash -> same bucket."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts, title_hash="abc123def456")]
        rows1 = build_features_from_aw_events(events)
        rows2 = build_features_from_aw_events(events)
        assert rows1[0].window_title_bucket == rows2[0].window_title_bucket

    def test_title_repeat_count_increments(self) -> None:
        """title_repeat_count_session increments for same title hash in session."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, title_hash="same_hash"),
            _make_event(base + dt.timedelta(minutes=1), title_hash="same_hash"),
            _make_event(base + dt.timedelta(minutes=2), title_hash="same_hash"),
        ]
        rows = build_features_from_aw_events(events)
        assert rows[0].title_repeat_count_session == 1
        assert rows[1].title_repeat_count_session == 2
        assert rows[2].title_repeat_count_session == 3

    def test_title_repeat_count_different_titles(self) -> None:
        """Different title hashes each start at 1."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, title_hash="hash_a"),
            _make_event(base + dt.timedelta(minutes=1), title_hash="hash_b"),
            _make_event(base + dt.timedelta(minutes=2), title_hash="hash_a"),
        ]
        rows = build_features_from_aw_events(events)
        assert rows[0].title_repeat_count_session == 1
        assert rows[1].title_repeat_count_session == 1
        assert rows[2].title_repeat_count_session == 2

    def test_rolling_means_none_without_input(self) -> None:
        """Rolling means are None when no input events provided."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts)]
        rows = build_features_from_aw_events(events)
        assert rows[0].keys_per_min_rolling_5 is None
        assert rows[0].keys_per_min_rolling_15 is None
        assert rows[0].mouse_distance_rolling_5 is None
        assert rows[0].mouse_distance_rolling_15 is None

    def test_rolling_means_populated_with_input(self) -> None:
        """Rolling means computed when input events are provided."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [_make_event(base)]
        input_events = [_make_input_event(base, presses=60, delta_x=200, delta_y=100)]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert rows[0].keys_per_min_rolling_5 == rows[0].keys_per_min
        assert rows[0].mouse_distance_rolling_5 == rows[0].mouse_distance

    def test_deltas_none_for_first_bucket(self) -> None:
        """Deltas are None for the very first bucket."""
        ts = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [_make_event(ts)]
        input_events = [_make_input_event(ts)]
        rows = build_features_from_aw_events(events, input_events=input_events)
        assert rows[0].keys_per_min_delta is None
        assert rows[0].clicks_per_min_delta is None
        assert rows[0].mouse_distance_delta is None

    def test_deltas_computed_for_second_bucket(self) -> None:
        """Deltas are the difference from the previous bucket."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [
            _make_event(base),
            _make_event(base + dt.timedelta(minutes=1)),
        ]
        input_events = [
            _make_input_event(base, presses=30, clicks=2, delta_x=100, delta_y=50),
            _make_input_event(base + dt.timedelta(minutes=1), presses=60, clicks=4,
                              delta_x=200, delta_y=100),
        ]
        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert rows[1].keys_per_min_delta is not None
        assert rows[1].keys_per_min_delta == rows[1].keys_per_min - rows[0].keys_per_min

    def test_app_switch_count_last_15m(self) -> None:
        """app_switch_count_last_15m counts unique apps over 15 minutes."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        events = [
            _make_event(base, app_id="app.one"),
            _make_event(base + dt.timedelta(minutes=1), app_id="app.two"),
            _make_event(base + dt.timedelta(minutes=6), app_id="app.three"),
            _make_event(base + dt.timedelta(minutes=10), app_id="app.four"),
        ]
        rows = build_features_from_aw_events(events)
        last = rows[-1]
        assert last.app_switch_count_last_15m >= 3
        assert last.app_switch_count_last_15m >= last.app_switch_count_last_5m


class TestIdleSegmentFeatures:
    def test_idle_segments_produce_zero_active(self) -> None:
        """TC-FEAT-003: idle segments produce active_seconds=0 and correct idle flags."""
        base = dt.datetime(2026, 2, 23, 10, 0, 0)
        window_events = [
            _make_event(base, duration=60.0),
        ]
        input_events = [
            _make_input_event(base, presses=0, clicks=0, delta_x=0, delta_y=0,
                              scroll_x=0, scroll_y=0),
            _make_input_event(base + dt.timedelta(seconds=5), presses=0, clicks=0,
                              delta_x=0, delta_y=0, scroll_x=0, scroll_y=0),
        ]

        rows = build_features_from_aw_events(window_events, input_events=input_events)
        assert len(rows) == 1
        row = rows[0]

        assert row.active_seconds_any == 0.0
        assert row.active_seconds_keyboard == 0.0
        assert row.active_seconds_mouse == 0.0
        assert row.max_idle_run_seconds == 10.0
        assert row.keys_per_min == 0.0
        assert row.clicks_per_min == 0.0
