"""Feature computation pipeline: build bucketed feature rows and write to parquet."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from taskclf.adapters.activitywatch.types import AWInputEvent
from taskclf.core.defaults import (
    DEFAULT_APP_SWITCH_WINDOW_15M,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_DUMMY_ROWS,
    DEFAULT_IDLE_GAP_SECONDS,
    DEFAULT_ROLLING_WINDOW_5,
    DEFAULT_ROLLING_WINDOW_15,
    DEFAULT_TITLE_HASH_BUCKETS,
)
from taskclf.core.hashing import stable_hash
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import write_parquet
from taskclf.core.time import align_to_bucket
from taskclf.core.types import Event, FeatureRow
from taskclf.features.domain import classify_domain
from taskclf.features.dynamics import DynamicsTracker
from taskclf.features.sessions import detect_session_boundaries, session_start_for_bucket
from taskclf.features.text import title_hash_bucket
from taskclf.features.windows import app_switch_count_in_window

_DUMMY_APPS: list[tuple[str, bool, bool, bool, str]] = [
    # (app_id, is_browser, is_editor, is_terminal, app_category)
    ("com.apple.Terminal", False, False, True, "terminal"),
    ("org.mozilla.firefox", True, False, False, "browser"),
    ("com.microsoft.VSCode", False, True, False, "editor"),
    ("com.apple.mail", False, False, False, "email"),
    ("us.zoom.xos", False, False, False, "meeting"),
    ("com.tinyspeck.slackmacgap", False, False, False, "chat"),
    ("com.google.Chrome", True, False, False, "browser"),
    ("com.jetbrains.intellij", False, True, False, "editor"),
    ("com.apple.finder", False, False, False, "file_manager"),
    ("com.apple.Notes", False, False, False, "docs"),
]


def generate_dummy_features(
    date: dt.date,
    n_rows: int = DEFAULT_DUMMY_ROWS,
    *,
    user_id: str = "dummy-user-001",
    device_id: str | None = None,
) -> list[FeatureRow]:
    """Create *n_rows* synthetic FeatureRow instances spanning *date*.

    Args:
        date: The calendar date to generate buckets for (hours 9-17).
        n_rows: Number of rows to generate.
        user_id: User identifier for all generated rows.
        device_id: Optional device identifier.

    Returns:
        Validated ``FeatureRow`` instances with dummy app/keyboard/mouse data.
    """
    rows: list[FeatureRow] = []
    day_of_week = date.weekday()
    session_start = dt.datetime(date.year, date.month, date.day, 9, 0)
    sid = stable_hash(f"{user_id}:{session_start.isoformat()}")

    tracker = DynamicsTracker()
    title_counts: dict[str, int] = defaultdict(int)

    for i in range(n_rows):
        hour = 9 + (i * 8 // max(n_rows, 1))
        minute = (i * 7) % 60
        ts = dt.datetime(date.year, date.month, date.day, hour, minute)
        end_ts = ts + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)

        app_id, is_browser, is_editor, is_terminal, app_category = _DUMMY_APPS[
            i % len(_DUMMY_APPS)
        ]
        title_hash = stable_hash(f"window-title-{app_id}-{i}")
        title_counts[title_hash] += 1

        keys = float(40 + i * 10)
        clicks = float(3 + i % 8)
        mouse_dist = float(200 + i * 50)
        dynamics = tracker.update(keys, clicks, mouse_dist)

        rows.append(
            FeatureRow(
                user_id=user_id,
                device_id=device_id,
                session_id=sid,
                bucket_start_ts=ts,
                bucket_end_ts=end_ts,
                schema_version=FeatureSchemaV1.VERSION,
                schema_hash=FeatureSchemaV1.SCHEMA_HASH,
                source_ids=[f"dummy-{i:03d}"],
                app_id=app_id,
                app_category=app_category,
                window_title_hash=title_hash,
                is_browser=is_browser,
                is_editor=is_editor,
                is_terminal=is_terminal,
                app_switch_count_last_5m=i % 5,
                app_foreground_time_ratio=round(0.5 + (i % 5) * 0.1, 2),
                app_change_count=i % 4,
                keys_per_min=keys,
                backspace_ratio=round(0.05 + (i % 5) * 0.02, 2),
                shortcut_rate=round(0.1 + (i % 3) * 0.05, 2),
                clicks_per_min=clicks,
                scroll_events_per_min=float(i % 6),
                mouse_distance=mouse_dist,
                active_seconds_keyboard=float(20 + (i % 8) * 5),
                active_seconds_mouse=float(15 + (i % 9) * 5),
                active_seconds_any=float(30 + (i % 6) * 5),
                max_idle_run_seconds=float(5 + (i % 4) * 5),
                event_density=round(1.5 + (i % 5) * 0.3, 2),
                domain_category=classify_domain(None, is_browser=is_browser),
                window_title_bucket=title_hash_bucket(title_hash, DEFAULT_TITLE_HASH_BUCKETS),
                title_repeat_count_session=title_counts[title_hash],
                keys_per_min_rolling_5=dynamics["keys_per_min_rolling_5"],
                keys_per_min_rolling_15=dynamics["keys_per_min_rolling_15"],
                mouse_distance_rolling_5=dynamics["mouse_distance_rolling_5"],
                mouse_distance_rolling_15=dynamics["mouse_distance_rolling_15"],
                keys_per_min_delta=dynamics["keys_per_min_delta"],
                clicks_per_min_delta=dynamics["clicks_per_min_delta"],
                mouse_distance_delta=dynamics["mouse_distance_delta"],
                app_switch_count_last_15m=i % 8,
                hour_of_day=hour,
                day_of_week=day_of_week,
                session_length_so_far=float(i * 5),
            )
        )

    return rows


_INPUT_NULL_FIELDS: dict[str, None] = {
    "keys_per_min": None,
    "clicks_per_min": None,
    "scroll_events_per_min": None,
    "mouse_distance": None,
    "active_seconds_keyboard": None,
    "active_seconds_mouse": None,
    "active_seconds_any": None,
    "max_idle_run_seconds": None,
    "event_density": None,
}


def _aggregate_input_for_bucket(
    bucket_ts: dt.datetime,
    bucket_input: list[AWInputEvent],
    bucket_seconds: int,
) -> dict[str, float | None]:
    """Aggregate input events within a single time bucket into feature values.

    Computes per-minute rates, total mouse distance, and activity
    occupancy metrics.  If *bucket_input* is empty every value is
    ``None``.
    """
    if not bucket_input:
        return dict(_INPUT_NULL_FIELDS)

    total_presses = sum(e.presses for e in bucket_input)
    total_clicks = sum(e.clicks for e in bucket_input)
    total_scroll = sum(e.scroll_x + e.scroll_y for e in bucket_input)
    total_distance = sum(e.delta_x + e.delta_y for e in bucket_input)

    minutes = bucket_seconds / 60.0

    # Activity occupancy
    kb_seconds = 0.0
    mouse_seconds = 0.0
    any_seconds = 0.0
    active_event_count = 0

    sorted_input = sorted(bucket_input, key=lambda e: e.timestamp)
    idle_run = 0.0
    max_idle = 0.0

    for ev in sorted_input:
        has_kb = ev.presses > 0
        has_mouse = (ev.clicks > 0 or ev.delta_x + ev.delta_y > 0
                     or ev.scroll_x + ev.scroll_y > 0)
        has_any = has_kb or has_mouse

        if has_kb:
            kb_seconds += ev.duration_seconds
        if has_mouse:
            mouse_seconds += ev.duration_seconds
        if has_any:
            any_seconds += ev.duration_seconds
            active_event_count += 1
            if idle_run > max_idle:
                max_idle = idle_run
            idle_run = 0.0
        else:
            idle_run += ev.duration_seconds

    if idle_run > max_idle:
        max_idle = idle_run

    density = (active_event_count / any_seconds) if any_seconds > 0 else 0.0

    return {
        "keys_per_min": round(total_presses / minutes, 2),
        "clicks_per_min": round(total_clicks / minutes, 2),
        "scroll_events_per_min": round(total_scroll / minutes, 2),
        "mouse_distance": float(total_distance),
        "active_seconds_keyboard": round(kb_seconds, 2),
        "active_seconds_mouse": round(mouse_seconds, 2),
        "active_seconds_any": round(any_seconds, 2),
        "max_idle_run_seconds": round(max_idle, 2),
        "event_density": round(density, 4),
    }


def build_features_from_aw_events(
    events: Sequence[Event],
    *,
    user_id: str = "default-user",
    device_id: str | None = None,
    input_events: Sequence[AWInputEvent] | None = None,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    session_start: dt.datetime | None = None,
    idle_gap_seconds: float = DEFAULT_IDLE_GAP_SECONDS,
) -> list[FeatureRow]:
    """Convert normalised events into per-bucket :class:`FeatureRow` instances.

    Events are grouped into fixed-width time buckets.  For each bucket
    the *dominant* application (longest total duration) is selected and
    its metadata (app ID, title hash, flags) is used to populate the
    context columns.

    When *input_events* from ``aw-watcher-input`` are provided, keyboard
    and mouse features (``keys_per_min``, ``clicks_per_min``,
    ``scroll_events_per_min``, ``mouse_distance``) are computed by
    aggregating the 5-second input samples that fall within each bucket.
    Without input events those fields remain ``None``.

    Session detection is performed automatically via idle-gap analysis
    (see :func:`~taskclf.features.sessions.detect_session_boundaries`).
    In online mode the caller may pass a known *session_start* to
    avoid resetting the session each poll cycle.

    Args:
        events: Sorted, normalised events satisfying the
            :class:`~taskclf.core.types.Event` protocol (e.g. from
            :func:`~taskclf.adapters.activitywatch.client.parse_aw_export`).
        user_id: Random UUID identifying the user (not PII).
        device_id: Optional device identifier.
        input_events: Optional sorted input events from
            ``aw-watcher-input``.  When provided, keyboard/mouse feature
            columns are populated; otherwise they remain ``None``.
        bucket_seconds: Width of each time bucket in seconds (default 60).
        session_start: If provided, used as the session start for every
            bucket (online mode).  When ``None`` (batch mode), sessions
            are detected from idle gaps in *events*.
        idle_gap_seconds: Minimum gap in seconds that splits sessions
            (only used when *session_start* is ``None``).

    Returns:
        Validated ``FeatureRow`` instances ordered by ``bucket_start_ts``.
    """
    if not events:
        return []

    bucket_events: dict[dt.datetime, list[Event]] = defaultdict(list)
    for ev in events:
        bucket_ts = align_to_bucket(ev.timestamp, bucket_seconds)
        bucket_events[bucket_ts].append(ev)

    bucket_input_events: dict[dt.datetime, list[AWInputEvent]] = defaultdict(list)
    if input_events:
        for ie in input_events:
            ie_bucket = align_to_bucket(ie.timestamp, bucket_seconds)
            bucket_input_events[ie_bucket].append(ie)

    has_input = bool(input_events)

    sorted_buckets = sorted(bucket_events.keys())
    all_events_sorted = sorted(events, key=lambda e: e.timestamp)

    if session_start is not None:
        session_starts: list[dt.datetime] = [
            align_to_bucket(session_start, bucket_seconds),
        ]
    else:
        session_starts = [
            align_to_bucket(ts, bucket_seconds)
            for ts in detect_session_boundaries(
                all_events_sorted, idle_gap_seconds=idle_gap_seconds,
            )
        ]

    # Pre-compute session_id for each session start
    session_id_map: dict[dt.datetime, str] = {
        ss: stable_hash(f"{user_id}:{ss.isoformat()}")
        for ss in session_starts
    }

    dynamics = DynamicsTracker(
        rolling_5=DEFAULT_ROLLING_WINDOW_5,
        rolling_15=DEFAULT_ROLLING_WINDOW_15,
    )
    title_counts: dict[str, int] = defaultdict(int)

    rows: list[FeatureRow] = []
    for bucket_ts in sorted_buckets:
        evs = bucket_events[bucket_ts]

        app_durations: dict[str, float] = defaultdict(float)
        for ev in evs:
            app_durations[ev.app_id] += ev.duration_seconds
        dominant_app_id = max(app_durations, key=app_durations.get)  # type: ignore[arg-type]

        dominant_ev = next(ev for ev in evs if ev.app_id == dominant_app_id)

        foreground_ratio = min(app_durations[dominant_app_id] / bucket_seconds, 1.0)

        sorted_evs = sorted(evs, key=lambda e: e.timestamp)
        change_count = sum(
            1 for a, b in zip(sorted_evs, sorted_evs[1:])
            if a.app_id != b.app_id
        )

        switch_count = app_switch_count_in_window(
            all_events_sorted, bucket_ts, bucket_seconds=bucket_seconds,
        )
        switch_count_15m = app_switch_count_in_window(
            all_events_sorted, bucket_ts,
            window_minutes=DEFAULT_APP_SWITCH_WINDOW_15M,
            bucket_seconds=bucket_seconds,
        )

        cur_session = session_start_for_bucket(bucket_ts, session_starts)
        elapsed_minutes = (bucket_ts - cur_session).total_seconds() / 60.0
        sid = session_id_map[cur_session]

        input_agg = _aggregate_input_for_bucket(
            bucket_ts, bucket_input_events.get(bucket_ts, []), bucket_seconds,
        )

        # Title clustering (item 39)
        title_hash = dominant_ev.window_title_hash
        title_counts[title_hash] += 1
        w_title_bucket = title_hash_bucket(title_hash, DEFAULT_TITLE_HASH_BUCKETS)

        # Domain classification (item 38)
        domain_cat = classify_domain(None, is_browser=dominant_ev.is_browser)

        # Temporal dynamics (item 40)
        dyn = dynamics.update(
            input_agg["keys_per_min"],
            input_agg["clicks_per_min"],
            input_agg["mouse_distance"],
        )

        source_ids = ["aw-watcher-window"]
        if has_input:
            source_ids.append("aw-watcher-input")

        rows.append(
            FeatureRow(
                user_id=user_id,
                device_id=device_id,
                session_id=sid,
                bucket_start_ts=bucket_ts,
                bucket_end_ts=bucket_ts + dt.timedelta(seconds=bucket_seconds),
                schema_version=FeatureSchemaV1.VERSION,
                schema_hash=FeatureSchemaV1.SCHEMA_HASH,
                source_ids=source_ids,
                app_id=dominant_app_id,
                app_category=dominant_ev.app_category,
                window_title_hash=title_hash,
                is_browser=dominant_ev.is_browser,
                is_editor=dominant_ev.is_editor,
                is_terminal=dominant_ev.is_terminal,
                app_switch_count_last_5m=switch_count,
                app_foreground_time_ratio=round(foreground_ratio, 4),
                app_change_count=change_count,
                keys_per_min=input_agg["keys_per_min"],
                backspace_ratio=None,
                shortcut_rate=None,
                clicks_per_min=input_agg["clicks_per_min"],
                scroll_events_per_min=input_agg["scroll_events_per_min"],
                mouse_distance=input_agg["mouse_distance"],
                active_seconds_keyboard=input_agg["active_seconds_keyboard"],
                active_seconds_mouse=input_agg["active_seconds_mouse"],
                active_seconds_any=input_agg["active_seconds_any"],
                max_idle_run_seconds=input_agg["max_idle_run_seconds"],
                event_density=input_agg["event_density"],
                domain_category=domain_cat,
                window_title_bucket=w_title_bucket,
                title_repeat_count_session=title_counts[title_hash],
                keys_per_min_rolling_5=dyn["keys_per_min_rolling_5"],
                keys_per_min_rolling_15=dyn["keys_per_min_rolling_15"],
                mouse_distance_rolling_5=dyn["mouse_distance_rolling_5"],
                mouse_distance_rolling_15=dyn["mouse_distance_rolling_15"],
                keys_per_min_delta=dyn["keys_per_min_delta"],
                clicks_per_min_delta=dyn["clicks_per_min_delta"],
                mouse_distance_delta=dyn["mouse_distance_delta"],
                app_switch_count_last_15m=switch_count_15m,
                hour_of_day=bucket_ts.hour,
                day_of_week=bucket_ts.weekday(),
                session_length_so_far=round(elapsed_minutes, 2),
            )
        )

    return rows


def build_features_for_date(date: dt.date, data_dir: Path) -> Path:
    """Generate dummy features for *date*, validate, and write to parquet.

    Args:
        date: Calendar date to build features for.
        data_dir: Root of processed data (e.g. ``Path("data/processed")``).
            Output lands at ``data_dir/features_v1/date=YYYY-MM-DD/features.parquet``.

    Returns:
        Path of the written parquet file.

    Raises:
        ValueError: If generated data fails ``FeatureSchemaV1`` validation.
    """
    rows = generate_dummy_features(date)
    df = pd.DataFrame([r.model_dump() for r in rows])

    FeatureSchemaV1.validate_dataframe(df)

    out_path = (
        data_dir / f"features_v1/date={date.isoformat()}" / "features.parquet"
    )
    return write_parquet(df, out_path)
