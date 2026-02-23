"""Feature computation pipeline: build bucketed feature rows and write to parquet."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from taskclf.core.defaults import (
    DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_DUMMY_ROWS,
    DEFAULT_IDLE_GAP_SECONDS,
)
from taskclf.core.hashing import stable_hash
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import write_parquet
from taskclf.core.time import align_to_bucket
from taskclf.core.types import Event, FeatureRow
from taskclf.features.sessions import detect_session_boundaries, session_start_for_bucket

_DUMMY_APPS: list[tuple[str, bool, bool, bool]] = [
    # (app_id, is_browser, is_editor, is_terminal)
    ("com.apple.Terminal", False, False, True),
    ("org.mozilla.firefox", True, False, False),
    ("com.microsoft.VSCode", False, True, False),
    ("com.apple.mail", False, False, False),
    ("us.zoom.xos", False, False, False),
    ("com.tinyspeck.slackmacgap", False, False, False),
    ("com.google.Chrome", True, False, False),
    ("com.jetbrains.intellij", False, True, False),
    ("com.apple.finder", False, False, False),
    ("com.apple.Notes", False, False, False),
]


def generate_dummy_features(
    date: dt.date, n_rows: int = DEFAULT_DUMMY_ROWS
) -> list[FeatureRow]:
    """Create *n_rows* synthetic FeatureRow instances spanning *date*.

    Args:
        date: The calendar date to generate buckets for (hours 9-17).
        n_rows: Number of rows to generate.

    Returns:
        Validated ``FeatureRow`` instances with dummy app/keyboard/mouse data.
    """
    rows: list[FeatureRow] = []
    day_of_week = date.weekday()

    for i in range(n_rows):
        hour = 9 + (i * 8 // max(n_rows, 1))
        minute = (i * 7) % 60
        ts = dt.datetime(date.year, date.month, date.day, hour, minute)

        app_id, is_browser, is_editor, is_terminal = _DUMMY_APPS[
            i % len(_DUMMY_APPS)
        ]
        title_hash = stable_hash(f"window-title-{app_id}-{i}")

        rows.append(
            FeatureRow(
                bucket_start_ts=ts,
                schema_version=FeatureSchemaV1.VERSION,
                schema_hash=FeatureSchemaV1.SCHEMA_HASH,
                source_ids=[f"dummy-{i:03d}"],
                app_id=app_id,
                window_title_hash=title_hash,
                is_browser=is_browser,
                is_editor=is_editor,
                is_terminal=is_terminal,
                app_switch_count_last_5m=i % 5,
                keys_per_min=float(40 + i * 10),
                backspace_ratio=round(0.05 + (i % 5) * 0.02, 2),
                shortcut_rate=round(0.1 + (i % 3) * 0.05, 2),
                clicks_per_min=float(3 + i % 8),
                scroll_events_per_min=float(i % 6),
                mouse_distance=float(200 + i * 50),
                hour_of_day=hour,
                day_of_week=day_of_week,
                session_length_so_far=float(i * 5),
            )
        )

    return rows


def build_features_from_aw_events(
    events: Sequence[Event],
    *,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    session_start: dt.datetime | None = None,
    idle_gap_seconds: float = DEFAULT_IDLE_GAP_SECONDS,
) -> list[FeatureRow]:
    """Convert normalised events into per-bucket :class:`FeatureRow` instances.

    Events are grouped into fixed-width time buckets.  For each bucket
    the *dominant* application (longest total duration) is selected and
    its metadata (app ID, title hash, flags) is used to populate the
    context columns.  Keyboard and mouse columns are left as ``None``
    because the AW window watcher does not capture those signals.

    Session detection is performed automatically via idle-gap analysis
    (see :func:`~taskclf.features.sessions.detect_session_boundaries`).
    In online mode the caller may pass a known *session_start* to
    avoid resetting the session each poll cycle.

    Args:
        events: Sorted, normalised events satisfying the
            :class:`~taskclf.core.types.Event` protocol (e.g. from
            :func:`~taskclf.adapters.activitywatch.client.parse_aw_export`).
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

    rows: list[FeatureRow] = []
    for bucket_ts in sorted_buckets:
        evs = bucket_events[bucket_ts]

        # Dominant app: highest total duration in this bucket
        app_durations: dict[str, float] = defaultdict(float)
        for ev in evs:
            app_durations[ev.app_id] += ev.duration_seconds
        dominant_app_id = max(app_durations, key=app_durations.get)  # type: ignore[arg-type]

        dominant_ev = next(ev for ev in evs if ev.app_id == dominant_app_id)

        # App switches in the preceding 5 minutes
        window_start = bucket_ts - dt.timedelta(minutes=DEFAULT_APP_SWITCH_WINDOW_MINUTES)
        apps_in_window: set[str] = set()
        for ev in all_events_sorted:
            if ev.timestamp < window_start:
                continue
            if ev.timestamp >= bucket_ts + dt.timedelta(seconds=bucket_seconds):
                break
            apps_in_window.add(ev.app_id)
        switch_count = max(0, len(apps_in_window) - 1)

        cur_session = session_start_for_bucket(bucket_ts, session_starts)
        elapsed_minutes = (bucket_ts - cur_session).total_seconds() / 60.0

        rows.append(
            FeatureRow(
                bucket_start_ts=bucket_ts,
                schema_version=FeatureSchemaV1.VERSION,
                schema_hash=FeatureSchemaV1.SCHEMA_HASH,
                source_ids=["aw-watcher-window"],
                app_id=dominant_app_id,
                window_title_hash=dominant_ev.window_title_hash,
                is_browser=dominant_ev.is_browser,
                is_editor=dominant_ev.is_editor,
                is_terminal=dominant_ev.is_terminal,
                app_switch_count_last_5m=switch_count,
                keys_per_min=None,
                backspace_ratio=None,
                shortcut_rate=None,
                clicks_per_min=None,
                scroll_events_per_min=None,
                mouse_distance=None,
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
