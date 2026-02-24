"""Rolling-window aggregations over event streams.

Provides functions that compute windowed metrics (e.g. unique-app switch
counts) across a sorted sequence of events.  These are extracted from
the inline logic in :mod:`~taskclf.features.build` so they can be tested
and reused independently.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

from taskclf.core.defaults import DEFAULT_APP_SWITCH_WINDOW_MINUTES, DEFAULT_BUCKET_SECONDS
from taskclf.core.types import Event


def app_switch_count_in_window(
    events: Sequence[Event],
    bucket_ts: dt.datetime,
    window_minutes: int = DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> int:
    """Count unique-app switches in the look-back window ending at *bucket_ts*.

    The window spans ``[bucket_ts - window_minutes, bucket_ts + bucket_seconds)``.
    The return value is ``max(0, unique_apps - 1)`` — i.e. one app means zero
    switches, two apps means one switch, etc.

    Args:
        events: Chronologically sorted events satisfying the
            :class:`~taskclf.core.types.Event` protocol.
        bucket_ts: The aligned start of the current time bucket.
        window_minutes: How many minutes to look back from *bucket_ts*.
        bucket_seconds: Width of the current bucket in seconds.

    Returns:
        Non-negative count of app switches within the window.
    """
    window_start = bucket_ts - dt.timedelta(minutes=window_minutes)
    window_end = bucket_ts + dt.timedelta(seconds=bucket_seconds)

    apps: set[str] = set()
    for ev in events:
        if ev.timestamp < window_start:
            continue
        if ev.timestamp >= window_end:
            break
        apps.add(ev.app_id)

    return max(0, len(apps) - 1)


def compute_rolling_app_switches(
    sorted_events: Sequence[Event],
    sorted_buckets: Sequence[dt.datetime],
    window_minutes: int = DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[int]:
    """Compute :func:`app_switch_count_in_window` for every bucket.

    Args:
        sorted_events: Chronologically sorted events.
        sorted_buckets: Bucket timestamps in ascending order.
        window_minutes: Look-back window in minutes.
        bucket_seconds: Width of each bucket in seconds.

    Returns:
        A list of switch counts, one per bucket, in the same order as
        *sorted_buckets*.
    """
    return [
        app_switch_count_in_window(sorted_events, bt, window_minutes, bucket_seconds)
        for bt in sorted_buckets
    ]
