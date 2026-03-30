"""Rolling-window aggregations over event streams.

Provides functions that compute windowed metrics (e.g. unique-app switch
counts, app-distribution entropy) across a sorted sequence of events.
These are extracted from the inline logic in :mod:`~taskclf.features.build`
so they can be tested and reused independently.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Sequence

from taskclf.core.defaults import (
    DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    DEFAULT_BUCKET_SECONDS,
)
from taskclf.core.time import ts_utc_aware_get
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

    def _epoch(ts: dt.datetime) -> float:
        return ts_utc_aware_get(ts).timestamp()

    ws_epoch = _epoch(window_start)
    we_epoch = _epoch(window_end)

    apps: set[str] = set()
    for ev in events:
        ev_epoch = _epoch(ev.timestamp)
        if ev_epoch < ws_epoch:
            continue
        if ev_epoch >= we_epoch:
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


def app_entropy_in_window(
    events: Sequence[Event],
    bucket_ts: dt.datetime,
    window_minutes: int = DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> float | None:
    """Shannon entropy of the app duration distribution in a look-back window.

    The window spans ``[bucket_ts - window_minutes, bucket_ts + bucket_seconds)``.
    Duration per app is summed, converted to probabilities, and entropy
    is computed as ``H = -sum(p_i * log2(p_i))``.

    Args:
        events: Chronologically sorted events satisfying the
            :class:`~taskclf.core.types.Event` protocol.
        bucket_ts: The aligned start of the current time bucket.
        window_minutes: How many minutes to look back from *bucket_ts*.
        bucket_seconds: Width of the current bucket in seconds.

    Returns:
        Non-negative Shannon entropy in bits, or ``None`` when no events
        fall within the window.
    """
    window_start = bucket_ts - dt.timedelta(minutes=window_minutes)
    window_end = bucket_ts + dt.timedelta(seconds=bucket_seconds)

    def _epoch(ts: dt.datetime) -> float:
        return ts_utc_aware_get(ts).timestamp()

    ws_epoch = _epoch(window_start)
    we_epoch = _epoch(window_end)

    app_durations: dict[str, float] = defaultdict(float)
    for ev in events:
        ev_epoch = _epoch(ev.timestamp)
        if ev_epoch < ws_epoch:
            continue
        if ev_epoch >= we_epoch:
            break
        app_durations[ev.app_id] += ev.duration_seconds

    if not app_durations:
        return None

    total = sum(app_durations.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for dur in app_durations.values():
        p = dur / total
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


def top2_app_concentration_in_window(
    events: Sequence[Event],
    bucket_ts: dt.datetime,
    window_minutes: int = DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> float | None:
    """Combined time share of the two most-used apps in a look-back window.

    The window spans ``[bucket_ts - window_minutes, bucket_ts + bucket_seconds)``.
    Duration per app is summed; the two largest shares are added together
    and returned as a value in ``[0, 1]``.

    Args:
        events: Chronologically sorted events satisfying the
            :class:`~taskclf.core.types.Event` protocol.
        bucket_ts: The aligned start of the current time bucket.
        window_minutes: How many minutes to look back from *bucket_ts*.
        bucket_seconds: Width of the current bucket in seconds.

    Returns:
        Concentration ratio in ``[0, 1]``, or ``None`` when no events
        fall within the window.
    """
    window_start = bucket_ts - dt.timedelta(minutes=window_minutes)
    window_end = bucket_ts + dt.timedelta(seconds=bucket_seconds)

    def _epoch(ts: dt.datetime) -> float:
        return ts_utc_aware_get(ts).timestamp()

    ws_epoch = _epoch(window_start)
    we_epoch = _epoch(window_end)

    app_durations: dict[str, float] = defaultdict(float)
    for ev in events:
        ev_epoch = _epoch(ev.timestamp)
        if ev_epoch < ws_epoch:
            continue
        if ev_epoch >= we_epoch:
            break
        app_durations[ev.app_id] += ev.duration_seconds

    if not app_durations:
        return None

    total = sum(app_durations.values())
    if total <= 0:
        return None

    top2 = sorted(app_durations.values(), reverse=True)[:2]
    return round(sum(top2) / total, 4)
