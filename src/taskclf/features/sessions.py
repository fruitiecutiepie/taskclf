"""Session boundary detection from sorted event sequences.

A *session* is a contiguous run of activity.  A new session starts
whenever the gap between the end of one event (``timestamp +
duration_seconds``) and the start of the next exceeds a configurable
idle-gap threshold (default 5 minutes).
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta
from typing import Sequence

from taskclf.core.types import Event

_DEFAULT_IDLE_GAP_SECONDS: float = 300.0


def detect_session_boundaries(
    events: Sequence[Event],
    idle_gap_seconds: float = _DEFAULT_IDLE_GAP_SECONDS,
) -> list[datetime]:
    """Return the start timestamp of each detected session.

    The input *events* must be sorted by ``timestamp`` (ascending).
    The first event always opens the first session.  Subsequent sessions
    begin when the gap between one event's end and the next event's
    start is >= *idle_gap_seconds*.

    Args:
        events: Sorted normalised events.
        idle_gap_seconds: Minimum gap (in seconds) that splits sessions.

    Returns:
        Sorted list of session-start timestamps (one per session).
        Empty if *events* is empty.
    """
    if not events:
        return []

    gap = timedelta(seconds=idle_gap_seconds)
    starts: list[datetime] = [events[0].timestamp]

    for prev, cur in zip(events, events[1:]):
        prev_end = prev.timestamp + timedelta(seconds=prev.duration_seconds)
        if cur.timestamp - prev_end >= gap:
            starts.append(cur.timestamp)

    return starts


def session_start_for_bucket(
    bucket_ts: datetime,
    session_starts: list[datetime],
) -> datetime:
    """Look up the session that *bucket_ts* belongs to.

    Uses binary search over the sorted *session_starts* list to find
    the latest session start that is <= *bucket_ts*.

    Args:
        bucket_ts: The bucket-aligned timestamp to query.
        session_starts: Sorted output of :func:`detect_session_boundaries`.

    Returns:
        The session-start timestamp for the bucket.
    """
    idx = bisect.bisect_right(session_starts, bucket_ts) - 1
    return session_starts[max(idx, 0)]
