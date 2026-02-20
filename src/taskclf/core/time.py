"""Time-bucket alignment and range generation.

All bucket logic operates on UTC datetimes.  Timezone-aware inputs are
converted to UTC before alignment so that DST transitions never produce
duplicate or missing buckets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

_DEFAULT_BUCKET_SECONDS = 60


def align_to_bucket(
    ts: datetime,
    bucket_seconds: int = _DEFAULT_BUCKET_SECONDS,
) -> datetime:
    """Floor *ts* to the nearest bucket boundary.

    If *ts* is timezone-aware it is first converted to UTC; the returned
    datetime is always a naive UTC value (consistent with how feature rows
    store ``bucket_start_ts``).

    Args:
        ts: Timestamp to align.
        bucket_seconds: Bucket width in seconds (default 60).

    Returns:
        A naive UTC datetime whose POSIX epoch is a multiple of
        *bucket_seconds*.
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

    epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
    aligned_epoch = (epoch // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(aligned_epoch, tz=timezone.utc).replace(tzinfo=None)


def generate_bucket_range(
    start: datetime,
    end: datetime,
    bucket_seconds: int = _DEFAULT_BUCKET_SECONDS,
) -> list[datetime]:
    """Enumerate bucket-start timestamps from *start* to *end* (exclusive).

    Both *start* and *end* are aligned first, so callers need not pre-align.

    Args:
        start: Earliest timestamp (inclusive after alignment).
        end: Latest timestamp (exclusive after alignment).
        bucket_seconds: Bucket width in seconds (default 60).

    Returns:
        Sorted list of naive-UTC bucket-start datetimes.
    """
    cur = align_to_bucket(start, bucket_seconds)
    end_aligned = align_to_bucket(end, bucket_seconds)
    step = timedelta(seconds=bucket_seconds)

    buckets: list[datetime] = []
    while cur < end_aligned:
        buckets.append(cur)
        cur += step
    return buckets
