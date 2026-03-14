"""Time-bucket alignment and range generation.

All bucket logic operates on UTC datetimes.  Timezone-aware inputs are
converted to UTC before alignment so that DST transitions never produce
duplicate or missing buckets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS


def to_naive_utc(ts: datetime) -> datetime:
    """Normalize a datetime to naive UTC.

    Aware datetimes are converted to UTC then stripped of tzinfo.
    Naive datetimes are returned as-is (assumed to already represent UTC).

    This is the canonical conversion for timestamps entering the feature
    pipeline or Parquet storage, where the naive-UTC convention is used.
    """
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def align_to_bucket(
    ts: datetime,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> datetime:
    """Floor *ts* to the nearest bucket boundary.

    If *ts* is timezone-aware it is first converted to UTC.  Naive
    datetimes are assumed to represent UTC.  The returned datetime is
    always a **timezone-aware** UTC value (``tzinfo=timezone.utc``).

    Args:
        ts: Timestamp to align.
        bucket_seconds: Bucket width in seconds (default 60).

    Returns:
        A timezone-aware UTC datetime whose POSIX epoch is a multiple of
        *bucket_seconds*.
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

    epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
    aligned_epoch = (epoch // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)


def generate_bucket_range(
    start: datetime,
    end: datetime,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[datetime]:
    """Enumerate bucket-start timestamps from *start* to *end* (exclusive).

    Both *start* and *end* are aligned first, so callers need not pre-align.

    Args:
        start: Earliest timestamp (inclusive after alignment).
        end: Latest timestamp (exclusive after alignment).
        bucket_seconds: Bucket width in seconds (default 60).

    Returns:
        Sorted list of timezone-aware UTC bucket-start datetimes.
    """
    cur = align_to_bucket(start, bucket_seconds)
    end_aligned = align_to_bucket(end, bucket_seconds)
    step = timedelta(seconds=bucket_seconds)

    buckets: list[datetime] = []
    while cur < end_aligned:
        buckets.append(cur)
        cur += step
    return buckets
