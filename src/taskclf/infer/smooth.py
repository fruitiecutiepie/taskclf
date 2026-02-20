"""Post-prediction smoothing and segmentization.

Smoothing reduces short prediction spikes via majority vote.
Segmentization merges consecutive identical labels into contiguous spans.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence


@dataclass(frozen=True)
class Segment:
    """A contiguous run of identical predicted labels."""

    start_ts: datetime
    end_ts: datetime
    label: str
    bucket_count: int


def rolling_majority(labels: Sequence[str], window: int = 3) -> list[str]:
    """Smooth *labels* with a centred sliding-window majority vote.

    For each position the most common label within the window wins.
    Ties are broken by keeping the original label.

    Args:
        labels: Ordered sequence of predicted labels (one per bucket).
        window: Odd window size centred on each position.

    Returns:
        Smoothed label list of the same length as *labels*.
    """
    if not labels:
        return []

    n = len(labels)
    half = window // 2
    smoothed: list[str] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts = Counter(labels[lo:hi])
        winner, _ = counts.most_common(1)[0]
        if counts[winner] == counts[labels[i]]:
            smoothed.append(labels[i])
        else:
            smoothed.append(winner)

    return smoothed


def segmentize(
    bucket_starts: Sequence[datetime],
    labels: Sequence[str],
    bucket_seconds: int = 60,
) -> list[Segment]:
    """Merge runs of identical labels into :class:`Segment` spans.

    Args:
        bucket_starts: Sorted bucket-start timestamps (one per bucket).
        labels: Predicted (or smoothed) label per bucket, same length.
        bucket_seconds: Width of each bucket in seconds.

    Returns:
        Ordered, non-overlapping list of segments covering every input bucket.

    Raises:
        ValueError: If *bucket_starts* and *labels* differ in length or are empty.
    """
    if len(bucket_starts) != len(labels):
        raise ValueError(
            f"bucket_starts ({len(bucket_starts)}) and labels ({len(labels)}) "
            "must have the same length"
        )
    if not bucket_starts:
        return []

    step = timedelta(seconds=bucket_seconds)
    segments: list[Segment] = []

    run_start = bucket_starts[0]
    run_label = labels[0]
    run_count = 1

    for i in range(1, len(labels)):
        if labels[i] == run_label:
            run_count += 1
        else:
            segments.append(Segment(
                start_ts=run_start,
                end_ts=bucket_starts[i - 1] + step,
                label=run_label,
                bucket_count=run_count,
            ))
            run_start = bucket_starts[i]
            run_label = labels[i]
            run_count = 1

    segments.append(Segment(
        start_ts=run_start,
        end_ts=bucket_starts[-1] + step,
        label=run_label,
        bucket_count=run_count,
    ))

    return segments
