"""Post-prediction smoothing and segmentization.

Smoothing reduces short prediction spikes via majority vote.
Segmentization merges consecutive identical labels into contiguous spans.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from taskclf.core.defaults import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_SMOOTH_WINDOW,
    MIN_BLOCK_DURATION_SECONDS,
)


@dataclass(frozen=True)
class Segment:
    """A contiguous run of identical predicted labels."""

    start_ts: datetime
    end_ts: datetime
    label: str
    bucket_count: int


def rolling_majority(labels: Sequence[str], window: int = DEFAULT_SMOOTH_WINDOW) -> list[str]:
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
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
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


def flap_rate(labels: Sequence[str]) -> float:
    """Compute label flap rate: label changes / total windows.

    As defined in ``docs/guide/acceptance.md`` section 5.
    Acceptance thresholds: raw <= 0.25, smoothed <= 0.15.

    Args:
        labels: Ordered sequence of predicted labels (one per bucket).

    Returns:
        Flap rate in [0, 1].  Returns 0.0 for sequences of length <= 1.
    """
    if len(labels) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return changes / len(labels)


def merge_short_segments(
    segments: Sequence[Segment],
    *,
    min_duration_seconds: int = MIN_BLOCK_DURATION_SECONDS,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[Segment]:
    """Absorb segments shorter than *min_duration_seconds* into neighbours.

    Implements the hysteresis rule from the time spec: label changes
    lasting less than ``MIN_BLOCK_DURATION_SECONDS`` (default 180 s /
    3 minutes) are smoothed into the surrounding label.

    Strategy for each short segment:

    1. If either neighbour has the same label, merge into that neighbour.
    2. Otherwise merge into the *longer* neighbour (prefer the
       preceding one on ties).
    3. First and last segments are never removed (but may absorb
       neighbours).

    The function iterates until no more short segments can be merged,
    guaranteeing convergence because total segment count strictly
    decreases each pass.

    Args:
        segments: Ordered, non-overlapping segments (as from
            :func:`segmentize`).
        min_duration_seconds: Minimum block duration in seconds.
        bucket_seconds: Width of each time bucket in seconds.

    Returns:
        A new list of segments with short blocks absorbed.
    """
    if len(segments) <= 1:
        return list(segments)

    min_buckets = max(1, min_duration_seconds // bucket_seconds)
    result = list(segments)

    changed = True
    while changed:
        changed = False
        i = 0
        merged: list[Segment] = []

        while i < len(result):
            seg = result[i]
            if seg.bucket_count >= min_buckets or len(result) <= 1:
                merged.append(seg)
                i += 1
                continue

            prev = merged[-1] if merged else None
            nxt = result[i + 1] if i + 1 < len(result) else None

            if prev is not None and prev.label == seg.label:
                merged[-1] = Segment(
                    start_ts=prev.start_ts,
                    end_ts=seg.end_ts,
                    label=prev.label,
                    bucket_count=prev.bucket_count + seg.bucket_count,
                )
                changed = True
                i += 1
                continue

            if nxt is not None and nxt.label == seg.label:
                merged.append(Segment(
                    start_ts=seg.start_ts,
                    end_ts=nxt.end_ts,
                    label=seg.label,
                    bucket_count=seg.bucket_count + nxt.bucket_count,
                ))
                changed = True
                i += 2
                continue

            if prev is None:
                merged.append(seg)
                i += 1
                continue
            if nxt is None:
                merged[-1] = Segment(
                    start_ts=prev.start_ts,
                    end_ts=seg.end_ts,
                    label=prev.label,
                    bucket_count=prev.bucket_count + seg.bucket_count,
                )
                changed = True
                i += 1
                continue

            prev_count = prev.bucket_count
            nxt_count = nxt.bucket_count
            if prev_count >= nxt_count:
                merged[-1] = Segment(
                    start_ts=prev.start_ts,
                    end_ts=seg.end_ts,
                    label=prev.label,
                    bucket_count=prev.bucket_count + seg.bucket_count,
                )
            else:
                merged.append(Segment(
                    start_ts=seg.start_ts,
                    end_ts=nxt.end_ts,
                    label=nxt.label,
                    bucket_count=seg.bucket_count + nxt.bucket_count,
                ))
                i += 1
            changed = True
            i += 1

        result = merged

    return result
