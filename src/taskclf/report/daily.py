"""Daily report generation from prediction segments.

Aggregates segments, per-bucket predictions, and feature-level statistics
into a :class:`DailyReport` suitable for time-tracking summaries.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Sequence

from pydantic import BaseModel, Field

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.infer.smooth import Segment, flap_rate


class ContextSwitchStats(BaseModel, frozen=True):
    """Aggregated context-switching statistics for a day.

    Derived from the ``app_switch_count_last_5m`` feature across all
    buckets in a day.
    """

    mean: float = Field(ge=0, description="Mean app switches per bucket.")
    median: float = Field(ge=0, description="Median app switches per bucket.")
    max_value: int = Field(ge=0, description="Peak app switches in a single bucket.")
    total_switches: int = Field(ge=0, description="Sum of app switches across all buckets.")
    buckets_counted: int = Field(ge=0, description="Number of buckets with valid data.")


class DailyReport(BaseModel, frozen=True):
    """Aggregated daily summary of predicted task-type activity.

    ``core_breakdown`` maps each core label to its total minutes.
    ``mapped_breakdown`` does the same for user-facing taxonomy buckets
    (populated when per-bucket mapped labels are provided).
    """

    date: str = Field(description="Calendar date (YYYY-MM-DD) this report covers.")
    total_minutes: float = Field(ge=0, description="Total minutes of activity.")
    core_breakdown: dict[str, float] = Field(
        description="Core label -> total minutes mapping."
    )
    mapped_breakdown: dict[str, float] | None = Field(
        default=None, description="Mapped (taxonomy) label -> total minutes."
    )
    segments_count: int = Field(ge=0, description="Number of segments in the day.")
    context_switch_stats: ContextSwitchStats | None = Field(
        default=None,
        description="App-switching statistics from feature data.",
    )
    flap_rate_raw: float | None = Field(
        default=None,
        description="Label changes / total windows before smoothing.",
    )
    flap_rate_smoothed: float | None = Field(
        default=None,
        description="Label changes / total windows after smoothing.",
    )


def _build_context_switch_stats(
    app_switch_counts: Sequence[float | int | None],
) -> ContextSwitchStats | None:
    valid = [int(v) for v in app_switch_counts if v is not None]
    if not valid:
        return None
    return ContextSwitchStats(
        mean=round(statistics.mean(valid), 2),
        median=round(statistics.median(valid), 2),
        max_value=max(valid),
        total_switches=sum(valid),
        buckets_counted=len(valid),
    )


def build_daily_report(
    segments: Sequence[Segment],
    *,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    raw_labels: Sequence[str] | None = None,
    smoothed_labels: Sequence[str] | None = None,
    mapped_labels: Sequence[str] | None = None,
    app_switch_counts: Sequence[float | int | None] | None = None,
) -> DailyReport:
    """Aggregate prediction data into a :class:`DailyReport`.

    Args:
        segments: Prediction segments (typically from one calendar day).
        bucket_seconds: Width of each time bucket in seconds (used to
            convert bucket counts to minutes).
        raw_labels: Per-bucket labels *before* smoothing — used for
            ``flap_rate_raw``.
        smoothed_labels: Per-bucket labels *after* smoothing — used for
            ``flap_rate_smoothed``.
        mapped_labels: Per-bucket taxonomy-mapped labels — used for
            ``mapped_breakdown``.
        app_switch_counts: Per-bucket ``app_switch_count_last_5m`` values
            from the feature data — used for ``context_switch_stats``.

    Returns:
        A ``DailyReport`` with per-label totals, flap rates, and
        context-switching statistics.

    Raises:
        ValueError: If *segments* is empty.
    """
    if not segments:
        raise ValueError("Cannot build a daily report from zero segments")

    core_minutes: dict[str, float] = defaultdict(float)
    for seg in segments:
        minutes = seg.bucket_count * bucket_seconds / 60.0
        core_minutes[seg.label] += minutes

    total = sum(core_minutes.values())
    date_str = segments[0].start_ts.date().isoformat()

    mapped_breakdown: dict[str, float] | None = None
    if mapped_labels is not None:
        mb: dict[str, float] = defaultdict(float)
        bucket_minutes = bucket_seconds / 60.0
        for lbl in mapped_labels:
            mb[lbl] += bucket_minutes
        mapped_breakdown = dict(mb)

    ctx_stats = (
        _build_context_switch_stats(app_switch_counts)
        if app_switch_counts is not None
        else None
    )

    return DailyReport(
        date=date_str,
        total_minutes=round(total, 2),
        core_breakdown=dict(core_minutes),
        mapped_breakdown=mapped_breakdown,
        segments_count=len(segments),
        context_switch_stats=ctx_stats,
        flap_rate_raw=round(flap_rate(raw_labels), 4) if raw_labels is not None else None,
        flap_rate_smoothed=round(flap_rate(smoothed_labels), 4) if smoothed_labels is not None else None,
    )
