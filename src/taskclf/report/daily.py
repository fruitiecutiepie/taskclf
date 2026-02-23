"""Daily report generation from prediction segments."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from pydantic import BaseModel, Field

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.infer.smooth import Segment


class DailyReport(BaseModel, frozen=True):
    """Aggregated daily summary of predicted task-type segments.

    ``breakdown`` maps each label to its total minutes.  ``total_minutes``
    is the sum across all labels and must equal the sum of all segment
    durations.
    """

    date: str = Field(description="Calendar date (YYYY-MM-DD) this report covers.")
    total_minutes: float = Field(ge=0, description="Total minutes of activity.")
    breakdown: dict[str, float] = Field(description="Label -> total minutes mapping.")
    segments_count: int = Field(ge=0, description="Number of segments in the day.")


def build_daily_report(
    segments: Sequence[Segment],
    *,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> DailyReport:
    """Aggregate *segments* into a :class:`DailyReport`.

    Args:
        segments: Prediction segments (typically from one calendar day).
        bucket_seconds: Width of each time bucket in seconds (used to
            convert bucket counts to minutes).

    Returns:
        A ``DailyReport`` with per-label totals and overall duration.

    Raises:
        ValueError: If *segments* is empty.
    """
    if not segments:
        raise ValueError("Cannot build a daily report from zero segments")

    label_minutes: dict[str, float] = defaultdict(float)
    for seg in segments:
        minutes = seg.bucket_count * bucket_seconds / 60.0
        label_minutes[seg.label] += minutes

    total = sum(label_minutes.values())
    date_str = segments[0].start_ts.date().isoformat()

    return DailyReport(
        date=date_str,
        total_minutes=round(total, 2),
        breakdown=dict(label_minutes),
        segments_count=len(segments),
    )
