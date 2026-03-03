# report.daily

Daily report generation from prediction segments.

## Overview

Aggregates [`Segment`](../infer/smooth.md) spans, per-bucket predictions,
and feature-level statistics into a `DailyReport` suitable for
time-tracking summaries:

```
segments + optional per-bucket labels/features
    → core_breakdown (label → minutes)
    → mapped_breakdown (taxonomy label → minutes, optional)
    → flap rates (raw & smoothed)
    → context-switch stats
    → DailyReport
```

## Models

### ContextSwitchStats

Aggregated context-switching statistics for a day, derived from the
`app_switch_count_last_5m` feature across all buckets.

| Field | Type | Description |
|-------|------|-------------|
| `mean` | `float` | Mean app switches per bucket |
| `median` | `float` | Median app switches per bucket |
| `max_value` | `int` | Peak app switches in a single bucket |
| `total_switches` | `int` | Sum of app switches across all buckets |
| `buckets_counted` | `int` | Number of buckets with valid (non-`None`) data |

All fields have a `ge=0` constraint.

### DailyReport

Aggregated daily summary of predicted task-type activity.

| Field | Type | Description |
|-------|------|-------------|
| `date` | `str` | Calendar date (`YYYY-MM-DD`) this report covers |
| `total_minutes` | `float` | Total minutes of activity (sum of `core_breakdown`) |
| `core_breakdown` | `dict[str, float]` | Core label to total minutes mapping |
| `mapped_breakdown` | `dict[str, float] \| None` | Taxonomy label to total minutes (populated when `mapped_labels` are provided) |
| `segments_count` | `int` | Number of segments in the day |
| `context_switch_stats` | `ContextSwitchStats \| None` | App-switching statistics from feature data |
| `flap_rate_raw` | `float \| None` | Label changes / total windows before smoothing |
| `flap_rate_smoothed` | `float \| None` | Label changes / total windows after smoothing |

`total_minutes` and `segments_count` have `ge=0` constraints.  The
model is frozen (immutable after construction).

## Functions

### build_daily_report

```python
build_daily_report(
    segments: Sequence[Segment],
    *,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    raw_labels: Sequence[str] | None = None,
    smoothed_labels: Sequence[str] | None = None,
    mapped_labels: Sequence[str] | None = None,
    app_switch_counts: Sequence[float | int | None] | None = None,
) -> DailyReport
```

Aggregates prediction data for one calendar day into a `DailyReport`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segments` | *(required)* | Prediction segments (typically from [`segmentize`](../infer/smooth.md)) |
| `bucket_seconds` | `DEFAULT_BUCKET_SECONDS` (60) | Width of each time bucket in seconds; scales bucket counts to minutes |
| `raw_labels` | `None` | Per-bucket labels *before* smoothing — used for `flap_rate_raw` |
| `smoothed_labels` | `None` | Per-bucket labels *after* smoothing — used for `flap_rate_smoothed` |
| `mapped_labels` | `None` | Per-bucket taxonomy-mapped labels — used for `mapped_breakdown` |
| `app_switch_counts` | `None` | Per-bucket `app_switch_count_last_5m` values — used for `context_switch_stats` |

Raises `ValueError` if `segments` is empty.

The `date` field is taken from the first segment's `start_ts`.  Flap
rates are computed via [`flap_rate`](../infer/smooth.md) and rounded to
4 decimal places.  The bucket-to-minutes conversion uses
`bucket_count * bucket_seconds / 60`.

## Usage

```python
from taskclf.infer.smooth import segmentize, rolling_majority
from taskclf.report.daily import build_daily_report

segments = segmentize(bucket_starts, smoothed_labels, bucket_seconds=60)

report = build_daily_report(
    segments,
    raw_labels=raw_labels,
    smoothed_labels=smoothed_labels,
    mapped_labels=mapped_labels,
    app_switch_counts=app_switch_counts,
)

print(f"{report.date}: {report.total_minutes:.0f} min across {report.segments_count} segments")
for label, minutes in sorted(report.core_breakdown.items()):
    print(f"  {label}: {minutes:.1f} min")
```

See [`infer.smooth`](../infer/smooth.md) for segment and flap-rate
details, and [`core.defaults`](../core/defaults.md) for
`DEFAULT_BUCKET_SECONDS`.

::: taskclf.report.daily
