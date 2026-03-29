# core.telemetry

Point-in-time quality telemetry: snapshot computation and persistence.

## Overview

Collects aggregate statistics on feature quality, prediction confidence,
reject rates, and class distributions.  Stores only numerical summaries;
never raw content.
`TelemetryStore` is implemented as a slotted dataclass and keeps the
same constructor argument (`store_dir`).

## TelemetrySnapshot

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | When the snapshot was computed |
| `user_id` | `str \| None` | Scoped user (None = global) |
| `window_start` | `datetime \| None` | Earliest bucket in the window |
| `window_end` | `datetime \| None` | Latest bucket in the window |
| `total_windows` | `int` | Number of prediction windows |
| `feature_missingness` | `dict[str, float]` | Fraction missing per feature |
| `confidence_stats` | `ConfidenceStats \| None` | mean, median, p5, p95, std |
| `reject_rate` | `float` | Fraction of rejected predictions |
| `mean_entropy` | `float` | Mean prediction entropy |
| `class_distribution` | `dict[str, float]` | Fraction per class |
| `schema_version` | `str` | Feature schema version |
| `suggestions_per_day` | `int` | Number of suggestions surfaced on the snapshot's day (default 0) |

## compute_telemetry

```python
from taskclf.core.telemetry import compute_telemetry
snapshot = compute_telemetry(
    features_df,
    labels=predicted_labels,
    confidences=confidence_array,
    core_probs=probability_matrix,
    user_id="user-1",
)
```

## TelemetryStore

Append-only JSONL store.  One file per user (or global).

```python
from taskclf.core.telemetry import TelemetryStore

store = TelemetryStore("artifacts/telemetry")
store.append(snapshot)

recent = store.read_recent(10, user_id="user-1")
in_range = store.read_range(start_dt, end_dt)
```

File layout:

```
artifacts/telemetry/
  telemetry_global.jsonl
  telemetry_user-1.jsonl
  telemetry_user-2.jsonl
```

## SuggestionTracker

In-memory counter of suggestion events grouped by calendar date.
Implements the decision-#4 guardrail: a loaded model must produce
at least one suggestion per active day.

```python
from datetime import datetime, timezone
from taskclf.core.telemetry import SuggestionTracker

tracker = SuggestionTracker()

# Record a suggestion event
tracker.record(datetime.now(tz=timezone.utc))

# Query the count
count = tracker.count_for_date("2026-03-28")

# End-of-day check (logs a warning if zero suggestions with a loaded model)
tracker.check_zero_suggestions("2026-03-28", model_loaded=True)
```

| Method | Description |
|--------|-------------|
| `record(ts)` | Increment the suggestion count for the date derived from `ts` |
| `count_for_date(date_str)` | Return the count for a `YYYY-MM-DD` date string |
| `check_zero_suggestions(date_str, *, model_loaded)` | Log a warning if `model_loaded` is True and count is 0 |

::: taskclf.core.telemetry
