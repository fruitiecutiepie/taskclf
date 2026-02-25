# core.telemetry

Point-in-time quality telemetry: snapshot computation and persistence.

## Overview

Collects aggregate statistics on feature quality, prediction confidence,
reject rates, and class distributions.  Stores only numerical summaries;
never raw content.

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

::: taskclf.core.telemetry
