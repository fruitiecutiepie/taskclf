# infer.monitor

Drift monitoring orchestrator: runs all drift checks and optionally
auto-creates labeling tasks when triggers fire.

## Overview

Ties together the pure drift statistics from `core.drift` with the
telemetry store and the active labeling queue.

## Models

### DriftTrigger

StrEnum of trigger types:

- `feature_psi` — Feature PSI exceeded threshold
- `feature_ks` — Feature KS test significant
- `reject_rate_increase` — Reject rate increased beyond threshold
- `entropy_spike` — Mean prediction entropy spiked
- `class_shift` — Class distribution shifted

### DriftAlert

| Field | Type | Description |
|-------|------|-------------|
| `trigger` | `DriftTrigger` | Which check triggered |
| `details` | `dict` | Trigger-specific data |
| `severity` | `"warning" \| "critical"` | Alert severity |
| `affected_user_ids` | `list[str]` | Users affected |
| `affected_features` | `list[str]` | Features affected |
| `timestamp` | `datetime` | When alert was raised |

### DriftReport

| Field | Type | Description |
|-------|------|-------------|
| `alerts` | `list[DriftAlert]` | All alerts raised |
| `feature_report` | `FeatureDriftReport \| None` | Per-feature PSI/KS |
| `reject_rate_drift` | `RejectRateDrift \| None` | Reject rate comparison |
| `entropy_drift` | `EntropyDrift \| None` | Entropy comparison |
| `class_shift` | `ClassShiftResult \| None` | Class distribution shift |
| `telemetry_snapshot` | `TelemetrySnapshot \| None` | Current-window telemetry |
| `summary` | `str` | Human-readable summary |
| `any_critical` | `bool` | Whether any critical alert fired |

## Functions

### run_drift_check

Run all drift checks and return a consolidated report.

```python
from taskclf.infer.monitor import run_drift_check

report = run_drift_check(
    ref_features_df, cur_features_df,
    ref_labels, cur_labels,
    ref_probs=ref_probs,
    cur_probs=cur_probs,
)
print(report.summary)
```

### auto_enqueue_drift_labels

Create labeling tasks for drifted buckets.  Selects buckets with the
lowest confidence from the current window and enqueues them via
`ActiveLabelingQueue`.

```python
from taskclf.infer.monitor import auto_enqueue_drift_labels

count = auto_enqueue_drift_labels(
    report, cur_features_df,
    queue_path=Path("data/processed/labels_v1/queue.json"),
    cur_confidences=confidences,
    limit=50,
)
```

### write_drift_report

Persist a drift report as JSON.

```python
from taskclf.infer.monitor import write_drift_report
write_drift_report(report, Path("artifacts/drift_report.json"))
```

::: taskclf.infer.monitor
