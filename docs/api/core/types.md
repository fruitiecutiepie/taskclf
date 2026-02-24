# core.types

Pydantic models for the core data contracts.

## FeatureRow identity fields

Every `FeatureRow` carries stable identity columns alongside the schema
metadata and feature values:

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `str` | Random UUID identifying the user (not PII). |
| `device_id` | `str \| None` | Optional device identifier. |
| `session_id` | `str` | Deterministic session identifier derived from `user_id` + session start timestamp. |
| `bucket_start_ts` | `datetime` | Start of the 60 s bucket (UTC, inclusive). |
| `bucket_end_ts` | `datetime` | End of the 60 s bucket (UTC, exclusive). |

The primary key is `(user_id, bucket_start_ts)`.

## LabelSpan fields

`LabelSpan` represents a contiguous time span carrying a single task-type
label.  Gold labels and weak labels share this structure.

| Field | Type | Description |
|-------|------|-------------|
| `start_ts` | `datetime` | Span start (UTC, inclusive). |
| `end_ts` | `datetime` | Span end (UTC, exclusive). |
| `label` | `str` | Task-type label from `LABEL_SET_V1`. |
| `provenance` | `str` | Origin tag, e.g. `"manual"` or `"weak:app_rule"`. |
| `user_id` | `str \| None` | User who created this label (optional, default `None`). |
| `confidence` | `float \| None` | Labeler confidence 0-1 (optional, default `None`). |

## TitlePolicy

`TitlePolicy` controls whether raw window titles may appear in a
`FeatureRow`.

| Member | Value | Behaviour |
|---|---|---|
| `HASH_ONLY` | `"hash_only"` | Default. All `raw_*` fields are rejected. |
| `RAW_WINDOW_TITLE_OPT_IN` | `"raw_window_title_opt_in"` | `raw_window_title` is accepted but excluded from `model_dump()`, preventing leakage into `data/processed/`. All other `raw_*` fields remain prohibited. |

Pass the policy via Pydantic validation context:

```python
from taskclf.core.types import FeatureRow, TitlePolicy

row = FeatureRow.model_validate(
    data,
    context={"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN},
)
row.raw_window_title   # available on the instance
row.model_dump()       # raw_window_title is NOT included
```

::: taskclf.core.types
