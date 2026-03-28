# infer.feature_state

Persistent rolling feature state for the online inference loop.

## Overview

The online loop polls ActivityWatch in short windows, so
`build_features_from_aw_events()` only sees a narrow slice of recent
events.  Rolling features (15-minute switch counts, rolling
keyboard/mouse means, deltas, session length) are therefore truncated
to the poll window rather than reflecting the full history the model was
trained on.

`OnlineFeatureState` solves this by maintaining a circular buffer of
recent `FeatureRow` values across poll cycles.  After each row is built,
it is pushed into the state, and `get_context()` returns corrected
rolling aggregates that are overlaid onto the row before prediction.

## Pipeline position

```
poll AW events → build_features_from_aw_events()
    → feature_state.push(row)
    → context = feature_state.get_context()
    → row.model_copy(update=context)
    → predictor.predict_bucket(row)
```

## OnlineFeatureState

Circular buffer of recent `FeatureRow` values with rolling aggregate
computation.  Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_minutes` | `int` | `15` | Minutes of history to retain |
| `bucket_seconds` | `int` | `60` | Width of each time bucket in seconds |
| `idle_gap_seconds` | `float` | `300.0` | Gap between rows that triggers a session reset |

### Methods

#### `push(row: FeatureRow) -> None`

Record a newly built feature row.  Feeds the row's input metrics into
an internal `DynamicsTracker` and detects idle-gap session boundaries.

#### `get_context() -> dict`

Return rolling aggregates derived from the full buffer.  The returned
dict maps `FeatureRow` field names to corrected values:

| Key | Type | Description |
|-----|------|-------------|
| `app_switch_count_last_15m` | `int` | Unique app switches across the buffered 15-minute window |
| `keys_per_min_rolling_5` | `float \| None` | 5-bucket rolling mean of `keys_per_min` |
| `keys_per_min_rolling_15` | `float \| None` | 15-bucket rolling mean of `keys_per_min` |
| `mouse_distance_rolling_5` | `float \| None` | 5-bucket rolling mean of `mouse_distance` |
| `mouse_distance_rolling_15` | `float \| None` | 15-bucket rolling mean of `mouse_distance` |
| `keys_per_min_delta` | `float \| None` | Change in `keys_per_min` from previous bucket |
| `clicks_per_min_delta` | `float \| None` | Change in `clicks_per_min` from previous bucket |
| `mouse_distance_delta` | `float \| None` | Change in `mouse_distance` from previous bucket |
| `session_length_so_far` | `float` | Minutes since the current session started |

### Session tracking

A new session starts when the gap between consecutive pushed rows
exceeds `idle_gap_seconds`.  The `session_length_so_far` field resets
to `0.0` at the boundary.

### Model hot-reload

When the online loop hot-reloads a new model, the `OnlineFeatureState`
instance is **preserved** (not reset), since it tracks feature history
rather than model state.

::: taskclf.infer.feature_state
