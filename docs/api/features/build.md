# features.build

Feature computation pipeline: convert normalised ActivityWatch events
into bucketed, schema-validated `FeatureRow` instances and optionally
write them to Parquet.

## Pipeline overview

The feature build pipeline operates in three modes:

1. **Batch** (`build_features_from_aw_events`) -- converts a sorted
   sequence of normalised `Event` objects (plus optional `AWInputEvent`
   objects) into per-bucket `FeatureRow` instances.
2. **Dummy** (`generate_dummy_features`) -- produces synthetic feature
   rows for a given date, useful for testing and development.
3. **File** (`build_features_for_date`) -- fetches events from a
   running ActivityWatch server (or generates dummy features when no
   server is available) and writes them to the Parquet partition layout.

The batch pipeline follows this data flow:

```
Events ──► bucket by time ──► dominant app per bucket ──► context features
                                                           │
Input events ──► bucket ──► aggregate (keys/clicks/mouse) ─┘
                                                           │
                        ┌──────────────────────────────────┘
                        ▼
              session detection ──► dynamics tracker ──► FeatureRow
```

## build_features_from_aw_events

Core function that converts normalised events into per-bucket feature
rows.  Events are grouped into fixed-width time buckets (default 60 s).
For each bucket, the **dominant application** (longest total duration)
determines the context columns (app ID, category, title hash, flags).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `events` | `Sequence[Event]` | -- | Sorted normalised events |
| `user_id` | `str` | `"default-user"` | Pseudonymous user identifier |
| `device_id` | `str \| None` | `None` | Optional device identifier |
| `input_events` | `Sequence[AWInputEvent] \| None` | `None` | Input watcher events for keyboard/mouse features |
| `bucket_seconds` | `int` | `DEFAULT_BUCKET_SECONDS` | Width of each time bucket |
| `session_start` | `datetime \| None` | `None` | Fixed session start (online mode); `None` triggers auto-detection |
| `idle_gap_seconds` | `float` | `DEFAULT_IDLE_GAP_SECONDS` | Minimum gap that splits sessions (batch mode) |

When `input_events` is `None`, all keyboard/mouse feature columns
(`keys_per_min`, `clicks_per_min`, etc.) are set to `None`.

When `session_start` is `None` (batch mode), sessions are detected
automatically from idle gaps via
[`features.sessions.detect_session_boundaries`](sessions.md).
In online mode, the caller passes the known session start to avoid
resetting the session each poll cycle.

Sub-modules invoked per bucket:

| Sub-module | Columns produced |
|------------|------------------|
| [`features.windows`](windows.md) | `app_switch_count_last_5m`, `app_switch_count_last_15m` |
| [`features.sessions`](sessions.md) | `session_id`, `session_length_so_far` |
| [`features.domain`](domain.md) | `domain_category` |
| [`features.text`](text.md) | `window_title_bucket`, `title_repeat_count_session` |
| [`features.dynamics`](dynamics.md) | Rolling means and deltas (7 columns) |
| *(inline)* | `app_dwell_time_seconds` |

`app_dwell_time_seconds` is computed directly in
`build_features_from_aw_events` (not via a sub-module).  It tracks how
long the current dominant app has been foreground continuously across
consecutive buckets.  When the dominant app changes the counter resets;
when it stays the same the previous dwell is accumulated.

```python
from taskclf.features.build import build_features_from_aw_events

rows = build_features_from_aw_events(
    events,
    user_id="user-001",
    input_events=input_events,
)
```

## generate_dummy_features

Creates `n_rows` synthetic `FeatureRow` instances spanning hours 9--17
of the given date.  Cycles through 10 dummy applications to produce
realistic variety.  Useful for pipeline testing without real
ActivityWatch data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | `date` | -- | Calendar date to generate buckets for |
| `n_rows` | `int` | `DEFAULT_DUMMY_ROWS` | Number of rows to generate |
| `user_id` | `str` | `"dummy-user-001"` | User identifier |
| `device_id` | `str \| None` | `None` | Optional device identifier |

## build_features_for_date

Builds feature rows for a given date and writes them to Parquet using
the partitioned layout:

```
data_dir/features_v1/date=YYYY-MM-DD/features.parquet
```

When `aw_host` is provided (and `synthetic` is `False`), events are
fetched live from a running ActivityWatch server via the REST API.
Both `aw-watcher-window` and `aw-watcher-input` buckets are queried
automatically.  Without `aw_host` (or with `synthetic=True`), dummy
features are generated for testing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | `date` | -- | Calendar date to build features for |
| `data_dir` | `Path` | -- | Root of processed data |
| `aw_host` | `str \| None` | `None` | AW server URL; `None` falls back to dummy generation |
| `title_salt` | `str \| None` | `None` | Salt for hashing window titles (required when `aw_host` is set) |
| `user_id` | `str` | `"default-user"` | Pseudonymous user identifier |
| `device_id` | `str \| None` | `None` | Optional device identifier |
| `synthetic` | `bool` | `False` | Force dummy feature generation |

Raises `ValueError` if generated data fails schema validation or if
`aw_host` is set without `title_salt`.

## See also

- [`core.schema`](../core/schema.md) -- `FeatureSchemaV1` contract validated by this pipeline
- [`core.types`](../core/types.md) -- `FeatureRow` and `Event` models
- [`adapters.activitywatch`](../adapters/activitywatch.md) -- event normalisation upstream of this pipeline

::: taskclf.features.build
