# core.schema

Feature schema versioning, deterministic hashing, and DataFrame
validation.  The schema is the **versioned contract** between feature
producers (`features.build`) and consumers (`train`, `infer`).  Per
`AGENTS.md`, inference must refuse to run when the schema hash
recorded in a model bundle differs from the hash of the feature
pipeline that produced the input data.

## FeatureSchemaV1

Central class that owns the canonical column registry, the schema hash,
and both row-level and DataFrame-level validators.
`FeatureSchemaV1` is implemented as a frozen slotted dataclass with
class-level constants (`VERSION`, `COLUMNS`, `SCHEMA_HASH`).

| Attribute | Type | Description |
|-----------|------|-------------|
| `VERSION` | `str` | `"v1"` -- schema generation tag |
| `COLUMNS` | `dict[str, type]` | Ordered column-name to Python-type mapping (41 columns) |
| `SCHEMA_HASH` | `str` | Deterministic hex digest derived from column names + types |

The hash is computed at import time by JSON-serialising the ordered
`[[name, type_name], ...]` pairs and passing them through
[`stable_hash`](hashing.md).  Any column addition, removal, rename,
or type change produces a different hash automatically.

## Column registry

Columns are grouped by role.  All columns are required; nullable
fields (e.g. `keys_per_min` when no input watcher is present) are
typed `float` but may contain `None` at the Pydantic model level.

### Identity and time

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | `str` | Pseudonymous user identifier |
| `device_id` | `str` | Optional device identifier |
| `session_id` | `str` | Hash-based session ID (see [`features.sessions`](../features/sessions.md)) |
| `bucket_start_ts` | `datetime` | UTC-aligned bucket start |
| `bucket_end_ts` | `datetime` | `bucket_start_ts + bucket_seconds` |

### Schema metadata

| Column | Type | Description |
|--------|------|-------------|
| `schema_version` | `str` | Must equal `FeatureSchemaV1.VERSION` |
| `schema_hash` | `str` | Must equal `FeatureSchemaV1.SCHEMA_HASH` |
| `source_ids` | `list` | Collector IDs that contributed (e.g. `["aw-watcher-window"]`) |

### Application context

| Column | Type | Description |
|--------|------|-------------|
| `app_id` | `str` | Bundle ID of the dominant app in the bucket |
| `app_category` | `str` | Semantic category (e.g. `"editor"`, `"browser"`) |
| `window_title_hash` | `str` | Privacy-safe hash of the window title |
| `is_browser` | `bool` | Whether the dominant app is a browser |
| `is_editor` | `bool` | Whether the dominant app is a code editor |
| `is_terminal` | `bool` | Whether the dominant app is a terminal |
| `domain_category` | `str` | Browser domain classification (see [`features.domain`](../features/domain.md)) |
| `window_title_bucket` | `int` | Hash-bucketed title ID (see [`features.text`](../features/text.md)) |
| `title_repeat_count_session` | `int` | How many times this title hash appeared in the current session |

### App-switching metrics

| Column | Type | Description |
|--------|------|-------------|
| `app_switch_count_last_5m` | `int` | Unique-app switches in the 5-minute look-back window |
| `app_switch_count_last_15m` | `int` | Same metric over 15 minutes |
| `app_foreground_time_ratio` | `float` | Fraction of the bucket the dominant app was foreground |
| `app_change_count` | `int` | App changes within the bucket itself |
| `top2_app_concentration_15m` | `float` | Combined time share of the two most-used apps over the last 15 minutes |

### Input activity

| Column | Type | Description |
|--------|------|-------------|
| `keys_per_min` | `float` | Keystrokes per minute (aggregate, no raw keys stored) |
| `backspace_ratio` | `float` | Fraction of keystrokes that are backspace |
| `shortcut_rate` | `float` | Fraction of keystrokes involving modifier keys |
| `clicks_per_min` | `float` | Mouse clicks per minute |
| `scroll_events_per_min` | `float` | Scroll events per minute |
| `mouse_distance` | `float` | Total mouse travel in pixels |
| `active_seconds_keyboard` | `float` | Seconds with keyboard activity in the bucket |
| `active_seconds_mouse` | `float` | Seconds with mouse activity |
| `active_seconds_any` | `float` | Seconds with any input |
| `max_idle_run_seconds` | `float` | Longest consecutive idle stretch |
| `event_density` | `float` | Active events per second of activity |

### Temporal dynamics (rolling)

| Column | Type | Description |
|--------|------|-------------|
| `keys_per_min_rolling_5` | `float` | 5-bucket rolling mean of `keys_per_min` |
| `keys_per_min_rolling_15` | `float` | 15-bucket rolling mean |
| `mouse_distance_rolling_5` | `float` | 5-bucket rolling mean of `mouse_distance` |
| `mouse_distance_rolling_15` | `float` | 15-bucket rolling mean |
| `keys_per_min_delta` | `float` | Current minus rolling-5 mean |
| `clicks_per_min_delta` | `float` | Current minus rolling-5 mean |
| `mouse_distance_delta` | `float` | Current minus rolling-5 mean |

### Calendar and session

| Column | Type | Description |
|--------|------|-------------|
| `hour_of_day` | `int` | 0--23 hour extracted from `bucket_start_ts` |
| `day_of_week` | `int` | 0 (Monday) -- 6 (Sunday) |
| `session_length_so_far` | `float` | Minutes elapsed since session start |

## validate_row

Validates a raw dict as a `FeatureRow` via Pydantic, then checks that
`schema_version` and `schema_hash` match the current contract.

```python
from taskclf.core.schema import FeatureSchemaV1

row = FeatureSchemaV1.validate_row(raw_dict)
# raises ValueError on schema_version or schema_hash mismatch
```

Returns the validated `FeatureRow` on success.

## coerce_nullable_numeric

Converts nullable numeric columns from `object` dtype (caused by
`None` values from `FeatureRow.model_dump()`) to `float64` (with
`NaN`).  Call this **before** `validate_dataframe` whenever a
DataFrame is built from model-dumped rows that may contain `None` in
numeric fields.

The function modifies the DataFrame in place and also returns it for
chaining convenience.

```python
import pandas as pd
from taskclf.core.schema import FeatureSchemaV1, coerce_nullable_numeric

df = pd.DataFrame([row.model_dump() for row in feature_rows])
coerce_nullable_numeric(df)
FeatureSchemaV1.validate_dataframe(df)
```

## validate_dataframe

Checks that a DataFrame has exactly the expected columns (no missing,
no extra) and that pandas dtype kinds are compatible with the declared
Python types.

The dtype compatibility mapping:

| Python type | Accepted pandas dtype kinds |
|-------------|----------------------------|
| `int` | `i` (signed), `u` (unsigned) |
| `float` | `f` (float), `i`, `u` (promotion safe) |
| `bool` | `b` (bool), `i`, `u` (numpy coercion) |
| `str` | `O` (object), `U` (unicode) |

Types not in this map (e.g. `datetime`, `list`) are skipped during
dtype checking.

```python
import pandas as pd
from taskclf.core.schema import FeatureSchemaV1, coerce_nullable_numeric

df = pd.DataFrame([row.model_dump() for row in feature_rows])
coerce_nullable_numeric(df)
FeatureSchemaV1.validate_dataframe(df)  # raises ValueError on mismatch
```

## See also

- [`core.types`](types.md) -- `FeatureRow` Pydantic model
- [`core.hashing`](hashing.md) -- `stable_hash` used for schema hash computation
- [`features.build`](../features/build.md) -- feature computation pipeline that produces schema-conformant rows

::: taskclf.core.schema
