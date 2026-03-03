# features.windows

Rolling-window aggregations over event streams.  Extracted from
`features.build` so the windowed metric logic can be tested and
reused independently.

## app_switch_count_in_window

Counts the number of unique-app switches within a look-back window
ending at a given bucket timestamp.

The window spans
`[bucket_ts - window_minutes, bucket_ts + bucket_seconds)`.
The return value is `max(0, unique_apps - 1)` -- one app means zero
switches, two apps means one switch, and so on.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `events` | `Sequence[Event]` | -- | Chronologically sorted events |
| `bucket_ts` | `datetime` | -- | Aligned start of the current time bucket |
| `window_minutes` | `int` | `DEFAULT_APP_SWITCH_WINDOW_MINUTES` (5) | How many minutes to look back |
| `bucket_seconds` | `int` | `DEFAULT_BUCKET_SECONDS` (60) | Width of the current bucket |

```python
from taskclf.features.windows import app_switch_count_in_window

switches = app_switch_count_in_window(sorted_events, bucket_ts)
# 0 if only one app was used in the window
```

Events before `window_start` are skipped; iteration stops at
`window_end`, so pre-sorted input is required for correctness.

## compute_rolling_app_switches

Batch helper that calls `app_switch_count_in_window` for every bucket
in a sorted list of bucket timestamps.  Returns a list of switch
counts in the same order, one per bucket.

```python
from taskclf.features.windows import compute_rolling_app_switches

counts = compute_rolling_app_switches(sorted_events, sorted_buckets)
# len(counts) == len(sorted_buckets)
```

## See also

- [`features.build`](build.md) -- main pipeline that calls these functions
- [`core.defaults`](../core/defaults.md) -- `DEFAULT_APP_SWITCH_WINDOW_MINUTES` and `DEFAULT_BUCKET_SECONDS`

::: taskclf.features.windows
