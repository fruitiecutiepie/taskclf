# features/

Feature computation: convert normalized events into bucketed feature rows.

## Modules
- `build.py` — Main pipeline: `build_features_from_aw_events()` groups events
  into 60s buckets, selects the dominant app, computes app-switch counts, and
  fills temporal fields.  Passes through `app_id` and `app_category` from
  the dominant app for use as categorical training features.  Accepts optional
  `input_events` from `aw-watcher-input` to populate keyboard/mouse features
  (`keys_per_min`, `clicks_per_min`, `scroll_events_per_min`, `mouse_distance`)
  and activity occupancy metrics (`active_seconds_keyboard`, `active_seconds_mouse`,
  `active_seconds_any`, `max_idle_run_seconds`, `event_density`).
  Also provides `generate_dummy_features()` for testing.
- `sessions.py` — Session boundary detection via idle-gap analysis and binary
  search lookup for `session_length_so_far` computation.
- `domain.py` — Reverse-domain app-id feature extraction and normalization.
- `dynamics.py` — Temporal dynamics features: rate-of-change and acceleration
  for input metrics across rolling windows.
- `text.py` — Title-hash feature extraction (hash trick / local tokenization).
- `windows.py` — Rolling window aggregations over feature columns.

## Responsibilities
- Per-minute (or configurable bucket) feature generation
- Rolling-window aggregations (e.g., last 5 minutes switch count)
- Focus metrics (`app_foreground_time_ratio`, `app_change_count`)
- Keyboard/mouse input aggregation from `aw-watcher-input` events
- Activity occupancy (active seconds, idle runs, event density)
- Session boundary detection (idle-gap based)
- Temporal dynamics (rate-of-change, acceleration)
- Title featurization (hash trick / local tokenization) per policy

## Invariants
- Output must conform to `FeatureSchemaV1`.
- Any new feature requires updating schema definition.
- No raw titles persisted (unless explicitly opted-in locally).
