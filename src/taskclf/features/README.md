# features/

Feature computation: convert normalized events into bucketed feature rows.

## Modules
- `build.py` — Main pipeline: `build_features_from_aw_events()` groups events
  into 60s buckets, selects the dominant app, computes app-switch counts, and
  fills temporal fields. Also provides `generate_dummy_features()` for testing.
- `sessions.py` — Session boundary detection via idle-gap analysis and binary
  search lookup for `session_length_so_far` computation.

## Responsibilities
- Per-minute (or configurable bucket) feature generation
- Rolling-window aggregations (e.g., last 5 minutes switch count)
- Session boundary detection (idle-gap based)
- Title featurization (hash trick / local tokenization) per policy

## Invariants
- Output must conform to `FeatureSchemaV1`.
- Any new feature requires updating schema definition.
- No raw titles persisted (unless explicitly opted-in locally).
