# features/

Feature computation: convert normalized events into bucketed feature rows.

## Responsibilities
- Per-minute (or configurable bucket) feature generation
- Rolling-window aggregations (e.g., last 5 minutes switch count)
- Title featurization (hash trick / local tokenization) per policy

## Invariants
- Output must conform to `FeatureSchemaV1`.
- Any new feature requires updating schema definition.
- No raw titles persisted (unless explicitly opted-in locally).
