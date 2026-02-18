# core/

Stable, test-heavy code. This is where invariants live.

## Responsibilities
- Type definitions: Event, FeatureRow, LabelSpan
- Feature schema versioning + schema hashing
- Bucketization and sessionization helpers
- Storage primitives (parquet/duckdb IO)
- Model artifact IO (save/load + metadata checks)
- Metrics and evaluation utilities

## Invariants (enforced)
- Feature rows include schema version + schema hash.
- Prohibit sensitive fields (raw keystrokes, raw titles) in persisted data.
- Fail fast on schema mismatch.
