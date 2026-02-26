# core/

Stable, test-heavy code. This is where invariants live.

## Responsibilities
- Type definitions: Event, FeatureRow, LabelSpan (`types.py`)
- Feature schema versioning + schema hashing (`schema.py`)
- Bucketization and sessionization helpers (`time.py`)
- Storage primitives — parquet/duckdb IO (`store.py`)
- Model artifact IO — save/load + metadata checks (`model_io.py`)
- Metrics and evaluation utilities (`metrics.py`)
- Log sanitization to prevent sensitive data leakage (`logging.py`)
- Title hashing with configurable salt (`hashing.py`)
- Input validation and contract enforcement (`validation.py`)
- Centralized default constants (`defaults.py`)
- Feature and prediction drift detection — PSI, KS tests (`drift.py`)
- Telemetry snapshot computation and storage (`telemetry.py`)

## Invariants (enforced)
- Feature rows include schema version + schema hash.
- Prohibit sensitive fields (raw keystrokes, raw titles) in persisted data.
- Fail fast on schema mismatch.
