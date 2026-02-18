# train/

Training pipeline code.

## Responsibilities
- Join features with label spans
- Split strategy (by day/week to avoid leakage)
- Train baseline model (LightGBM first)
- Save model bundles with metadata and metrics
- Optional calibration

## Invariants
- Splits must be time-aware (no random shuffles by default).
- Training writes a new run directory every time.
- Model bundle includes schema hash and label set.
