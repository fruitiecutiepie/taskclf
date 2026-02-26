# train/

Training pipeline code.

## Modules
- `build_dataset.py` — Dataset construction: join features with labels,
  exclude unlabeled/ambiguous rows, split, and persist X/y/splits arrays.
- `dataset.py` — Split-by-time logic (chronological train/val/test) with
  optional holdout-user partitioning.
- `lgbm.py` — LightGBM multiclass training with class-weight balancing.
- `evaluate.py` — Full evaluation framework: per-class/per-user metrics,
  acceptance checks, reject threshold tuning, stratification warnings.
- `calibrate.py` — Per-user probability calibrator fitting (temperature
  scaling or isotonic regression) with eligibility checks.
- `retrain.py` — Champion/challenger retraining pipeline: cadence checks,
  dataset hashing, regression gates, model promotion.

## Responsibilities
- Join features with label spans
- Split strategy (by day/week to avoid leakage, optional user holdout)
- Train baseline model (LightGBM first)
- Evaluate with acceptance checks and per-user stratification
- Fit per-user probability calibrators
- Automated retraining with regression gating and promotion
- Save model bundles with metadata and metrics

## Invariants
- Splits must be time-aware (no random shuffles by default).
- Training writes a new run directory every time.
- Model bundle includes schema hash and label set.
- Retraining never overwrites the champion model without passing regression gates.
