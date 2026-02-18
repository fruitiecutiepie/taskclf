# infer/

Inference code for online and batch prediction.

## Responsibilities
- Load model bundle + validate schema hash
- Predict per bucket
- Smooth predictions into stable labels
- Emit minute-level predictions + merged segments

## Invariants
- Refuse to run if feature schema mismatch.
- Online inference should not retrain.
- Output segments are derived artifacts (rebuildable).
