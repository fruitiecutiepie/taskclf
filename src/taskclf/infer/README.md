# infer/

Inference code for online and batch prediction.

## Modules
- `batch.py` -- Batch inference: predict, smooth, and segmentize over a feature DataFrame
- `smooth.py` -- Rolling majority smoothing and segmentization
- `online.py` -- Real-time prediction loop: polls ActivityWatch, predicts, smooths, and reports

## Responsibilities
- Load model bundle + validate schema hash
- Predict per bucket
- Smooth predictions into stable labels
- Emit minute-level predictions + merged segments
- Online mode: poll AW REST API, build features, predict, write running outputs

## Invariants
- Refuse to run if feature schema mismatch.
- Online inference must never retrain.
- Output segments are derived artifacts (rebuildable).
