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

## Session Tracking (online)
The online loop tracks session state across poll cycles.  A new session
starts when the gap between the last observed event and the earliest event
in the current poll exceeds `idle_gap_seconds` (default 300 s / 5 min).
This ensures `session_length_so_far` accumulates correctly over continuous
activity instead of resetting each poll.

## Invariants
- Refuse to run if feature schema mismatch.
- Online inference must never retrain.
- Output segments are derived artifacts (rebuildable).
