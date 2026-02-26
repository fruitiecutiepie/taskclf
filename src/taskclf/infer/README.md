# infer/

Inference code for online, batch, and baseline prediction.

## Modules
- `batch.py` — Batch inference: predict, smooth, and segmentize over a feature DataFrame.
  Writes predictions CSV and segments JSON.
- `online.py` — Real-time prediction loop: polls ActivityWatch, predicts, smooths, and reports.
- `smooth.py` — Rolling majority smoothing, segmentization, and flap-rate computation.
- `baseline.py` — Rule-based inference using app categories (no ML model required).
- `prediction.py` — Core predict logic shared by batch and online paths.
- `calibration.py` — Calibrator store load/save and probability calibration at inference time.
- `taxonomy.py` — User taxonomy mapping: remap core labels to personal buckets via YAML config.
- `monitor.py` — Drift detection (PSI, KS, entropy, class shift, reject-rate),
  auto-enqueue labeling tasks on drift.

## Responsibilities
- Load model bundle + validate schema hash
- Predict per bucket (ML or rule-based baseline)
- Apply per-user probability calibration
- Smooth predictions into stable labels
- Map core labels to user taxonomy buckets
- Emit minute-level predictions + merged segments
- Online mode: poll AW REST API, build features, predict, write running outputs
- Compare baseline vs ML model on labeled data
- Detect feature and prediction drift, surface labeling tasks

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
