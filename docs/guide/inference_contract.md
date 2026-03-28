# Inference Contract

Version: 1.0
Status: Stable
Last Updated: 2026-03-28

This document defines the canonical inference pipeline order for taskclf
and maps each stage to its implementation across the three runtime paths.

---

## 1. Canonical Pipeline Order

Every inference path must execute these stages in this order:

```
features → encode categoricals → impute missing → predict probabilities
→ calibrate → reject → smooth/aggregate → taxonomy map → UI label
```

| # | Stage                  | Input                        | Output                          |
|---|------------------------|------------------------------|---------------------------------|
| 1 | Build features         | Raw events                   | `FeatureRow` / feature DataFrame |
| 2 | Encode categoricals    | String columns               | Integer-encoded columns          |
| 3 | Impute missing values  | Encoded matrix (with gaps)   | Dense numeric matrix             |
| 4 | Predict probabilities  | Dense matrix                 | `(N, n_classes)` probability matrix |
| 5 | Calibrate              | Raw probabilities            | Calibrated probabilities         |
| 6 | Reject                 | Calibrated confidence        | Rejection flags + `Mixed/Unknown` labels |
| 7 | Smooth / aggregate     | Per-bucket labels            | Rolling-majority-smoothed labels |
| 8 | Taxonomy map           | Core labels + calibrated probs | Mapped labels + mapped probs   |
| 9 | UI label               | Mapped or smoothed label     | Final user-facing string         |

---

## 2. Runtime Paths

### 2.1 Batch

Entry point: `run_batch_inference()` in `src/taskclf/infer/batch.py`.

| Stage               | Implementation                                                        |
|----------------------|-----------------------------------------------------------------------|
| Build features       | Caller supplies a pre-built `features_df`.                           |
| Encode categoricals  | `predict_proba()` → `encode_categoricals()` from `train/lgbm.py`.   |
| Impute missing       | `predict_proba()` → `fillna(0)`.                                    |
| Predict probabilities| `predict_proba()` → `model.predict(x)`.                             |
| Calibrate            | `CalibratorStore.calibrate_batch()` (per-user) or single `Calibrator.calibrate()`. |
| Reject               | `max(proba) < reject_threshold` → label becomes `Mixed/Unknown`.    |
| Smooth / aggregate   | `rolling_majority()` from `infer/smooth.py`, then `segmentize()` + `merge_short_segments()`. |
| Taxonomy map         | `TaxonomyResolver.resolve_batch()` from `infer/taxonomy.py`.        |
| UI label             | `BatchInferenceResult.mapped_labels` (or `smoothed_labels` if no taxonomy). |

Helper functions `predict_proba()` and `predict_labels()` cover only
stages 2–4 (encode, impute, predict). They do not calibrate, reject,
smooth, or map. They are used for evaluation and calibrator fitting,
not end-user inference.

### 2.2 Online

Entry point: `OnlinePredictor.predict_bucket()` in `src/taskclf/infer/online.py`.

| Stage               | Implementation                                                        |
|----------------------|-----------------------------------------------------------------------|
| Build features       | `run_online_loop()` calls `build_features_from_aw_events()`.        |
| Encode categoricals  | `_encode_value()` per column (categorical → `LabelEncoder`, unknown → `-1`). |
| Impute missing       | `_encode_value()` returns `float("nan")` for missing numerics.      |
| Predict probabilities| `model.predict(x)` on the single-row array.                         |
| Calibrate            | `CalibratorStore.get_calibrator(row.user_id)` or fallback `Calibrator.calibrate()`. |
| Reject               | `confidence < reject_threshold` → label becomes `Mixed/Unknown`.    |
| Smooth / aggregate   | `rolling_majority()` over an internal deque buffer.                  |
| Taxonomy map         | `TaxonomyResolver.resolve()` per bucket.                             |
| UI label             | `WindowPrediction.mapped_label_name` (or `smoothed_label` if no taxonomy). |

### 2.3 Tray

Entry point: `_LabelSuggester.suggest()` in `src/taskclf/ui/tray.py`.

Delegates entirely to the online path:

1. Fetches AW events for the requested time window.
2. Builds features via `build_features_from_aw_events()`.
3. Calls `OnlinePredictor.predict_bucket()` on the last row.
4. Returns `(prediction.core_label_name, prediction.confidence)`.

The tray path inherits all online-path stages. Its return value
currently exposes `core_label_name`, not `mapped_label_name`.

---

## 3. Known Deviations

These deviations from the canonical order are documented here and
addressed in Phase 1.

### 3.1 Imputation mismatch (batch vs online)

- **Batch** `predict_proba()` uses `fillna(0)` — missing numerics become zero.
- **Online** `_encode_value()` returns `float("nan")` — missing numerics stay NaN.
- **Training** `prepare_xy()` in `train/lgbm.py` uses `fillna(0)`.

Batch matches training; online does not. LightGBM handles NaN natively
(routes to best split), so the model tolerates this, but predictions may
differ for identical inputs depending on path.

### 3.2 predict_proba / predict_labels skip calibration

`predict_proba()` and `predict_labels()` execute only encode → impute →
predict. They do not calibrate. This is intentional for evaluation and
calibrator fitting, but callers must not treat their output as
production-grade calibrated predictions.

### 3.3 Taxonomy inputs are pre-smooth

In both batch and online, taxonomy mapping is sequenced after smoothing
in the code flow. However, `TaxonomyResolver.resolve()` /
`resolve_batch()` receive the per-bucket argmax index and calibrated
probabilities — not the smoothed label. Taxonomy therefore maps from
the raw (pre-smooth) prediction, not the smoothed one.

### 3.4 Tray suggest returns core_label_name

`_LabelSuggester.suggest()` returns `core_label_name` and `confidence`.
It does not return `mapped_label_name`, so taxonomy configuration is
invisible to the tray UI return value.

---

## 4. References

- [Batch API reference](../api/infer/batch.md)
- [Online API reference](../api/infer/online.md)
- [Calibration API reference](../api/infer/calibration.md)
- [Smoothing API reference](../api/infer/smooth.md)
- [Taxonomy API reference](../api/infer/taxonomy.md)
- [Personalization guide](personalization.md)
