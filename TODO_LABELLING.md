## TODOs (ordered)

### 0) Lock the contract ✔

1. ~~Define **core label set v1** (8 labels): `Build, Debug, Review, Write, ReadResearch, Communicate, Meet, BreakIdle`.~~ — `CoreLabel` StrEnum in `src/taskclf/core/types.py`; synced with `schema/labels_v1.json`.
2. ~~Write a **labeling guide** (1–3 bullets per label; observable rules; include "Mixed/Unknown" rule via reject threshold).~~ — `docs/guide/labels_v1.md`.
3. ~~Define **windowing spec**: bucket size (e.g., 30s/60s), session definition (idle gap minutes), and how labels attach to windows vs blocks.~~ — `docs/guide/time_spec.md` (60s buckets, 5min idle gap, block→window projection rules).

### 1) Data + schemas ✔

4. ~~Define a versioned **feature schema** (names, types, units, nullability; include schema version in every row).~~ — `FeatureSchemaV1` in `src/taskclf/core/schema.py`; canonical JSON at `schema/features_v1.json`. `FeatureRow` carries `schema_version` + `schema_hash` on every row.
5. ~~Add/confirm stable IDs: `user_id`, `session_id`, `bucket_ts`, `device_id` (optional).~~ — `FeatureRow` now has `user_id`, `session_id`, `bucket_start_ts`, `bucket_end_ts`, `device_id` (optional). Primary key is `(user_id, bucket_start_ts)`.
6. ~~Implement a **training dataset builder** that outputs:~~

   * ~~`X.parquet` (features + ids + schema_version)~~
   * ~~`y.parquet` (label per window + provenance)~~
   * ~~`splits.json` (train/val/test by time and/or by session)~~

   — `build_training_dataset()` in `src/taskclf/train/build_dataset.py`; `split_by_time()` in `src/taskclf/train/dataset.py` (3-way per-user chronological 70/15/15 + cross-user holdout). CLI: `taskclf train build-dataset`.
7. ~~Add **data validation** (hard checks): ranges, missing rates, monotonic timestamps, session boundaries, feature distributions.~~ — `validate_feature_dataframe()` in `src/taskclf/core/validation.py`; hard checks (ranges, nulls, monotonic ts, bucket_end consistency) + soft checks (distributions, class balance).

### 2) Ground-truth collection (blocks, not windows) ✔

8. ~~Build labeling UI/CLI for **time blocks**:~~

   * ~~show last N minutes summary + top predicted label~~
   * ~~user selects label + optional confidence~~
   * ~~store as `label_block(start_ts, end_ts, label, user_id, source="manual")`~~

   — CLI: `taskclf labels add-block` (Rich summary table + optional model prediction). Streamlit UI: `src/taskclf/ui/labeling.py` (queue panel, summary panel, label form, history). `LabelSpan` extended with `user_id` and `confidence` fields. `append_label_span()` in `labels/store.py` with overlap validation.
9. ~~Implement **block → window label projection**:~~

   * ~~for each window, assign label if fully inside a labeled block~~
   * ~~if overlapping multiple labels → `Mixed/Unknown` (or drop)~~

   — `project_blocks_to_windows()` in `src/taskclf/labels/projection.py`; strict containment per `time_spec.md` Section 6. `build_training_dataset()` uses strict projection. CLI: `taskclf labels project`.
10. ~~Implement **active labeling queue**:~~

* ~~enqueue windows/blocks when model confidence low or drift detected~~
* ~~limit asks per day~~

   — `ActiveLabelingQueue` in `src/taskclf/labels/queue.py`; `LabelRequest` model, `enqueue_low_confidence()`, `enqueue_drift()`, `get_pending()` (sorted by confidence, daily cap), `mark_done()`. JSON persistence at `data/processed/labels_v1/queue.json`. CLI: `taskclf labels show-queue`.

### 3) Baseline system (cold start) ✔

11. ~~Implement **rule baseline** (no ML):~~

~~* idle gap → `BreakIdle`~~
~~* browser+scroll high+keys low → `ReadResearch`~~
~~* editor/terminal+keys high+shortcuts → `Build`~~
~~* else → `Mixed/Unknown`~~

   — `classify_single_row()`, `predict_baseline()`, `run_baseline_inference()` in `src/taskclf/infer/baseline.py`; priority-ordered rules with configurable thresholds in `core/defaults.py`. CLI: `taskclf infer baseline`.

12. ~~Add metrics comparing baseline vs later ML (so you can prove improvement).~~

   — `reject_rate()`, `per_class_metrics()`, `compare_baselines()` in `src/taskclf/core/metrics.py`. CLI: `taskclf infer compare` (Rich side-by-side table + JSON report).

### 4) Global model (core labels) ✔

13. ~~Train LightGBM multiclass on core labels:~~

~~* time-based split (avoid leakage): train on earlier days, validate on later days~~
~~* per-user stratification check (don't let one user dominate)~~

   — `train_lgbm()` in `src/taskclf/train/lgbm.py` with `class_weight` param; `split_by_time()` in `train/dataset.py` (chronological per-user 70/15/15); `user_stratification_report()` in `core/metrics.py` flags dominant users. CLI: `taskclf train lgbm --class-weight balanced`.

14. ~~Add class-imbalance handling:~~

~~* class weights or focal-ish sampling (simpler: weights)~~

   — `compute_sample_weights()` in `src/taskclf/train/lgbm.py`; inverse-frequency per-sample weights passed to `lgb.Dataset`. Method recorded in `metadata.json` as `class_weight_method`.

15. ~~Implement evaluation:~~

~~* overall macro-F1~~
~~* per-class precision/recall~~
~~* confusion matrix~~
~~* per-user metrics~~
~~* calibration curves (reliability)~~

   — `evaluate_model()` in `src/taskclf/train/evaluate.py` returns `EvaluationReport` (macro-F1, weighted-F1, per-class P/R/F1, confusion matrix, per-user macro-F1, calibration curves, seen/unseen user splits, acceptance checks). `write_evaluation_artifacts()` writes `evaluation.json`, `calibration.json`, `confusion_matrix.csv`, `calibration.png`. CLI: `taskclf train evaluate`.

### 5) Reject option (Mixed/Unknown by threshold) ✔

16. ~~Pick reject strategy:~~

~~* if `max_proba < p_reject` → `Mixed/Unknown`~~

   — `predict_labels()` and `run_batch_inference()` in `src/taskclf/infer/batch.py` accept `reject_threshold` parameter; `OnlinePredictor` in `src/taskclf/infer/online.py` applies threshold per bucket. `DEFAULT_REJECT_THRESHOLD = 0.55` in `core/defaults.py`. `predict_proba()` extracted as shared function. CLI: `taskclf infer batch --reject-threshold`, `taskclf infer online --reject-threshold`.

17. ~~Tune `p_reject` on validation set to trade off coverage vs accuracy.~~

   — `tune_reject_threshold()` in `src/taskclf/train/evaluate.py` sweeps thresholds (0.10–0.95), returns `RejectTuningResult` with optimal threshold maximising accuracy within acceptance bounds (5–30% reject rate). CLI: `taskclf train tune-reject`.

18. ~~Log "rejected" rate per user/day; treat spikes as drift signals.~~

   — `reject_rate_by_group()` in `src/taskclf/core/metrics.py` groups by `(user_id, date)`, flags groups exceeding `spike_multiplier × global_reject_rate` as drift signals. `reject_threshold` recorded in `ModelMetadata` for reproducibility.

### 6) Personalization (without label explosion)

19. Add `user_id` as categorical feature in LightGBM (or hashed categorical).
20. Implement **per-user probability calibration**:

* store calibrator per user (start with global calibrator until enough labels)
* choose method: temperature scaling (simple) or isotonic (more flexible)

21. Define "enough data" thresholds:

* e.g., enable per-user calibrator after ≥200 labeled windows across ≥3 days.

### 7) User-specific taxonomy mapping layer

22. Design mapping config format:

* core → user buckets (many-to-one)
* optional weights / priorities

23. Implement mapping resolver:

* predicted core label + probs → mapped label + aggregated probs
* handle unmapped core labels (fallback bucket)

24. Add UI for users to edit mappings (and version them).

### 8) Online inference pipeline

25. Implement feature extraction pipeline (stream → window rows) with stable ordering.
26. Implement inference service/module:

* loads global model + user calibrator + user mapping
* outputs: `core_label`, `core_probs`, `mapped_label`, `mapped_probs`, `is_rejected`, `confidence`

27. Persist predictions for time tracking:

* write per-window predictions
* aggregate to blocks (merge adjacent same-label windows with hysteresis)

### 9) Aggregation for time tracking (the "product" output)

28. Implement smoothing / hysteresis:

* prevent label flapping (minimum block length, majority vote over k windows)

29. Implement daily summaries:

* totals by mapped label
* totals by core label
* context switching stats (from `app_switch_count_last_5m`)

30. Export formats:

* CSV/Parquet + JSON summary (stable schema)

### 10) Drift + quality monitoring

31. Implement telemetry:

* feature missingness, distribution shifts, reject rate, confidence stats

32. Drift detection triggers:

* PSI/KS on features per user
* sudden increase in reject rate or entropy

33. Auto-create labeling tasks when drift triggers.

### 11) Retraining workflow

34. Define retrain cadence:

* global retrain weekly/monthly
* per-user calibrator update daily/weekly

35. Build a reproducible training pipeline:

* config in git
* dataset snapshot hashes
* model artifact versioning

36. Add regression tests:

* schema compat
* "no worse than baseline" gates
* invariant checks (e.g., BreakIdle precision >= X)

### 12) Feature upgrades (optional but high leverage)

37. Add app identity features: `app_id` (bundle/process), not just boolean flags.
38. Add browser URL/domain category (privacy-preserving: eTLD+1 or hashed category).
39. Add window-title clustering (you already have `window_title_hash` concept): frequency stats, not raw titles.
40. Add temporal dynamics features:

* rolling means over 5m/15m
* deltas vs last window
* counts of switches in last N windows

---

If you want this to be implementable fast: do **0 → 1 → 2 → 3 → 4 → 5 → 7 → 8 → 9** first, then add **6/10/11/12** iteratively.

Sources:

* LightGBM docs (multiclass, categorical): [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
* Temperature scaling (calibration): [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
* Isotonic regression (calibration concept): [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html)
