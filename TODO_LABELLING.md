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

### 6) Personalization (without label explosion) ✔

19. ~~Add `user_id` as categorical feature in LightGBM (or hashed categorical).~~

   — `user_id` appended to `FEATURE_COLUMNS` and `CATEGORICAL_COLUMNS` in `src/taskclf/train/lgbm.py`. LightGBM treats it as a native categorical; unknown users at inference time fall back to code `-1` via existing `encode_categoricals()`.

20. ~~Implement **per-user probability calibration**:~~

~~* store calibrator per user (start with global calibrator until enough labels)~~
~~* choose method: temperature scaling (simple) or isotonic (more flexible)~~

   — `TemperatureCalibrator` and `IsotonicCalibrator` in `src/taskclf/infer/calibration.py`. `CalibratorStore` holds a global calibrator plus per-user calibrators, with `get_calibrator(user_id)` fallback. `fit_temperature_calibrator()` and `fit_isotonic_calibrator()` in `src/taskclf/train/calibrate.py`. `fit_calibrator_store()` orchestrates the full pipeline. CLI: `taskclf train calibrate --method temperature|isotonic`. Store serialized as directory (`store.json` + `global.json` + `users/<uid>.json`). Integrated into `run_batch_inference()` and `OnlinePredictor` via `calibrator_store` param. CLI: `taskclf infer batch --calibrator-store`, `taskclf infer online --calibrator-store`.

21. ~~Define "enough data" thresholds:~~

~~* e.g., enable per-user calibrator after ≥200 labeled windows across ≥3 days.~~

   — `check_personalization_eligible()` in `src/taskclf/train/calibrate.py` gates per-user calibration. Thresholds from `docs/guide/acceptance.md` §8: ≥200 labeled windows, ≥3 distinct days, ≥3 distinct core labels. Constants in `src/taskclf/core/defaults.py` (`DEFAULT_MIN_LABELED_WINDOWS`, `DEFAULT_MIN_LABELED_DAYS`, `DEFAULT_MIN_DISTINCT_LABELS`). Overridable via CLI flags `--min-windows`, `--min-days`, `--min-labels`.

### 7) User-specific taxonomy mapping layer ✔

22. ~~Design mapping config format:~~

~~* core → user buckets (many-to-one)~~
~~* optional weights / priorities~~

   — `TaxonomyConfig`, `TaxonomyBucket`, `TaxonomyAdvanced` Pydantic models in `src/taskclf/infer/taxonomy.py`; YAML format matching `configs/user_taxonomy_example.yaml`. Supports sum/max probability aggregation, per-label reweighting, hex colors, and user-specific reject label. Validation: unique bucket names, valid `CoreLabel` members, valid hex colors, positive reweights.

23. ~~Implement mapping resolver:~~

~~* predicted core label + probs → mapped label + aggregated probs~~
~~* handle unmapped core labels (fallback bucket)~~

   — `TaxonomyResolver` class in `src/taskclf/infer/taxonomy.py` with `resolve()` (single row) and `resolve_batch()` (vectorised). Produces `TaxonomyResult` (mapped_label + mapped_probs summing to 1.0). Unmapped core labels assigned to automatic "Other" fallback bucket. Integrated into `run_batch_inference()` (optional `taxonomy` param) and `OnlinePredictor` (optional `taxonomy` param). CLI: `taskclf infer batch --taxonomy`, `taskclf infer online --taxonomy`.

24. ~~Add UI for users to edit mappings (and version them).~~

   — CLI taxonomy management: `taskclf taxonomy validate` (validates YAML), `taskclf taxonomy show` (Rich table display), `taskclf taxonomy init` (generates default identity-mapping YAML). YAML I/O via `load_taxonomy()` / `save_taxonomy()`. Versioned via `version` field in config. Streamlit UI deferred to future iteration.

### 8) Online inference pipeline ✔

25. ~~Implement feature extraction pipeline (stream → window rows) with stable ordering.~~

   — `build_features_from_aw_events()` in `src/taskclf/features/build.py` converts normalised events to per-bucket `FeatureRow` instances with deterministic ordering. Online loop in `run_online_loop()` passes `session_start` across poll cycles for session continuity.

26. ~~Implement inference service/module:~~

~~* loads global model + user calibrator + user mapping~~
~~* outputs: `core_label`, `core_probs`, `mapped_label`, `mapped_probs`, `is_rejected`, `confidence`~~

   — `OnlinePredictor.predict_bucket()` returns `WindowPrediction` (Pydantic model in `src/taskclf/infer/prediction.py`) matching model_io section 6: `user_id`, `bucket_start_ts`, `core_label_id`, `core_label_name`, `core_probs`, `confidence`, `is_rejected`, `mapped_label_name`, `mapped_probs`, `model_version`, `schema_version`, `label_version`. Accepts optional `Calibrator` (protocol in `src/taskclf/infer/calibration.py`; identity + temperature scaling implementations) and `TaxonomyConfig`. Calibrator hook applied between raw proba and reject decision. CLI: `--calibrator` flag on `taskclf infer online`.

27. ~~Persist predictions for time tracking:~~

~~* write per-window predictions~~
~~* aggregate to blocks (merge adjacent same-label windows with hysteresis)~~

   — `_append_prediction_csv()` writes full `WindowPrediction` fields (core_label, core_probs, confidence, is_rejected, mapped_label, mapped_probs, model_version). `write_predictions_csv()` in batch.py also accepts `core_probs`. `merge_short_segments()` in `src/taskclf/infer/smooth.py` absorbs segments shorter than `MIN_BLOCK_DURATION_SECONDS` (180s) into neighbours. Integrated into `OnlinePredictor.get_segments()` and `run_batch_inference()`.

### 9) Aggregation for time tracking (the "product" output) ✔

28. ~~Implement smoothing / hysteresis:~~

~~* prevent label flapping (minimum block length, majority vote over k windows)~~

   — `rolling_majority()` + `merge_short_segments()` in `src/taskclf/infer/smooth.py`; `flap_rate()` metric (label changes / total windows) for acceptance verification (raw ≤ 0.25, smoothed ≤ 0.15). Integrated into batch (`infer batch` prints flap rates) and online (on shutdown report).

29. ~~Implement daily summaries:~~

~~* totals by mapped label~~
~~* totals by core label~~
~~* context switching stats (from `app_switch_count_last_5m`)~~

   — `DailyReport` in `src/taskclf/report/daily.py` with `core_breakdown` (minutes per core label), `mapped_breakdown` (minutes per taxonomy bucket), `ContextSwitchStats` (mean/median/max/total from `app_switch_count_last_5m`), `flap_rate_raw`, `flap_rate_smoothed`. `build_daily_report()` accepts segments + optional per-bucket predictions and feature data. CLI: `taskclf report daily` with `--predictions-file`, `--features-dir`, `--format` flags.

30. ~~Export formats:~~

~~* CSV/Parquet + JSON summary (stable schema)~~

   — `export_report_json()`, `export_report_csv()`, `export_report_parquet()` in `src/taskclf/report/export.py`. CSV/Parquet use flat schema: one row per label (`date`, `label_type`, `label`, `minutes`). All formats pass sensitive-field blocklist. CLI: `taskclf report daily --format json|csv|parquet|all`.

### 10) Drift + quality monitoring ✔

31. ~~Implement telemetry:~~

~~* feature missingness, distribution shifts, reject rate, confidence stats~~

   — `TelemetrySnapshot` and `compute_telemetry()` in `src/taskclf/core/telemetry.py`; `TelemetryStore` (append-only JSONL per user/global). Computes feature missingness rates, confidence stats (mean/median/p5/p95/std), reject rate, mean entropy, class distribution. CLI: `taskclf monitor telemetry`. Integrated into `run_online_loop()` shutdown path.

32. ~~Drift detection triggers:~~

~~* PSI/KS on features per user~~
~~* sudden increase in reject rate or entropy~~

   — `compute_psi()`, `compute_ks()`, `feature_drift_report()` in `src/taskclf/core/drift.py` (PSI with quantile binning, KS via `scipy.stats.ks_2samp`). `detect_reject_rate_increase()` flags absolute increase >= 10%. `detect_entropy_spike()` flags mean entropy > multiplier × reference. `detect_class_shift()` flags class proportion changes > 15%. `run_drift_check()` in `src/taskclf/infer/monitor.py` orchestrates all checks, returns `DriftReport` with `DriftAlert` list. CLI: `taskclf monitor drift-check`. Thresholds from `docs/guide/acceptance.md` §7.

33. ~~Auto-create labeling tasks when drift triggers.~~

   — `auto_enqueue_drift_labels()` in `src/taskclf/infer/monitor.py`; selects lowest-confidence buckets from current window and enqueues via `ActiveLabelingQueue.enqueue_drift()`. Configurable limit (default 50). Integrated into `taskclf monitor drift-check --auto-label`.

### 11) Retraining workflow ✔

34. ~~Define retrain cadence:~~

~~* global retrain weekly/monthly~~
~~* per-user calibrator update daily/weekly~~

   — `RetrainConfig` in `src/taskclf/train/retrain.py` with `global_retrain_cadence_days` (default 7) and `calibrator_update_cadence_days` (default 7). `check_retrain_due()` and `check_calibrator_update_due()` compare model/calibrator `created_at` against cadence. Config in `configs/retrain.yaml`. CLI: `taskclf train check-retrain`.

35. ~~Build a reproducible training pipeline:~~

~~* config in git~~
~~* dataset snapshot hashes~~
~~* model artifact versioning~~

   — `run_retrain_pipeline()` in `src/taskclf/train/retrain.py` orchestrates: build dataset → compute `dataset_hash` (SHA-256) → train challenger → evaluate → regression gates → promote. `dataset_hash` is a required field on `ModelMetadata` in `src/taskclf/core/model_io.py`. Config versioned at `configs/retrain.yaml`. `DatasetSnapshot` records hash, row count, date range, user count, class distribution. CLI: `taskclf train retrain --config configs/retrain.yaml`.

36. ~~Add regression tests:~~

~~* schema compat~~
~~* "no worse than baseline" gates~~
~~* invariant checks (e.g., BreakIdle precision >= X)~~

   — `check_regression_gates()` in `src/taskclf/train/retrain.py` runs four gates: (1) macro-F1 no regression within `regression_tolerance` (default 0.02), (2) BreakIdle precision >= 0.95, (3) no class precision < 0.50, (4) all acceptance checks pass. Schema compat enforced by existing `load_model_bundle()` validation. 15 tests in `tests/test_retrain.py` covering hash determinism, cadence checks, all gate pass/fail paths, config roundtrip, and full pipeline integration.

### 12) Feature upgrades (optional but high leverage) ✔

37. ~~Add app identity features: `app_id` (bundle/process), not just boolean flags.~~ — `app_id` (reverse-domain identifier) and `app_category` (semantic category) already in `FeatureSchemaV1`, `FEATURE_COLUMNS`, and `CATEGORICAL_COLUMNS`. LightGBM treats both as native categoricals.

38. ~~Add browser URL/domain category (privacy-preserving: eTLD+1 or hashed category).~~ — `domain_category` field in `FeatureRow` and schema. `classify_domain()` in `src/taskclf/features/domain.py` maps eTLD+1 domains to privacy-safe categories (search, docs, social, video, code_hosting, news, email_web, productivity, chat, design, other, unknown, non_browser). Non-browser apps get `"non_browser"`. Added to `FEATURE_COLUMNS` and `CATEGORICAL_COLUMNS`.

39. ~~Add window-title clustering (you already have `window_title_hash` concept): frequency stats, not raw titles.~~ — `window_title_bucket` (hash-trick bucket 0–255 via `title_hash_bucket()` in `features/text.py`) and `title_repeat_count_session` (cumulative count of each title hash within the current session). Both non-nullable, computed in `build_features_from_aw_events()`.

40. ~~Add temporal dynamics features:~~

~~* rolling means over 5m/15m~~
~~* deltas vs last window~~
~~* counts of switches in last N windows~~

   — `DynamicsTracker` in `src/taskclf/features/dynamics.py`. Rolling means: `keys_per_min_rolling_5`, `keys_per_min_rolling_15`, `mouse_distance_rolling_5`, `mouse_distance_rolling_15`. Deltas: `keys_per_min_delta`, `clicks_per_min_delta`, `mouse_distance_delta`. Extended switch count: `app_switch_count_last_15m` (15-minute window via existing `app_switch_count_in_window()`). All 11 new features added to schema, FeatureRow, FEATURE_COLUMNS, and validation. 42 new tests across `test_features_domain.py`, `test_features_dynamics.py`, and `test_features_from_aw.py`.

---

If you want this to be implementable fast: do **0 → 1 → 2 → 3 → 4 → 5 → 7 → 8 → 9** first, then add **6/10/11/12** iteratively.

Sources:

* LightGBM docs (multiclass, categorical): [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
* Temperature scaling (calibration): [https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
* Isotonic regression (calibration concept): [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html)
