## TODOs (ordered)

### 0) Lock the contract ✔

1. ~~Define **core label set v1** (8 labels): `Build, Debug, Review, Write, ReadResearch, Communicate, Meet, BreakIdle`.~~ — `CoreLabel` StrEnum in `src/taskclf/core/types.py`; synced with `schema/labels_v1.json`.
2. ~~Write a **labeling guide** (1–3 bullets per label; observable rules; include “Mixed/Unknown” rule via reject threshold).~~ — `docs/guide/labels_v1.md`.
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

### 2) Ground-truth collection (blocks, not windows)

8. Build labeling UI/CLI for **time blocks**:

   * show last N minutes summary + top predicted label
   * user selects label + optional confidence
   * store as `label_block(start_ts, end_ts, label, user_id, source="manual")`
9. Implement **block → window label projection**:

   * for each window, assign label if fully inside a labeled block
   * if overlapping multiple labels → `Mixed/Unknown` (or drop)
10. Implement **active labeling queue**:

* enqueue windows/blocks when model confidence low or drift detected
* limit asks per day

### 3) Baseline system (cold start)

11. Implement **rule baseline** (no ML):

* idle gap → `BreakIdle`
* browser+scroll high+keys low → `ReadResearch`
* editor/terminal+keys high+shortcuts → `Build`
* else → `Mixed/Unknown`

12. Add metrics comparing baseline vs later ML (so you can prove improvement).

### 4) Global model (core labels)

13. Train LightGBM multiclass on core labels:

* time-based split (avoid leakage): train on earlier days, validate on later days
* per-user stratification check (don’t let one user dominate)

14. Add class-imbalance handling:

* class weights or focal-ish sampling (simpler: weights)

15. Implement evaluation:

* overall macro-F1
* per-class precision/recall
* confusion matrix
* per-user metrics
* calibration curves (reliability)

### 5) Reject option (Mixed/Unknown by threshold)

16. Pick reject strategy:

* if `max_proba < p_reject` → `Mixed/Unknown`

17. Tune `p_reject` on validation set to trade off coverage vs accuracy.
18. Log “rejected” rate per user/day; treat spikes as drift signals.

### 6) Personalization (without label explosion)

19. Add `user_id` as categorical feature in LightGBM (or hashed categorical).
20. Implement **per-user probability calibration**:

* store calibrator per user (start with global calibrator until enough labels)
* choose method: temperature scaling (simple) or isotonic (more flexible)

21. Define “enough data” thresholds:

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

### 9) Aggregation for time tracking (the “product” output)

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
* “no worse than baseline” gates
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
