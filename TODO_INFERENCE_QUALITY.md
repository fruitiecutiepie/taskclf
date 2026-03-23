# TODO — Model Inference Quality

Context dump and ordered plan for improving inference quality in `taskclf`.

This file is intended to be the working brief before code changes. It captures:

- what the current LightGBM inference stack actually does
- where train/serve mismatches exist
- which quality risks matter most
- the order of implementation work
- how to measure whether changes helped

---

## Goals

- Improve real-world inference quality for both:
  - continuous online inference
  - tray/UI label suggestions
- Preserve repo invariants:
  - privacy-safe features only
  - schema hash compatibility
  - deterministic model bundles
  - time-aware evaluation
- Optimize the system we have now:
  - LightGBM on fixed-width tabular features
  - not a sequence model
  - quality comes from better features, better train/serve parity, better calibration, and better aggregation

---

## Non-Negotiables

- Do not store raw keystrokes.
- Do not store raw window titles by default.
- Do not break `FeatureSchemaV1` or schema gating casually.
- Do not optimize only offline metrics if live inference behavior differs.
- Do not change model behavior without updating evaluation and docs together.

---

## Current Pipeline Summary

### 1) Training

Primary files:

- `src/taskclf/train/lgbm.py`
- `src/taskclf/train/build_dataset.py`
- `src/taskclf/train/dataset.py`
- `src/taskclf/core/model_io.py`

Current behavior:

- Model is LightGBM multiclass.
- `FEATURE_COLUMNS` in `src/taskclf/train/lgbm.py` define the exact model input contract.
- Categorical features are:
  - `app_id`
  - `app_category`
  - `domain_category`
  - `user_id`
- Categoricals are label-encoded with unknown values mapped to `-1` at inference time.
- Numeric missing values are filled with `0` during training and batch inference.
- Training uses chronological per-user splits via `split_by_time()` in `src/taskclf/train/dataset.py`.
- Optional full-user holdout is supported for cold-start evaluation.
- Sample weighting defaults to inverse-frequency balancing.

### 2) Batch inference

Primary files:

- `src/taskclf/infer/batch.py`
- `src/taskclf/infer/calibration.py`
- `src/taskclf/infer/smooth.py`

Current behavior:

- Uses the same categorical encoding path as training.
- Fills numeric nulls with `0`.
- Applies optional calibrator or calibrator store.
- Applies reject threshold on the resulting probabilities.
- Converts low-confidence rows to `Mixed/Unknown`.
- Applies rolling-majority smoothing.
- Builds segments from smoothed bucket predictions.

### 3) Online inference

Primary file:

- `src/taskclf/infer/online.py`

Current behavior:

- Loads the frozen model bundle and categorical encoders.
- Polls ActivityWatch window data and, when available, input data.
- Builds fresh `FeatureRow`s from the latest events.
- Predicts one bucket at a time through `OnlinePredictor.predict_bucket()`.
- Applies calibrator or per-user calibrator store.
- Rejects low-confidence buckets.
- Smooths predictions with rolling majority.

### 4) Tray/UI suggestion path

Primary files:

- `src/taskclf/ui/tray.py`
- `src/taskclf/features/build.py`

Current behavior:

- `ActivityMonitor` detects transitions using dominant-app persistence.
- `_LabelSuggester.suggest(start, end)` fetches only ActivityWatch window events for that transition-bounded window.
- It calls `build_features_from_aw_events(events)` with defaults.
- It predicts on `rows[-1]` only.
- The returned suggestion is the predictor's `core_label_name` and confidence for that last bucket.

Implication:

- The suggestion flow is not a full interval classifier.
- It is effectively a single-bucket inference triggered by an interval boundary.

---

## Why This Matters For LightGBM

LightGBM is a strong fit for tabular classification, but it does not infer temporal structure on its own.

That means quality depends on:

- whether recent history is explicitly encoded into features
- whether live feature construction matches training exactly
- whether unknown categories are handled consistently
- whether confidence calibration matches the probabilities used for decisions
- whether per-bucket predictions are aggregated correctly for user-facing suggestions

If the feature vector is wrong, stale, incomplete, or inconsistent between training and serving, model quality will degrade even if the model itself is fine.

---

## Current Quality Risks

## 0) Highest priority: train/serve mismatch on missing numeric values

Files:

- `src/taskclf/train/lgbm.py`
- `src/taskclf/infer/batch.py`
- `src/taskclf/infer/online.py`

Observed behavior:

- Training and batch inference use `fillna(0)`.
- Online inference converts missing numerics to `NaN` in `OnlinePredictor._encode_value()`.

Why this is bad:

- The online model sees a different feature distribution than the one it was trained on.
- Confidence calibration and reject behavior can drift.
- This is the clearest correctness issue in the live pipeline.

## 1) Live rolling context is weaker than the model expects

Files:

- `src/taskclf/features/build.py`
- `src/taskclf/infer/online.py`
- `src/taskclf/features/windows.py`
- `src/taskclf/features/dynamics.py`

Observed behavior:

- Several features depend on recent history:
  - app-switch counts over 5m and 15m
  - rolling keyboard/mouse summaries
  - delta features
  - session-length features
- The live loop currently builds rows from the latest fetched slice rather than from a persistent rolling feature state aligned to how those features were conceived.

Why this is bad:

- Temporal features may be truncated or less informative online.
- The model may have learned from richer historical context than it gets during live inference.

## 2) Tray suggestions ignore most of the interval

Files:

- `src/taskclf/ui/tray.py`
- `src/taskclf/features/build.py`
- `src/taskclf/infer/online.py`

Observed behavior:

- Suggestion windows are transition-bounded.
- Feature rows are built for the whole window.
- Only the last bucket is scored for the returned suggestion.

Why this is bad:

- A long unlabeled block is reduced to its last minute.
- Earlier context in the interval does not influence the final suggestion except indirectly through features present in that last bucket.

## 3) Tray suggestions do not use input events

Files:

- `src/taskclf/ui/tray.py`
- `src/taskclf/adapters/activitywatch/client.py`
- `src/taskclf/features/build.py`

Observed behavior:

- The tray suggestion path fetches only window events.
- Input-derived fields remain `None`.

Why this is bad:

- If the trained model relies on keyboard/mouse activity, suggestion-time quality is weakened.
- This is another train/serve mismatch between the tray path and the main online path.

## 4) Stable `user_id` is not passed through live suggestion feature building

Files:

- `src/taskclf/train/lgbm.py`
- `src/taskclf/infer/online.py`
- `src/taskclf/ui/tray.py`
- `src/taskclf/core/config.py`
- `src/taskclf/features/build.py`

Observed behavior:

- `user_id` is part of `FEATURE_COLUMNS`.
- Online calibration selection uses `row.user_id`.
- The tray suggestion path builds rows with the default `user_id` instead of the stable config-backed `user_id`.

Why this is bad:

- Per-user calibration can be skipped or misapplied.
- The model loses a trained personalization signal.
- Unknown-user fallback behavior may trigger more often than intended.

## 5) Reject threshold is not consistently tied to calibrated probabilities

Files:

- `src/taskclf/train/evaluate.py`
- `src/taskclf/train/calibrate.py`
- `src/taskclf/infer/calibration.py`
- `src/taskclf/infer/batch.py`
- `src/taskclf/infer/online.py`

Observed behavior:

- Production can reject on calibrated probabilities.
- Evaluation and reject tuning are centered on raw `predict_proba()` outputs unless extended manually.

Why this is bad:

- Thresholds tuned offline may not be optimal for the probabilities used at inference time.
- A fixed threshold like `0.55` can look reasonable offline and still behave poorly live.

## 6) Bundle metadata does not define the active reject policy

Files:

- `src/taskclf/core/model_io.py`
- `src/taskclf/cli/main.py`
- `src/taskclf/ui/server.py`

Observed behavior:

- `ModelMetadata` has `reject_threshold`.
- The main training flows do not consistently persist or consume that threshold as the runtime default.

Why this is bad:

- There is no single source of truth for the threshold associated with a model bundle.
- Reproducing deployed behavior becomes harder.

## 7) Unknown-category handling needs validation

Files:

- `src/taskclf/train/lgbm.py`
- `src/taskclf/infer/online.py`
- `src/taskclf/infer/batch.py`

Observed behavior:

- Unknown categoricals map to `-1`.
- Training data usually does not contain that code.

Why this is bad:

- The fallback may work, but we should not assume it is harmless.
- New apps and new users are common in the real world.

## 8) Evaluation is mostly bucket-level, not operational-quality-level

Files:

- `src/taskclf/train/evaluate.py`
- `src/taskclf/core/metrics.py`
- `src/taskclf/infer/smooth.py`
- `docs/guide/acceptance.md`

Observed behavior:

- Current reporting emphasizes macro-F1, weighted F1, per-class metrics, confusion matrix, and reject rate.
- The production system also depends on:
  - smoothing quality
  - segment stability
  - suggestion usefulness over intervals
  - cold-start robustness

Why this is bad:

- We can improve offline bucket metrics while leaving the real UX unchanged or worse.

---

## Key Files And Symbols

### Training and feature contract

- `src/taskclf/train/lgbm.py`
  - `FEATURE_COLUMNS`
  - `CATEGORICAL_COLUMNS`
  - `encode_categoricals()`
  - `prepare_xy()`
  - `train_lgbm()`
- `src/taskclf/features/build.py`
  - `build_features_from_aw_events()`
- `src/taskclf/core/types.py`
  - `FeatureRow`

### Inference and post-processing

- `src/taskclf/infer/batch.py`
  - `predict_proba()`
  - `run_batch_inference()`
- `src/taskclf/infer/online.py`
  - `OnlinePredictor`
  - `OnlinePredictor._encode_value()`
  - `OnlinePredictor.predict_bucket()`
  - `run_online_loop()`
- `src/taskclf/infer/smooth.py`
  - `rolling_majority()`
  - `segmentize()`
  - `merge_short_segments()`
- `src/taskclf/infer/calibration.py`
  - `CalibratorStore`
  - `TemperatureCalibrator`
  - `IsotonicCalibrator`

### Evaluation and retraining

- `src/taskclf/train/evaluate.py`
  - `evaluate_model()`
  - `tune_reject_threshold()`
- `src/taskclf/train/calibrate.py`
  - `fit_calibrator_store()`
  - `check_personalization_eligible()`
- `src/taskclf/train/retrain.py`
  - `run_retrain_pipeline()`
  - regression and promotion gates

### Tray suggestions and UI path

- `src/taskclf/ui/tray.py`
  - `ActivityMonitor`
  - `_LabelSuggester`
  - `TrayLabeler._handle_transition()`
- `src/taskclf/adapters/activitywatch/client.py`
  - `fetch_aw_events()`
  - `fetch_aw_input_events()`

---

## Recorded Decisions

### 1) Personalization architecture

Decision:

- Long-term, personalization will move into calibrators and user-specific post-processing.
- Long-term, `user_id` should be removed from the core model feature contract.
- This requires a new schema/model generation rather than silently changing current models.

Implications:

- The base classifier should become more portable and less coupled to the training cohort.
- Per-user behavior should be expressed through:
  - per-user calibration
  - per-user reject thresholds
  - per-user priors or bias adjustments if needed
  - per-user smoothing, taxonomy, or other post-processing where justified
- Existing schema-v1 models still expect `user_id` in the feature vector, so live inference must continue passing the stable `user_id` correctly until those models are retired.

Migration rule:

- Do not remove `user_id` from live inference for current bundles until a new schema version is defined, models are retrained, and evaluation confirms that the new architecture is acceptable.

Docs note:

- `docs/guide/personalization.md` currently documents the existing hybrid design.
- That guide should be updated only when the schema/model migration is implemented, not before.

---

## Ordered Implementation Plan

### Phase 0) Lock the intended inference contract

1. Write down the canonical inference order and keep every path consistent: build features, encode categoricals, impute missing values, predict probabilities, calibrate, reject, smooth or aggregate, and map to the UI-facing label.

2. Decide which runtime behaviors are authoritative: batch inference, online live loop, and tray suggestion flow.

3. Define what a tray suggestion should mean: last-minute prediction, interval-level suggestion, or interval summary from bucket predictions.

4. Record the long-term direction explicitly: current schema/models still require `user_id` for compatibility, the next schema/model generation should remove `user_id` from the core model, and personalization should live in calibrators and user-specific post-processing.

5. Define the migration boundary clearly: old bundles keep current behavior, new bundles adopt the new personalization architecture, and schema/version gates must prevent mixing the two contracts accidentally.

### Phase 1) Remove correctness mismatches first

1. Make online numeric missing-value handling match training and batch inference exactly.

2. Until the schema migration lands, pass stable `user_id` through all live feature-construction paths so current bundles keep working correctly.

3. Ensure suggestion-time inference can use the same input-event sources as live online inference.

4. Decide whether unknown categoricals should continue mapping to `-1` or to an explicit trained fallback category.

### Phase 2) Migrate personalization out of the core model

1. Create the next schema/model contract that removes `user_id` from `FEATURE_COLUMNS` and categorical encoders.

2. Retrain and evaluate side-by-side: the current hybrid model with `user_id` in the base model, a new base model without `user_id`, and a new base model plus calibrator/post-processing personalization.

3. Extend personalization beyond calibration where needed: per-user reject thresholds, per-user priors or bias adjustments, and per-user smoothing or taxonomy/post-processing.

4. Keep backward compatibility during rollout: schema-v1 bundles still receive stable `user_id`, new bundles stop depending on it, and model loading plus docs make the split explicit.

### Phase 3) Strengthen live context for a tabular model

1. Add a persistent online feature-state layer so recent context is available at inference time instead of being reconstructed from a narrow slice.

2. Preserve enough history for all current derived features: 5m switch counts, 15m switch counts, rolling input statistics, delta features, and session features.

3. Ensure live session tracking and feature-state transitions are consistent with the assumptions in `build_features_from_aw_events()`.

### Phase 4) Fix tray suggestion quality

1. Replace "predict only `rows[-1]`" with an interval-aware suggestion strategy.

2. Compare at least these aggregation strategies: majority vote over bucket predictions, confidence-weighted vote over bucket predictions, highest-total-probability label over the interval, and the most recent confident contiguous segment within the interval.

3. Decide whether the tray should display the raw last-bucket label, a smoothed label, or an interval-aggregated label.

4. Include input events and any required user-scoped post-processing inputs in tray suggestion features.

### Phase 5) Align evaluation with deployed behavior

1. Add evaluation modes for raw probabilities, calibrated probabilities, reject-on-calibrated predictions, smoothed bucket labels, and interval-level suggestion evaluation.

2. Tune reject thresholds on the same probability outputs used in production.

3. Persist the chosen threshold in bundle metadata and make inference default to it unless explicitly overridden.

4. Report both bucket-level classification quality and operational-quality metrics that users actually feel.

### Phase 6) Improve features for LightGBM

1. Prioritize features that summarize recent behavior rather than features that require a different model family.

2. Candidate additions to evaluate include previous accepted label, minutes since last accepted label, app dwell time, app entropy over the last 5m and 15m, share of time in browser/editor/terminal/meeting categories, idle-return indicator, top-2 app concentration over the interval, and a stability score for the current block.

3. Keep new features privacy-safe and schema-versioned.

4. Add tests for every new feature and for schema-hash changes.

### Phase 7) Tune only after the pipeline is correct

1. After parity and context fixes, tune LightGBM parameters on time-based validation.

2. Tune at least `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, and learning rate versus boost rounds.

3. Compare tuning results against the champion model and existing regression gates.

---

## Measurement Plan

We should not treat "better macro-F1" as the only success criterion.

Track at least:

- macro-F1
- weighted F1
- per-class precision, recall, F1
- seen-user vs unseen-user macro-F1
- reject rate
- calibration quality
- confusion matrix
- flip rate before and after smoothing
- segment duration distribution
- suggestion acceptance rate in the tray/UI
- suggestion precision when users accept or overwrite

Add explicit comparison tables for:

- raw vs calibrated
- no-reject vs reject
- unsmoothed vs smoothed
- batch-style interval aggregation vs last-bucket-only suggestion
- with-input-events vs without-input-events
- stable-user-id vs default-user fallback

---

## Proposed Experiments

### Experiment A: train/serve parity only

Change nothing except:

- online missing-value handling
- stable `user_id`
- suggestion path input events

Goal:

- estimate how much quality is being lost to correctness mismatches alone

### Experiment B: interval suggestion aggregation

Compare:

- current `rows[-1]`
- majority vote over interval buckets
- confidence-weighted aggregation

Goal:

- measure whether tray suggestions become more aligned with user labels

### Experiment C: calibrated-threshold tuning

Tune reject thresholds on:

- raw scores
- calibrated scores

Goal:

- determine whether the current threshold policy is misaligned with deployed probabilities

### Experiment D: richer live context

Introduce persistent online history for derived features.

Goal:

- measure gains from better temporal context without changing model family

### Experiment E: feature additions

Add one small group of features at a time.

Goal:

- preserve attribution for what actually helped

---

## Open Questions

1. What should the tray suggestion represent semantically:
   - the last minute
   - the previous block
   - the whole unlabeled interval since last label
2. Should the canonical runtime threshold live in:
   - bundle metadata
   - calibrator store metadata
   - CLI config
   - all three with precedence rules
3. Do we want the tray/UI to prefer:
   - fewer but higher-confidence suggestions
   - more frequent but noisier suggestions
4. Do we need explicit unknown-category buckets in training instead of relying on inference-time `-1`?

---

## Suggested First Execution Slice

If we want the highest expected quality gain with the lowest risk, do this first:

1. Fix online `NaN` vs `fillna(0)` parity.
2. For current schema-v1 bundles, pass stable `user_id` through live feature construction so existing models remain correct.
3. Add input-event support to `_LabelSuggester`.
4. Replace tray last-bucket suggestion with interval aggregation over bucket predictions.
5. Re-run evaluation with calibrated reject tuning and compare before/after.

This slice improves correctness and product behavior without changing the model family.
It is also compatible with the chosen long-term direction: keep current bundles working now, then remove `user_id` from the core model in the next schema/model generation.

---

## Definition Of Done For "Inference Quality v1"

- All inference paths share the same preprocessing contract.
- Current schema-v1 inference paths preserve stable `user_id` where required for compatibility, and the next schema/model generation removes `user_id` from the core model contract.
- Personalization is applied through calibrators and user-specific post-processing rather than identity splits in the long-term base model.
- Suggestion-time inference uses the same feature families as training whenever the data source exists.
- Reject threshold is tuned and evaluated on the same probability outputs used in production.
- Tray suggestions are interval-aware, not last-bucket-only.
- Evaluation includes operational metrics, not only bucket-level macro-F1.
- Docs and tests cover the new behavior.
