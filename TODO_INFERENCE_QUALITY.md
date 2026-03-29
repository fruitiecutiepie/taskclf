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
- Categoricals are label-encoded with unknown values mapped to `-1` at inference time. Decision #5 changes this: unseen values will map to a trained `__unknown__` code instead, with frequency thresholding and random masking during training.
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

## 7) Unknown-category handling needs validation (resolved — decision #5)

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

Resolution: decision #5 adopts hybrid unknown-category training (frequency thresholding + random masking). Implementation is Phase 1 step 4; evaluation is Experiment F.

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

### 2) Canonical runtime threshold location

Decision:

- The canonical reject threshold lives in a dedicated **inference policy** artifact (`models/inference_policy.json`), not in bundle metadata, calibrator store metadata, or CLI config alone.
- The inference policy is a small versioned deployment descriptor that binds a specific model bundle + optional calibrator store + reject threshold into a single source of truth for how inference should behave.
- `ModelMetadata.reject_threshold` is retained as an advisory historical record of what was used during evaluation but is not authoritative at runtime.
- `CalibratorStore` is promoted to a model-bound artifact: `store.json` now records `model_bundle_id`, `model_schema_hash`, and `created_at` for traceability and cross-validation.

Implications:

- Inference resolution follows a strict precedence: explicit `--model-dir` CLI override > `inference_policy.json` > `active.json` fallback > best-model selection + code defaults.
- Hot-reload in the online loop now swaps threshold + calibrator alongside the model, eliminating silent drift where CLI-supplied values stuck after a model swap.
- `active.json` remains for model-registry selection but is no longer the canonical deployment mechanism; inference logs a warning when falling back to it.
- CLI flags (`--reject-threshold`, `--calibrator-store`) remain as explicit overrides that take precedence over policy values.
- Legacy code paths (`resolve_model_dir`, `ActiveModelReloader`, direct `_LabelSuggester(model_dir)` construction) are deprecated with log warnings and docstring annotations; target removal in a future major version.

Artifacts:

- `src/taskclf/core/inference_policy.py` — `InferencePolicy` model, `build_inference_policy`, `save_inference_policy`, `load_inference_policy`, `validate_policy`, `PolicyValidationError`
- `src/taskclf/infer/resolve.py` — `ResolvedInferenceConfig`, `resolve_inference_config`, `InferencePolicyReloader`
- CLI: `taskclf policy show|create|remove`, `--write-policy` on `train tune-reject` and `train retrain`
- Docs: `docs/api/core/inference_policy.md`

### 3) Tray suggestion semantics

Decision:

- The three candidate semantics are complementary product surfaces, not competing definitions of one tray suggestion.
- Live status should answer "what am I doing right now?" and may remain last-minute.
- Automatic transition-triggered tray suggestions should default to the previous completed block.
- Explicit backlog or gap-fill labeling should target the whole unlabeled interval since the last confirmed label.

Implications:

- Trigger semantics and target semantics must stay distinct in the implementation and UI copy.
- Confidence and explanation must always be scoped to the interval the surface actually represents.
- The default automatic tray suggestion should remain bounded and recoverable; over-segmentation is acceptable because adjacent same-label blocks can be merged later.
- Longer unlabeled intervals are still valuable, but they should be handled as an explicit workflow rather than silently overloading the automatic tray suggestion.

UI surface architecture: decision #6 specifies how these three semantics map to separate UI surfaces, including copy conventions, gap-fill prompting rules, and confidence display policy.

### 4) Suggestion confidence strategy

Decision:

- Automatic tray suggestions should default to **fewer but higher-confidence** suggestions.
- The rejected alternative — more frequent but noisier suggestions (lower reject threshold, higher coverage, lower precision) — was considered and rejected because the trust, label-quality, and notification-fatigue costs outweigh the coverage and feedback-loop benefits in a notification-driven tray tool. See Rationale below.
- The reject threshold should be set high enough that surfaced suggestions are correct the large majority of the time.
- Coverage gaps left by conservative automatic suggestions are handled by the explicit gap-fill workflow (decision #3), not by lowering the automatic threshold.
- Per-user reject thresholds (from decision #1) are the mechanism for progressively loosening the threshold as calibration quality improves for each user, rather than starting noisy and tightening later.

Rationale:

- Trust is asymmetric: many correct suggestions build trust slowly, but a few wrong ones break it quickly. In a notification-driven tray tool, this asymmetry is amplified because suggestions interrupt the user at context-switch moments when cognitive load is already high.
- The cost of a wrong suggestion exceeds the cost of no suggestion. No suggestion costs X effort (manual label). A correct suggestion costs less than X (glance + accept). A wrong suggestion costs more than X (read, evaluate, reject, relabel, plus frustration). So every wrong suggestion is strictly worse than silence.
- Label quality compounds. If noisy suggestions cause users to rubber-stamp incorrect labels, those bad labels enter training data, degrade the next model, and produce worse suggestions. A high threshold limits exposure to this feedback loop.
- The failure mode of too-conservative (system feels quiet) is recoverable. The failure mode of too-noisy (system feels unreliable, user disables notifications) is much harder to reverse.
- Interval aggregation (Phase 4) naturally improves confidence for homogeneous intervals and lowers it for ambiguous ones. A high reject threshold and interval aggregation are complementary.

Implications:

- The default reject threshold should be tuned to optimize suggestion precision rather than recall. When in doubt, suppress.
- The inference policy should define a minimum engagement floor: if a loaded model produces zero suggestions over a full active day, the system should log a warning and the calibration pipeline should flag that the threshold may be too high.
- Cold-start periods (new users, new apps) will produce fewer automatic suggestions. This is acceptable because the gap-fill workflow provides an explicit fallback, and the cold-start window is bounded by the time needed for personalization-eligible calibration.
- Per-user thresholds should start conservative and lower progressively as per-user calibration quality improves, not the reverse.
- The UI should not need to communicate uncertainty levels to the user. Suggestions should carry enough confidence that "we think you were doing X" is a clean, unhedged message.

User configurability:

- The reject threshold is not exposed as a user-facing preference. Users who pick "more suggestions" cannot connect that choice to the downstream effect on label quality and model degradation. The preference they actually have — notification frequency and timing — should be expressed through quiet hours, DND, batching, and snooze, which control interruption without changing the quality bar.
- Coverage control is already expressed through the three surfaces in decision #3: users who want more coverage use the gap-fill workflow actively; users who want fewer interruptions get conservative automatic suggestions only.
- Per-user calibration (decisions #1 and #4) achieves the same outcome as a "more suggestions" slider, but only when the model can actually back it up.
- The existing CLI override (`--reject-threshold`) and `inference_policy.json` remain available for power users who understand the tradeoff.

Per-surface thresholds:

- Decision #4 implies tray suggestions need a stricter operating point than batch or online inference. Currently all three surfaces share the single `InferencePolicy.reject_threshold`.
- This works for now because the surfaces already have different effective operating points after the threshold: batch and online apply rolling-majority smoothing and segment merging downstream, which absorbs single-bucket rejects. Tray suggestions show one prediction directly to the user with no downstream recovery. So a single threshold tuned for tray precision is the most conservative and batch/online tolerate it via their post-threshold processing.
- If Phase 5 evaluation reveals that a tray-tuned threshold leaves batch reports unacceptably gappy, the threshold should be promoted to per-surface values in the inference policy schema (`policy_version: "v2"`), not hidden as ad-hoc overrides in the tray or batch code. The architectural rule: any surface-specific operating point must live in the deployment descriptor.
- Trigger for the split: if the `Mixed/Unknown` rate in batch inference exceeds 25% of active buckets when using the tray-tuned threshold, and lowering the threshold for batch would recover at least half of those buckets at acceptable precision, the split is justified. Measure this as part of Phase 5 evaluation.
- Do not split before Phase 5 lands. Tuning multiple thresholds against miscalibrated scores means guessing at more numbers with no better information.

Guardrail:

- A threshold so high that users never see suggestions is functionally broken regardless of precision. The system must track suggestions-per-active-day and flag when it drops to zero for a user with a loaded model.

Artifacts and code paths:

- `InferencePolicy.reject_threshold` in `src/taskclf/core/inference_policy.py` — single threshold consumed by all surfaces today. If per-surface thresholds are added, they belong here as new fields.
- `OnlinePredictor.predict_bucket()` in `src/taskclf/infer/online.py` — applies the reject decision. Used by both the online loop and the tray suggester.
- `_LabelSuggester` in `src/taskclf/ui/tray.py` — wraps `OnlinePredictor` for tray suggestions. Must not introduce its own threshold override; it inherits from the policy.
- `run_batch_inference()` in `src/taskclf/infer/batch.py` — batch reject path. Shares the same threshold today.
- `resolve_inference_config()` in `src/taskclf/infer/resolve.py` — resolves the threshold from policy, CLI override, or fallback. If per-surface thresholds are added, resolution must be extended to accept a surface identifier.
- Suggestions-per-active-day tracking: should be recorded in the telemetry store (`src/taskclf/core/telemetry.py`) alongside existing session telemetry, not in the tray or predictor. The tray publishes suggestion events via `EventBus`; telemetry aggregates them.

### 5) Unknown-category handling in training

Decision:

- Unknown categoricals must be trained explicitly using a hybrid approach: frequency-thresholded tail collapsing followed by random categorical masking.
- The rejected alternative — continuing to rely on inference-time `-1` for unseen values with no training exposure — was considered and rejected because it produces untrained confidence on novel inputs, which undermines the reject mechanism (decision #4) and risks feeding confidently-wrong labels into the training feedback loop.

Procedure:

- Step 1 (frequency thresholding): collapse all category values appearing fewer than a configurable threshold (default: 5 occurrences in training data) to a reserved `__unknown__` token. This gives the model natural training signal from the real tail distribution.
- Step 2 (random masking): independently mask a small fraction (default: 5%) of remaining known-category values to `__unknown__` during training. This forces the model to learn predictions from non-categorical features when the category is absent, and produces appropriately lower, broader probability distributions for unknown inputs.
- The `LabelEncoder` for each categorical column includes `__unknown__` as a first-class value. At inference time, unseen values map to the `__unknown__` code instead of `-1`.
- The frequency threshold and masking rate are training hyperparameters recorded in the model bundle metadata, not inference-time settings.

Rationale:

- Confidence must be meaningful for novel inputs. The reject mechanism, per-user calibration, and the "fewer but higher-confidence" suggestion strategy (decisions #1, #4) all depend on confidence being a reliable signal. When the model encounters a category it never trained on, inference-time `-1` produces leaf probabilities that were never optimized for that input. The model can be confidently wrong, which is the worst outcome for a system that uses confidence to gate suggestions.
- Calibration cannot fix what it has not seen. Calibrators are fitted on validation data containing only known categories. They correct systematic biases in the known distribution, not the behavior of a code absent from calibration data.
- The category space grows monotonically. New apps, new browser extensions, OS updates renaming bundle identifiers, new users — the fraction of predictions hitting unseen values increases over the system's lifetime. A model trained to handle unknowns becomes more robust over time, not less.
- The feedback loop is the decisive risk. If unknown categories produce confidently wrong suggestions that get rubber-stamped, those bad labels enter training data, degrade the next model, and produce worse suggestions. Explicit unknown training is a circuit breaker: it makes unknown-input confidence trustworthy, so the reject mechanism suppresses bad suggestions before they become bad labels.
- The accuracy cost is bounded and measurable. A 5% masking rate preserves 95% of the categorical signal. Frequency thresholding loses only tail categories that already carry weak signal. Both costs are visible in standard evaluation and can be tuned.

Implications:

- `encode_categoricals()` in `src/taskclf/train/lgbm.py` must be extended to support threshold-based collapsing and random masking during training, and to map unseen values to the `__unknown__` code (instead of `-1`) at inference time.
- `OnlinePredictor._encode_value()` in `src/taskclf/infer/online.py` must map unseen categoricals to the `__unknown__` code rather than `-1.0`.
- The frequency threshold and masking rate must be recorded in model bundle metadata so they are reproducible and auditable.
- Schema hash will change when `__unknown__` is added to the encoder vocabulary, requiring a new model generation. This aligns naturally with the schema-v2 migration (decision #1) and can be bundled with the `user_id` removal or done independently.
- Evaluation must include a held-out-category test: withhold some known categories from training, verify that the model produces appropriately uncertain predictions for them, and confirm that the reject threshold catches them. Add this to Phase 5 evaluation.
- Do not change the masking rate without re-evaluating both macro-F1 on known categories and reject-rate behavior on withheld categories. The two metrics are in tension: higher masking improves unknown robustness but reduces known-category accuracy.

### 6) Tray/UI surface architecture (resolves open question #1)

Decision:

- The three suggestion semantics (decision #3) must be implemented as **clearly separate UI surfaces from the start**, not as a single unified widget that changes meaning contextually.
- Ship in two stages: live status + transition suggestions first, gap-fill surface later. This reduces initial cognitive load while preserving the architectural separation.
- Gap-fill should be offered through a **passive indicator** with **contextual prompting at natural moments**, not through fixed-schedule automatic notifications or purely manual access.
- Confidence is **not exposed** to the user on live status or transition suggestion surfaces (per decision #4). In the gap-fill review surface, per-segment confidence may be represented as implicit visual weight (e.g., segment saturation/opacity) rather than as a numeric value.

Rejected alternatives and rationale:

- **Single unified surface with contextual mode switching** was rejected. The three surfaces have fundamentally different confidence profiles (single-bucket volatile, aggregated-block stable, long-interval heterogeneous), different interaction patterns (passive glance, notification accept/reject, explicit review workflow), and different evolution trajectories. A unified surface means the user can never develop calibrated expectations about what they are looking at. Mode confusion is the core risk: during a stable period the widget shows live status, after a transition it silently flips to a suggestion about the past, and the user may not notice the semantic shift. Splitting later forces a mental model migration on users who already formed habits, which is more costly than the upfront complexity of separate surfaces.
- **Hybrid single surface with strong visual mode indicators** was rejected as a weaker version of the same problem. If the indicators are obvious enough to prevent confusion, the surface is effectively two separate surfaces crammed into one container. If the indicators are subtle, they get ignored and the mode confusion returns.
- **Fully automatic gap-fill notifications at fixed triggers (end of day, after N minutes)** were rejected. Fixed triggers have bad timing (end-of-day prompts catch users leaving, mid-focus prompts interrupt deep work). If the model is uncertain about gap content — which is likely, since confident intervals would already be covered by transition suggestions — automatic gap-fill suggestions violate decision #4's precision-over-recall principle. The worst outcome is a notification the user dismisses reflexively, which spends the attention budget for zero signal.
- **Purely manual gap-fill (user must open a backlog view)** was rejected as too passive. Users will forget, gaps accumulate, and the model's retraining signal degrades. The system has information about unlabeled time that the user does not — surfacing it at the right moment is strictly more useful than silence.

Surface definitions:

- **Live status surface**: always-visible, passive, glanceable. Shows the model's current-bucket prediction. No interaction required. No confidence displayed. Answers: "what am I doing right now?"
- **Transition suggestion surface**: notification-style, triggered by `ActivityMonitor` transition detection. Shows an action-oriented prompt about the previous completed block with a concrete time range. Interaction: accept or reject. No confidence displayed. Answers: "was this [label]? [start]–[end]"
- **Gap-fill surface**: user-initiated review workflow, surfaced through a persistent passive indicator (badge, counter, or subtle timeline marker showing total unlabeled time). Actively prompted only at natural low-cost moments: returning from idle (>5 min), session start, or immediately after the user accepts a transition suggestion (piggybacks on existing labeling attention). Answers: "you have [duration] unlabeled — review?"

Copy and label conventions:

- Transition suggestions use action-oriented framing with a concrete time range to disambiguate interval scope: "Was this Coding? 12:00–12:47" rather than "We think you were coding."
- Live status uses a simple present-tense statement: "Now: Coding." No time range needed — it is always the current moment.
- Gap-fill uses action-oriented labeling that makes it clear it is a review task: "Review unlabeled: 9:00–11:30" or "You have 2h 30m unlabeled. Review?"
- Do not use hedging language ("We think you might have been..."). Decision #4 established that suggestions only surface when confidence is high enough for an unhedged statement. Copy should reflect that.
- Avoid language that conflates the surfaces. "Just now" is ambiguous for a block that ended 2 hours ago. The concrete time range carries the meaning.

Gap-fill prompting rules:

- The passive indicator (unlabeled-time badge) is always visible when unlabeled time exists. It does not interrupt.
- Active prompting occurs only at: (a) idle return (>5 min idle detected by `ActivityMonitor`), (b) session start, (c) immediately after the user accepts a transition suggestion ("You just labeled 12:00–12:47 as Coding. There's also 9:00–11:30 unlabeled. Review?").
- If unlabeled time exceeds a configurable large threshold (default: one full active day with zero labels), the passive indicator escalates — e.g., the tray icon itself changes state. Not a popup. This handles the case where a user completely ignores the system without adding notification pressure during normal operation.
- The escalation threshold is a user-facing config (unlike the reject threshold, which is not user-facing per decision #4), because it controls notification intensity, not quality.

Confidence display rules:

- Live status and transition suggestions: confidence is an internal gating signal only. The user sees an unhedged label or nothing (rejected by threshold). No percentages, no confidence bars, no uncertainty language.
- Gap-fill review surface: per-segment confidence may be represented as implicit visual weight (segment saturation, opacity, or similar) to help the user prioritize which parts of a long interval to review carefully vs rubber-stamp. This is not a numeric display — it is a visual hint. Defer implementation to the gap-fill phase; it is not required for v1.

Interaction with other decisions:

- Decision #3 defined the three semantics. Decision #6 specifies how those semantics map to UI surfaces and interaction patterns.
- Decision #4's precision-over-recall strategy applies to live status and transition suggestions. Gap-fill is a review workflow, not a notification, so its quality bar is different: the system should show its best guess for the interval and let the user correct, rather than suppressing uncertain segments entirely.
- Per-user calibration (decision #1) may eventually allow the transition suggestion surface to fire more often for well-calibrated users, but the surface architecture remains the same.

Implementation sequencing:

- Phase 4 step 1 implements the transition suggestion surface with interval aggregation (replacing last-bucket-only prediction).
- Phase 4 step 3 implements the live status surface as a separate read-only display.
- Gap-fill surface (passive indicator + contextual prompting) is a separate implementation phase after Phase 4. It depends on the transition suggestion surface working correctly but does not block it.

Artifacts and code paths:

- `src/taskclf/ui/tray.py` — `_LabelSuggester` currently serves both live status and transition suggestions through a single code path. Needs refactoring to separate the two surfaces with distinct display and interaction logic.
- `ActivityMonitor` in `src/taskclf/ui/tray.py` — already detects transitions and idle periods. Idle-return detection can be extended for gap-fill prompting triggers.
- Gap-fill passive indicator — new UI element. Implementation deferred but the tray architecture should not preclude it.
- Copy strings should be externalized or at minimum centralized in one module, not scattered across tray/UI code, to keep the labeling conventions consistent and changeable.

---

## Ordered Implementation Plan

### Phase 0) Lock the intended inference contract

1. Write down the canonical inference order and keep every path consistent: build features, encode categoricals, impute missing values, predict probabilities, calibrate, reject, smooth or aggregate, and map to the UI-facing label.

2. Decide which runtime behaviors are authoritative: batch inference, online live loop, and tray suggestion flow.

3. Apply the chosen semantic split consistently: live-status predictions can stay last-minute, automatic tray suggestions should default to the previous completed block, and explicit backlog labeling can target the whole unlabeled interval since the last confirmed label.

4. Record the long-term direction explicitly: current schema/models still require `user_id` for compatibility, the next schema/model generation should remove `user_id` from the core model, and personalization should live in calibrators and user-specific post-processing.

5. Define the migration boundary clearly: old bundles keep current behavior, new bundles adopt the new personalization architecture, and schema/version gates must prevent mixing the two contracts accidentally.

### Phase 1) Remove correctness mismatches first

1. Make online numeric missing-value handling match training and batch inference exactly.

2. Until the schema migration lands, pass stable `user_id` through all live feature-construction paths so current bundles keep working correctly.

3. Ensure suggestion-time inference can use the same input-event sources as live online inference.

4. Implement unknown-category handling per decision #5: add frequency-thresholded tail collapsing and random categorical masking to `encode_categoricals()`, map unseen inference values to the trained `__unknown__` code instead of `-1`.

### Phase 2) Migrate personalization out of the core model

1. Create the next schema/model contract that removes `user_id` from `FEATURE_COLUMNS` and categorical encoders. This migration is a natural bundling point for the `__unknown__` encoding change (decision #5), since both require a schema hash bump and new model generation. They can also land independently if sequencing requires it.

2. Retrain and evaluate side-by-side: the current hybrid model with `user_id` in the base model, a new base model without `user_id`, and a new base model plus calibrator/post-processing personalization.

3. Extend personalization beyond calibration where needed: per-user reject thresholds (starting conservative and lowering as calibration quality improves per decision #4), per-user priors or bias adjustments, and per-user smoothing or taxonomy/post-processing.

4. Keep backward compatibility during rollout: schema-v1 bundles still receive stable `user_id`, new bundles stop depending on it, and model loading plus docs make the split explicit.

### Phase 3) Strengthen live context for a tabular model

1. Add a persistent online feature-state layer so recent context is available at inference time instead of being reconstructed from a narrow slice.

2. Preserve enough history for all current derived features: 5m switch counts, 15m switch counts, rolling input statistics, delta features, and session features.

3. Ensure live session tracking and feature-state transitions are consistent with the assumptions in `build_features_from_aw_events()`.

### Phase 4) Fix tray suggestion quality and implement surface architecture

1. Replace "predict only `rows[-1]`" for automatic tray suggestions with interval-aware aggregation over the previous completed block.

2. Compare at least these aggregation strategies: majority vote over bucket predictions, confidence-weighted vote over bucket predictions, highest-total-probability label over the interval, and the most recent confident contiguous segment within the interval.

3. Refactor `_LabelSuggester` and tray UI code to separate the transition suggestion surface from the live status surface (decision #6). The transition suggestion surface shows action-oriented prompts with concrete time ranges ("Was this Coding? 12:00–12:47"). The live status surface is a passive, always-visible, read-only display of the current-bucket prediction ("Now: Coding").

4. Centralize copy strings for all surfaces in one module. Apply the copy conventions from decision #6: action-oriented framing, concrete time ranges, no hedging language, no confidence percentages on live or transition surfaces.

5. Include input events and any required user-scoped post-processing inputs in tray suggestion features.

### Phase 4b) Gap-fill surface

Depends on Phase 4 (transition suggestions and live status must work correctly first). Can be implemented independently after Phase 4 lands.

1. Implement the passive unlabeled-time indicator (badge, counter, or timeline marker) as a persistent, non-interrupting UI element.

2. Implement contextual gap-fill prompting at the three trigger points defined in decision #6: idle return (>5 min idle detected by `ActivityMonitor`), session start, and immediately after the user accepts a transition suggestion.

3. Implement the escalation threshold: if unlabeled time exceeds a configurable threshold (default: one full active day with zero labels), escalate the passive indicator (e.g., change tray icon state). Do not escalate to a popup.

4. If per-segment confidence visualization is included, represent it as implicit visual weight (saturation, opacity) in the gap-fill review surface, not as numeric values. This is optional for v1.

### Phase 5) Align evaluation with deployed behavior

1. Add evaluation modes for raw probabilities, calibrated probabilities, reject-on-calibrated predictions, smoothed bucket labels, and interval-level suggestion evaluation.

2. Tune reject thresholds on the same probability outputs used in production. Optimize for suggestion precision over recall (decision #4): when in doubt, suppress.

3. Persist the tuned threshold in the inference policy artifact (`taskclf train tune-reject --write-policy`) so inference defaults to it unless explicitly overridden via CLI.

4. Report both bucket-level classification quality and operational-quality metrics that users actually feel.

5. Track suggestions per active day and flag when a loaded model produces zero suggestions for a user, indicating the threshold may be too high (decision #4 guardrail).

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
- suggestions per active day (must not drop to zero for users with a loaded model)

Add explicit comparison tables for:

- raw vs calibrated
- no-reject vs reject
- unsmoothed vs smoothed
- batch-style interval aggregation vs last-bucket-only suggestion
- with-input-events vs without-input-events
- stable-user-id vs default-user fallback
- known-category vs withheld-category confidence and reject behavior (decision #5)

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

### Experiment F: unknown-category training (decision #5)

Compare:

- current inference-time `-1` (no training exposure)
- frequency-thresholded tail collapsing only
- random categorical masking only
- hybrid (threshold + masking)

Evaluate on:

- macro-F1 on known categories (should not regress meaningfully)
- confidence distribution on withheld categories (should shift toward lower, broader probabilities)
- reject rate on withheld categories (should increase, confirming that the reject mechanism catches unknowns)
- suggestion precision on withheld categories (should improve or remain stable)

Goal:

- confirm that the hybrid approach produces trustworthy confidence on novel inputs without meaningful accuracy loss on known inputs
- select frequency threshold and masking rate defaults

---

## Open Questions

No open questions at this time. All previously open questions have been resolved:

1. ~~How should the tray/UI expose the already-chosen complementary suggestion semantics without confusing users?~~ — Resolved by decision #6 (tray/UI surface architecture). Separate surfaces from the start; action-oriented copy with concrete time ranges; gap-fill via passive indicator with contextual prompting; confidence hidden from live/transition surfaces.

---

## Suggested First Execution Slice

If we want the highest expected quality gain with the lowest risk, do this first:

1. Fix online `NaN` vs `fillna(0)` parity.
2. For current schema-v1 bundles, pass stable `user_id` through live feature construction so existing models remain correct.
3. Add input-event support to `_LabelSuggester`.
4. Replace automatic tray last-bucket suggestion with previous-block interval aggregation over bucket predictions.
5. Separate the transition suggestion surface from the live status surface in the tray UI code (decision #6). Transition suggestions use action-oriented copy with concrete time ranges; live status is a passive, always-visible current-bucket display.
6. Re-run evaluation with calibrated reject tuning and compare before/after.

This slice improves correctness and product behavior without changing the model family.
It is also compatible with the chosen long-term direction: keep current bundles working now, then remove `user_id` from the core model in the next schema/model generation.
The surface separation (step 5) establishes the UI architecture needed for decision #6. The gap-fill surface (passive indicator + contextual prompting) is a follow-on that depends on the transition suggestion surface working correctly but does not block the first slice.

---

## Agent Implementation Guide

Step-by-step instructions for each phase. Every step names the exact files, functions, and edits required, followed by tests and docs to update. Steps within a phase are ordered by dependency; complete each step fully before starting the next.

### How to read this document

This file is ~2000 lines. Do NOT read it all at once. For each step:

1. Read lines 1–40 (Goals, Non-Negotiables) for project constraints.
2. Read the specific step you are implementing (use the line ranges below).
3. Read the corresponding Test Plan entries (referenced by test ID in each step).
4. Read `AGENTS.md` for repo-wide rules.
5. Read each target source file before editing it.

### Progress tracking

After completing a step, change its checkbox from `- [ ]` to `- [x]` and append the date:

```markdown
- [x] **Step 1.1 — Fix online numeric missing-value handling (Risk 0)** ✅ 2026-03-28
```

Do NOT delete or strikethrough completed steps — later phases reference earlier decisions and rationale.

To find the next step to work on, scan for the first unchecked `- [ ]` in this section.

### General rules for every step

- Read the target source file before editing.
- Run `make test` (or `pytest tests/`) after each step to confirm no regressions.
- Run `ruff check src/ tests/` after editing Python files.
- Update the API doc page under `docs/api/` for any file whose public interface changed.
- Mark the corresponding test IDs from the Test Plan as implemented.

---

### Phase 0 — Lock the inference contract

This phase is documentation and decision-recording only. No production code changes.

- [x] **Step 0.1 — Document canonical inference order** ✅ 2026-03-28

1. Create `docs/guide/inference_contract.md`.
2. Write down the canonical pipeline order (this is already captured in the TODO above — formalize it):
   `features → encode categoricals → impute missing → predict probabilities → calibrate → reject → smooth/aggregate → taxonomy map → UI label`
3. For each of the three runtime paths (batch, online, tray), list which functions implement each stage:
   - Batch: `predict_proba()` in `infer/batch.py` → `run_batch_inference()` in `infer/batch.py`
   - Online: `OnlinePredictor.predict_bucket()` in `infer/online.py`
   - Tray: `_LabelSuggester.suggest()` in `ui/tray.py` → delegates to `OnlinePredictor.predict_bucket()`
4. Note any current deviations from the canonical order (these are the Phase 1 fixes).
5. Update `docs/api/infer/online.md`, `docs/api/infer/batch.md` to cross-reference the contract.

- [x] **Step 0.2 — Record personalization migration boundary** ✅ 2026-03-28

1. Add a section to `docs/guide/personalization.md` titled "Migration boundary":
   - Schema-v1 bundles: `user_id` in `FEATURE_COLUMNS`, must be passed correctly.
   - Schema-v2 bundles (future): `user_id` removed from model, personalization via calibrators.
   - Gate: schema hash prevents mixing contracts.
2. No code changes.

Verification: review docs only; no tests needed for Phase 0.

---

### Phase 1 — Remove correctness mismatches

- [x] **Step 1.1 — Fix online numeric missing-value handling (Risk 0)** ✅ 2026-03-28

_Problem_: `OnlinePredictor._encode_value()` returns `float("nan")` for missing numerics, but training and batch inference use `fillna(0)`.

Files to modify:
- `src/taskclf/infer/online.py`

Edit:

In `OnlinePredictor._encode_value()` (line ~101), change the null-numeric branch:

```python
# BEFORE
return float(value) if value is not None else float("nan")

# AFTER
return float(value) if value is not None else 0.0
```

Tests to write/update:
- `tests/test_integration_train_infer_parity.py` (new file) — TSP-001, TSP-002, TSP-003, P0-001, P0-002
  - TSP-001: Build a `FeatureRow` with `None` numeric fields. Pass it through both `prepare_xy` (train path) and `OnlinePredictor._encode_value` (online path). Assert both produce `0.0`.
  - TSP-002: Run `predict_proba` (batch) and `predict_bucket` (online) on the same input. Assert `np.allclose(batch_proba, online_proba)`.
  - TSP-003: Parametrize over every numeric column in `FEATURE_COLUMNS`. For each, set it to `None` in a `FeatureRow`, encode via both paths, assert equal.
  - P0-001/P0-002: Trace the pipeline stages for batch, online, and tray paths. Assert same order: encode → impute → predict → calibrate → reject → smooth.

Existing test to update:
- `tests/test_infer_online.py` — find any test asserting `NaN` behavior for missing numerics (e.g. `test_numerical_none_returns_nan`) and update it to assert `0.0` instead.

Docs to update:
- `docs/api/infer/online.md` — update `_encode_value` docstring note about null handling.

- [x] **Step 1.2 — Pass stable user_id through tray suggestion path (Risk 4)** ✅ 2026-03-28

_Problem_: `_LabelSuggester.suggest()` calls `build_features_from_aw_events(events)` without passing a `user_id`, so it defaults to `"default-user"`. The model expects the config-backed stable UUID.

Files to modify:
- `src/taskclf/ui/tray.py`

Edit 1 — Add `_user_id` field to `_LabelSuggester`:

The suggester needs access to the stable user ID. Add it as a field and populate it during construction:

In `_LabelSuggester.__post_init__()`, after loading the model, read the user_id from config:

```python
# In __post_init__, after self._predictor is set:
self._user_id: str = "default-user"  # overridden by TrayLabeler
```

In `_LabelSuggester.from_policy()`, after `obj._title_salt = DEFAULT_TITLE_SALT`:

```python
obj._user_id = "default-user"  # overridden by TrayLabeler
```

Edit 2 — In `TrayLabeler.__post_init__()`, after setting `self._suggester._title_salt`, also set:

```python
self._suggester._user_id = self._config.user_id
```

Do this in both places where `_suggester` is created (the `from_policy` path around line 718 and the direct-construction fallback around line 733).

Also do this in `_on_model_trained()` and `_reload_model()` and `_switch_model()` wherever a new `_suggester` is assigned.

Edit 3 — In `_LabelSuggester.suggest()`, pass `user_id` to `build_features_from_aw_events`:

```python
# BEFORE
rows = build_features_from_aw_events(events)

# AFTER
rows = build_features_from_aw_events(events, user_id=self._user_id)
```

Tests to write:
- `tests/test_ui_tray_suggest.py` (new file) — UID-001, UID-003
  - UID-001: Mock `build_features_from_aw_events`, create a `_LabelSuggester` with a known `_user_id`, call `suggest()`, assert the mock was called with `user_id=` matching the config value.
  - UID-003: Create a `_LabelSuggester` without setting `_user_id`. Assert it falls back to `"default-user"` (no crash).

Existing test to extend:
- `tests/test_infer_online.py` — UID-002: Assert `calibrator_store.get_calibrator()` is called with the row's `user_id`, not `"default-user"`.

Docs to update:
- `docs/api/ui/labeling.md` — note that `_LabelSuggester` now propagates the stable config user_id.

- [x] **Step 1.3 — Add input events to tray suggestion path (Risk 3)** ✅ 2026-03-28

_Problem_: `_LabelSuggester.suggest()` only fetches window events. Input-derived features (keyboard, mouse) are all `None`.

Files to modify:
- `src/taskclf/ui/tray.py`

Edit — In `_LabelSuggester.suggest()`, add input event fetching after the window event fetch:

```python
# Add import at the top of the method (inside the existing lazy import block):
from taskclf.adapters.activitywatch.client import (
    fetch_aw_events,
    fetch_aw_input_events,
    find_input_bucket_id,
    find_window_bucket_id,
)

# After fetching window events and checking `if not events: return None`:
input_events = None
try:
    input_bucket_id = find_input_bucket_id(self._aw_host)
    if input_bucket_id:
        input_events = fetch_aw_input_events(
            self._aw_host, input_bucket_id, start, end
        ) or None
except Exception:
    logger.debug("Could not fetch input events for suggestion", exc_info=True)

# Update the build call:
rows = build_features_from_aw_events(
    events,
    user_id=self._user_id,
    input_events=input_events,
)
```

Tests to write/update:
- `tests/test_ui_tray_suggest.py` — INP-001
  - INP-001: Mock `fetch_aw_input_events` and `find_input_bucket_id`. Call `suggest()`. Assert the mock was called and `input_events` was passed to `build_features_from_aw_events`.

- `tests/test_features_build.py` — INP-002, INP-003
  - INP-002: Call `build_features_from_aw_events` with `input_events` provided. Assert `keys_per_min`, `clicks_per_min`, etc. are not `None`.
  - INP-003: Call `build_features_from_aw_events` without `input_events`. Assert those fields are `None`.

Docs to update:
- `docs/api/ui/labeling.md` — note input event support in tray suggestions.

- [x] **Step 1.4 — Implement unknown-category handling (Decision 5)** ✅ 2026-03-28

_Problem_: Unseen categorical values map to `-1` at inference, a code never seen during training. Decision #5 requires explicit `__unknown__` training.

Files to modify:
- `src/taskclf/train/lgbm.py`
- `src/taskclf/infer/online.py`
- `src/taskclf/core/model_io.py`

Edit 1 — Extend `encode_categoricals()` in `train/lgbm.py`:

Add two new parameters: `min_category_freq: int = 5` and `unknown_mask_rate: float = 0.05`.

When `cat_encoders is None` (training mode):
1. For each categorical column, count value frequencies.
2. Replace values with count < `min_category_freq` with `"__unknown__"`.
3. Randomly mask `unknown_mask_rate` fraction of remaining known values to `"__unknown__"` (use a seed for reproducibility).
4. Fit the `LabelEncoder` on the result (which now includes `"__unknown__"` as a class).

When `cat_encoders is not None` (inference mode):
1. Map unseen values to `"__unknown__"` instead of `-1`.
2. Transform via the fitted encoder (which knows `"__unknown__"`).

```python
def encode_categoricals(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    *,
    min_category_freq: int = 5,
    unknown_mask_rate: float = 0.05,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
```

The full implementation:
- During training (`cat_encoders is None`):
  - For each col in `CATEGORICAL_COLUMNS`:
    - `vals = df[col].astype(str)`
    - `freq = vals.value_counts()`
    - `rare_mask = vals.isin(freq[freq < min_category_freq].index)`
    - `vals[rare_mask] = "__unknown__"`
    - If `unknown_mask_rate > 0`: randomly set `unknown_mask_rate` fraction of non-`__unknown__` rows to `"__unknown__"` using `np.random.RandomState(random_state)`.
    - Fit `LabelEncoder` on the result.
- During inference (`cat_encoders is not None`):
  - For each col: map values not in `le.classes_` to `"__unknown__"` (which IS in `le.classes_`).
  - Transform normally.

Edit 2 — Update `OnlinePredictor._encode_value()` in `infer/online.py`:

```python
# BEFORE
return -1.0

# AFTER (for unseen categoricals):
unknown_code = "__unknown__"
if le is not None and unknown_code in set(le.classes_):
    return float(le.transform([unknown_code])[0])
return -1.0  # fallback for legacy encoders without __unknown__
```

Edit 3 — Extend `ModelMetadata` in `core/model_io.py`:

Add optional fields:
```python
unknown_category_freq_threshold: int | None = None
unknown_category_mask_rate: float | None = None
```

Edit 4 — In `train_lgbm()` and `save_model_bundle()`, persist the threshold and mask rate in metadata.

Tests to write/update:
- `tests/test_train_lgbm.py` — UNK-001, UNK-002, UNK-003, PER-001
  - UNK-001: Train with `min_category_freq=5`. Assert categories with < 5 occurrences become `"__unknown__"` in the encoded output.
  - UNK-002: Train with `unknown_mask_rate=0.05`, fixed seed. Assert approximately 5% of known-category rows are masked.
  - UNK-003: After fitting, assert `"__unknown__"` is in `le.classes_` for every categorical encoder.
  - PER-001: Assert `"user_id"` is in `FEATURE_COLUMNS` for schema v1.

- `tests/test_infer_online.py` — UNK-004
  - UNK-004: Create an `OnlinePredictor` with encoders containing `"__unknown__"`. Call `_encode_value("app_id", "never-seen-app")`. Assert it returns the `__unknown__` code, not `-1`.

- `tests/test_integration_train_infer_unknown.py` (new file) — UNK-005, UNK-006, EXP-F
  - UNK-005: Train a model, predict on rows with withheld categories. Assert mean max-probability for unknown rows < known rows.
  - UNK-006: Assert reject rate on withheld categories > reject rate on known categories.
  - EXP-F: Run the 4-condition comparison from Experiment F. Assert results contain macro-F1 and reject rate for each condition.

- `tests/test_core_model_io.py` — UNK-007
  - UNK-007: Save a bundle with threshold and mask rate. Load it. Assert `metadata.unknown_category_freq_threshold` and `metadata.unknown_category_mask_rate` are present.

Docs to update:
- `docs/api/train/lgbm.md` — document new parameters on `encode_categoricals`.
- `docs/api/core/model_io.md` — document new metadata fields.
- `docs/api/infer/online.md` — document `__unknown__` handling in `_encode_value`.

---

### Phase 2 — Migrate personalization out of the core model

**Prerequisites**: Phase 1 complete. All current tests passing.

- [x] **Step 2.1 — Define schema v2 without user_id** ✅ 2026-03-28

Files to modify:
- `src/taskclf/train/lgbm.py`
- `src/taskclf/core/schema.py`

Edit:
1. Add `FEATURE_COLUMNS_V2` and `CATEGORICAL_COLUMNS_V2` constants in `train/lgbm.py` that exclude `"user_id"`.
2. Add `FeatureSchemaV2` class in `core/schema.py` with the updated field list and a new `SCHEMA_HASH`.
3. Do NOT remove the v1 constants or schema class.
4. Add a helper `get_feature_columns(schema_version: str) -> list[str]` that dispatches by version.

Tests:
- `tests/test_train_lgbm.py` — P2-001: Assert `"user_id"` is not in `FEATURE_COLUMNS_V2`.
- `tests/test_core_schema.py` — P6-002 (adapted): Assert `FeatureSchemaV2.SCHEMA_HASH != FeatureSchemaV1.SCHEMA_HASH`.

- [x] **Step 2.2 — Side-by-side evaluation** ✅ 2026-03-28

This step is an experiment, not a production code change.

1. Train three models:
   - Model A: schema v1 with `user_id` in features (current).
   - Model B: schema v2 without `user_id`, no per-user calibration.
   - Model C: schema v2 without `user_id`, with per-user calibration.
2. Evaluate all three on the same time-based test split.
3. Compare macro-F1, per-user F1, reject rate, and calibration quality.
4. Record results in `artifacts/experiments/personalization_migration/`.

Tests:
- `tests/test_integration_train_infer.py` — P2-002, P2-003, P2-004
  - P2-002: Smoke test: schema-v2 model + calibrator produces metrics.
  - P2-003: Attempt to load a v1 bundle with a v2 feature vector. Assert schema hash mismatch raises.
  - P2-004: Reverse direction.

- [x] **Step 2.3 — Extend personalization post-processing** ✅ 2026-03-28

Files to modify:
- `src/taskclf/infer/resolve.py`

Edit:
1. Add `per_user_reject_thresholds: dict[str, float] | None` to `ResolvedInferenceConfig`.
2. In `resolve_inference_config()`, load per-user thresholds from the inference policy if present.
3. In `OnlinePredictor.predict_bucket()`, apply per-user threshold when available.

Tests:
- `tests/test_infer_resolve.py` — PER-003: Assert per-user threshold overrides global.

- [x] **Step 2.4 — Keep backward compatibility** ✅ 2026-03-28

No code changes needed beyond what was done in steps 2.1–2.3. Verify:
- Schema-v1 bundles still load and receive `user_id`.
- Schema-v2 bundles refuse `user_id` in the feature vector.
- Both can coexist in `models/` without conflict.

Docs to update:
- `docs/guide/personalization.md` — update with the schema-v2 migration details.

---

### Phase 3 — Strengthen live context

**Prerequisites**: Phase 1 complete.

- [x] **Step 3.1 — Add persistent online feature state** ✅ 2026-03-28

_Problem_: The online loop reconstructs feature state from each narrow poll slice. Rolling features (5m switch counts, 15m switch counts, rolling keyboard/mouse stats, delta features) may be truncated.

Files to create:
- `src/taskclf/infer/feature_state.py` (new module)

Design:
1. Create a `OnlineFeatureState` class that maintains a circular buffer of recent `FeatureRow` values (at least 15 minutes of buckets).
2. It exposes methods like `push(row: FeatureRow)` and `get_context() -> dict` that return rolling aggregates.
3. The online loop feeds each new row into this state before prediction.

Files to modify:
- `src/taskclf/infer/online.py`

Edit:
1. In `run_online_loop()`, create an `OnlineFeatureState` before the poll loop.
2. After building `rows` from `build_features_from_aw_events()`, push each row into the state.
3. Before calling `predict_bucket()`, overlay the state's rolling aggregates onto the row so that features like `app_switch_count_last_15m` reflect the full 15-minute window.

Tests:
- `tests/test_infer_online.py` — CTX-001, CTX-002, CTX-003
  - CTX-001: Push 16+ buckets into `OnlineFeatureState`. Assert `app_switch_count_15m` reflects the full 15-minute window.
  - CTX-002: Assert `delta_*` fields are non-zero on the second bucket.
  - CTX-003: Simulate an idle gap. Assert `session_length_minutes` resets.

Docs to create:
- `docs/api/infer/feature_state.md` (new page for the new module).

---

### Phase 4 — Fix tray suggestion quality and implement surface architecture

**Prerequisites**: Phase 1 complete.

- [x] **Step 4.1 — Implement interval-aware aggregation** ✅ 2026-03-28

_Problem_: `_LabelSuggester.suggest()` predicts only `rows[-1]`. This throws away context from the rest of the interval.

Files to create:
- `src/taskclf/infer/aggregation.py` (new module)

Design the aggregation module with these functions:

```python
def majority_vote(labels: list[str]) -> str: ...
def confidence_weighted_vote(
    labels: list[str], confidences: list[float]
) -> str: ...
def highest_total_probability(
    proba_matrix: np.ndarray, label_names: list[str]
) -> str: ...
def aggregate_interval(
    predictions: list[WindowPrediction],
    strategy: str = "majority",
) -> tuple[str, float]: ...
```

Files to modify:
- `src/taskclf/ui/tray.py`

Edit `_LabelSuggester.suggest()`:

```python
# BEFORE: predict only rows[-1]
prediction = self._predictor.predict_bucket(rows[-1])
return (prediction.core_label_name, prediction.confidence)

# AFTER: predict all rows, aggregate
predictions = [self._predictor.predict_bucket(row) for row in rows]
from taskclf.infer.aggregation import aggregate_interval
label, confidence = aggregate_interval(predictions, strategy="majority")
return (label, confidence)
```

Tests:
- `tests/test_infer_aggregation.py` (new file) — AGG-001 through AGG-004, P4-001, EXP-B
  - AGG-001: Aggregation over N buckets uses all N, not just the last.
  - AGG-002: 3x Coding, 2x Writing → majority vote returns "Coding".
  - AGG-003: High-confidence minority wins confidence-weighted vote.
  - AGG-004: Single-bucket interval returns same result as direct prediction.
  - P4-001: Different strategies produce different results on mixed input.
  - EXP-B: Compare 3+ strategies, assert per-strategy accuracy is returned.

Docs to create:
- `docs/api/infer/aggregation.md` (new page).

- [x] **Step 4.2 — Include input events in tray suggestion features** ✅ 2026-03-28

Already done in Phase 1 Step 1.3. Verify test P4-002 passes:
- `tests/test_ui_tray_suggest.py` — P4-002: After fix, tray feature rows have non-None input fields.

- [x] **Step 4.3 — Separate transition suggestion and live status surfaces (Decision 6)** ✅ 2026-03-28

Files to modify:
- `src/taskclf/ui/tray.py`

Edit 1 — Create a centralized copy-string module:

Create `src/taskclf/ui/copy.py` (new file):

```python
"""Centralized user-facing copy strings for all UI surfaces."""

def transition_suggestion_text(label: str, start: str, end: str) -> str:
    return f"Was this {label}? {start}\u2013{end}"

def live_status_text(label: str) -> str:
    return f"Now: {label}"

def gap_fill_prompt(duration_str: str) -> str:
    return f"You have {duration_str} unlabeled. Review?"

def gap_fill_detail(start: str, end: str) -> str:
    return f"Review unlabeled: {start}\u2013{end}"
```

Edit 2 — Refactor `_handle_transition()` in `TrayLabeler`:

- Import and use `transition_suggestion_text()` for the notification and EventBus `prompt_label` payload.
- Do NOT include numeric confidence in the user-facing message.
- Include the concrete time range in the prompt.

Edit 3 — Add a `_publish_live_status()` method to `TrayLabeler`:

- Called from `_handle_poll()`.
- If a model is loaded, predict the current bucket and publish a `live_status` event with `live_status_text()`.
- This is separate from the transition/suggestion flow.

Edit 4 — Update `_send_notification()` to use `transition_suggestion_text()` and exclude confidence from the user-facing notification body.

Tests:
- `tests/test_ui_tray_surfaces.py` (new file) — SEM-002, SRF-001 through SRF-004, SRF-008, CNF-004
  - SRF-001: Transition notification text matches `"Was this {label}? {start}–{end}"`.
  - SRF-002: Live status text matches `"Now: {label}"`.
  - SRF-003: `_LabelSuggester` and live-status are separate methods/code paths.
  - SRF-004: Notification payload does not contain numeric confidence.
  - SRF-008: All user-facing strings are imported from `ui/copy.py`.
  - CNF-004: Settings schema does not expose `reject_threshold`.
  - SEM-002: Live status uses only the latest bucket.

Docs to create:
- `docs/api/ui/copy.md` (new page for centralized copy strings).

Docs to update:
- `docs/api/ui/labeling.md` — document the surface separation.

---

### Phase 4b — Gap-fill surface

**Prerequisites**: Phase 4 complete (transition suggestions and live status working).

- [x] **Step 4b.1 — Passive unlabeled-time indicator** ✅ 2026-03-28

Files to modify:
- `src/taskclf/ui/tray.py`

Edit:
1. Add an `_unlabeled_minutes` tracked field to `TrayLabeler`.
2. In `_handle_poll()`, compute total unlabeled time since the last confirmed label (query the label store).
3. Publish an `unlabeled_time` event via `EventBus` with the total duration.
4. The frontend renders this as a badge/counter (frontend change, not covered here).

- [x] **Step 4b.2 — Contextual gap-fill prompting** ✅ 2026-03-28

Files to modify:
- `src/taskclf/ui/tray.py`

Edit:
1. In `ActivityMonitor`, add idle-return detection: when transitioning from idle (>5 min) back to active, publish a `gap_fill_prompt` event.
2. On session start (first poll after `ActivityMonitor` starts), if unlabeled time exists, publish a `gap_fill_prompt` event.
3. In `_handle_transition()`, immediately after the user accepts a transition suggestion (detected via EventBus feedback), if adjacent unlabeled time exists, publish a `gap_fill_prompt` event.

- [x] **Step 4b.3 — Escalation threshold** ✅ 2026-03-28

Files to modify:
- `src/taskclf/ui/tray.py`

Edit:
1. Add a configurable `gap_fill_escalation_minutes` (default: one active day = 480 minutes) to `TrayLabeler`.
2. In the poll loop, if unlabeled minutes exceed this threshold, publish a `gap_fill_escalated` event.
3. The tray icon changes state (e.g., color change). No popup.

Tests:
- `tests/test_ui_tray_gap_fill.py` (new file) — SEM-003, SRF-005 through SRF-007, P4b-001 through P4b-004
  - SEM-003: Gap-fill interval spans from last label end to current time.
  - SRF-005: Badge text includes unlabeled time duration.
  - SRF-006: Prompt fires only at idle return, session start, or post-acceptance.
  - SRF-007: Escalation changes tray icon state when threshold exceeded.
  - P4b-001: Idle return (>5 min) triggers gap-fill prompt event.
  - P4b-002: New session with unlabeled time triggers prompt.
  - P4b-003: After accepting transition suggestion with adjacent gap, prompt fires.
  - P4b-004: Escalation does not call the notification API (no popup).

---

### Phase 5 — Align evaluation with deployed behavior

**Prerequisites**: Phase 1 complete.

- [x] **Step 5.1 — Add operational evaluation modes** ✅ 2026-03-28

Files to modify:
- `src/taskclf/train/evaluate.py`

Edit:
1. Extend `evaluate_model()` (or create `evaluate_model_operational()`) to accept evaluation mode parameters:
   - `eval_mode: Literal["raw", "calibrated", "calibrated_reject", "smoothed", "interval"]`
2. For `"calibrated"`: apply the calibrator to raw probas before computing metrics.
3. For `"calibrated_reject"`: apply calibrator + reject threshold.
4. For `"smoothed"`: apply calibrator + reject + rolling majority smoothing, then compute metrics.
5. For `"interval"`: aggregate bucket predictions into interval-level labels, then compute interval accuracy.
6. Add flip rate, segment duration distribution, and reject rate to `EvaluationReport`.

Tests:
- `tests/test_train_evaluate.py` — EVL-001, EVL-002, EVL-003, P5-002, MSR-003, MSR-006
  - EVL-001: Report includes flip rate, segment duration distribution, reject rate.
  - EVL-002: Smoothed macro-F1 >= raw macro-F1.
  - EVL-003: Interval-level evaluation produces per-interval accuracy.
  - P5-002: Raw vs calibrated evaluation modes produce different F1 values.
  - MSR-003: Two evaluation runs produce comparable metric dicts.
  - MSR-006: Withhold one category; measure reject rate difference.

- [x] **Step 5.2 — Tune reject threshold on calibrated scores** ✅ 2026-03-28

Files to modify:
- `src/taskclf/train/evaluate.py`

Edit:
1. Extend `tune_reject_threshold()` to accept an optional `calibrator` parameter.
2. When provided, calibrate the raw probabilities before sweeping thresholds.
3. Optimize for precision over recall (decision #4).

Files to modify:
- `src/taskclf/cli/main.py`

Edit:
1. Add `--write-policy` flag to `taskclf train tune-reject`.
2. When set, call `save_inference_policy()` with the tuned threshold.

Tests:
- `tests/test_train_evaluate_reject.py` — REJ-003, P5-001, EXP-C
  - REJ-003: Sweep results differ between raw and calibrated inputs.
  - P5-001: `--write-policy` persists threshold in inference policy.
  - EXP-C: Raw and calibrated tuning produce different `best_threshold` values.

- [x] **Step 5.3 — Track suggestions per active day** ✅ 2026-03-28

Files to modify:
- `src/taskclf/core/telemetry.py`

Edit:
1. Add a `suggestions_per_day` metric to the telemetry snapshot.
2. In the online loop and tray, publish suggestion events via `EventBus`.
3. Telemetry aggregates them per active day.
4. When a loaded model produces zero suggestions for a full active day, log a warning.

Tests:
- `tests/test_core_telemetry.py` — CNF-002, CNF-003, P5-003
  - CNF-002: After N predictions with K accepted, telemetry shows K suggestions.
  - CNF-003: Loaded model + full active day + zero suggestions → warning logged.
  - P5-003: Simulated active day produces > 0 suggestions.

Docs to update:
- `docs/api/core/telemetry.md` — document the new metric.
- `docs/api/train/evaluate.md` — document operational evaluation modes.

---

### Phase 6 — Improve features for LightGBM

**Prerequisites**: Phases 1 and 5 complete (correct pipeline, proper evaluation).

- [ ] **Step 6.1 — Add candidate features one group at a time**

For each candidate feature group:

1. Add the feature to `FeatureRow` in `core/types.py` (with appropriate type annotation and `None` default).
2. Add the field to `FeatureSchemaV1` (or `V2`) in `core/schema.py`. The schema hash will change automatically.
3. Implement computation in the appropriate `features/` module:
   - `features/build.py` for per-bucket features.
   - `features/sessions.py` for session-derived features.
   - `features/windows.py` for app-window features.
4. Add the column name to `FEATURE_COLUMNS` in `train/lgbm.py`.
5. Add a unit test in the relevant `tests/test_features_*.py` file.
6. Verify the schema hash changed (`tests/test_core_schema.py`).
7. Verify privacy safety (`tests/test_security_privacy.py`).
8. Run the full evaluation to measure impact. Only keep the feature if it improves or maintains quality.

  - [x] app_dwell_time_seconds ✅ 2026-03-29
  - [x] app_entropy_5m, app_entropy_15m ✅ 2026-03-29

Candidate feature priority order (add one group, evaluate, then decide on the next):
1. `app_dwell_time_seconds` — how long the dominant app has been foreground continuously.
2. `app_entropy_5m`, `app_entropy_15m` — Shannon entropy of app distribution over 5/15 minute windows.
3. `top2_app_concentration` — combined time share of the two most-used apps in the interval.
4. `idle_return_indicator` — boolean, True if this bucket immediately follows an idle gap.
5. `browser_time_share`, `editor_time_share`, `terminal_time_share`, `meeting_time_share` — category shares over the interval.
6. `stability_score` — fraction of recent buckets with the same dominant app.

Tests per feature group:
- `tests/test_features_*.py` — P6-001: Feature builder produces expected values.
- `tests/test_core_schema.py` — P6-002: Schema hash changes.
- `tests/test_security_privacy.py` — P6-003: No raw keystrokes/titles stored.

Docs:
- Update `docs/api/features/build.md` for each new feature.

---

### Phase 7 — Tune only after the pipeline is correct

**Prerequisites**: All prior phases complete.

- [ ] **Step 7.1 — Hyperparameter tuning**

This is an experiment step, not a production code change.

1. Use `optuna` or manual grid search over:
   - `num_leaves`: [15, 31, 63, 127]
   - `min_data_in_leaf`: [5, 10, 20, 50]
   - `feature_fraction`: [0.6, 0.8, 1.0]
   - `bagging_fraction`: [0.6, 0.8, 1.0]
   - `lambda_l1`: [0, 0.1, 1.0]
   - `lambda_l2`: [0, 0.1, 1.0]
   - `learning_rate` + `num_boost_round`: [(0.1, 200), (0.05, 400), (0.01, 1000)]
2. Evaluate on time-based validation split (by day/week per `split_by_time()`).
3. Compare against the champion model using the regression gates in `retrain.py`.
4. If better, save as the new champion bundle.
5. Record results and parameters in `artifacts/experiments/hyperparameter_tuning/`.

No new tests needed for tuning itself. Verify existing regression gates pass.

---

## Test Plan

Comprehensive tests for every quality risk, recorded decision, and implementation phase in this document. Each test case has an ID, description, target file, and key assertions.

### Test naming convention

All test files follow `test_{package}_{module}[_{domain}].py` — structured from most common denominator (package) to most specific (domain concern):

- `{package}` = top-level subpackage: `infer`, `train`, `core`, `features`, `labels`, `ui`, `adapters`, `cli`, `report`, `migrate`
- `{module}` = source file name (without `.py`)
- `{domain}` = optional suffix narrowing to a specific concern within the module

Cross-module tests use `test_integration_{description}[_{domain}].py`.

Examples from the existing codebase:

- `test_infer_batch_reject.py` → `infer/batch.py`, reject domain
- `test_infer_batch_segments.py` → `infer/batch.py`, segments domain
- `test_core_metrics.py` → `core/metrics.py`
- `test_ui_server.py` → `ui/server.py`

### Prerequisite: rename existing test files (done)

All prerequisite test file renames have been completed. The following renames were executed:

| Old name | New name | Source file |
|---|---|---|
| `test_tray.py` | `test_ui_tray.py` | `ui/tray.py` |
| `test_inference_policy.py` | `test_core_inference_policy.py` | `core/inference_policy.py` |
| `test_telemetry.py` | `test_core_telemetry.py` | `core/telemetry.py` |
| `test_tune_reject.py` | `test_train_evaluate_reject.py` | `train/evaluate.py` |
| `test_drift.py` | `test_core_drift.py` | `core/drift.py` |
| `test_retrain.py` | `test_train_retrain.py` | `train/retrain.py` |
| `test_report.py` | `test_report_daily.py` | `report/daily.py`, `report/export.py` |
| `test_label_now.py` | `test_labels_label_now.py` | `labels/queue.py`, `labels/store.py` |
| `test_calibration.py` | `test_train_calibrate.py` | `train/calibrate.py` |
| `test_monitor.py` | (deleted — consolidated into `test_infer_monitor.py`) | `infer/monitor.py` |

### Test file to source file map

Every test file maps to the source file it exercises. When a source file has multiple distinct domains under test, the file is split by domain suffix.

Existing test files to extend (already correctly named):

- `tests/test_train_lgbm.py` → `train/lgbm.py`
- `tests/test_train_evaluate.py` → `train/evaluate.py`
- `tests/test_infer_online.py` → `infer/online.py`
- `tests/test_infer_batch_reject.py` → `infer/batch.py`, reject domain
- `tests/test_infer_smooth.py` → `infer/smooth.py`
- `tests/test_infer_calibration.py` → `infer/calibration.py`
- `tests/test_infer_resolve.py` → `infer/resolve.py`
- `tests/test_core_model_io.py` → `core/model_io.py`
- `tests/test_core_schema.py` → `core/schema.py`
- `tests/test_features_build.py` → `features/build.py`
- `tests/test_security_privacy.py` → cross-cutting privacy invariants

Existing test files to extend (after prerequisite rename):

- `tests/test_core_inference_policy.py` (was `test_inference_policy.py`) → `core/inference_policy.py`
- `tests/test_core_telemetry.py` (was `test_telemetry.py`) → `core/telemetry.py`
- `tests/test_train_evaluate_reject.py` (was `test_tune_reject.py`) → `train/evaluate.py`, reject-tuning domain

New domain-split test files for `ui/tray.py`:

`test_ui_tray.py` (renamed from `test_tray.py`) already has 31 test classes spanning many domains. New tests go into domain-specific files:

- `tests/test_ui_tray_suggest.py` → `ui/tray.py`, suggestion domain (`_LabelSuggester.suggest`, input events, user_id propagation)
- `tests/test_ui_tray_surfaces.py` → `ui/tray.py`, surface architecture domain (live status vs transition display, copy conventions, confidence hiding)
- `tests/test_ui_tray_gap_fill.py` → `ui/tray.py`, gap-fill domain (passive indicator, contextual prompting, escalation)

New domain-split test files for cross-module integration:

`test_integration_train_infer.py` currently has 3 test classes covering schema gating. New cross-cutting concerns go into domain-specific integration files:

- `tests/test_integration_train_infer_parity.py` → cross-module, preprocessing parity domain (train vs batch vs online numeric/categorical encoding)
- `tests/test_integration_train_infer_unknown.py` → cross-module, unknown-category domain (end-to-end model behavior on withheld categories)

New test file for planned source file:

- `tests/test_infer_aggregation.py` → `infer/aggregation.py` (planned Phase 4: interval aggregation strategies)

### Risk 0 — Train/serve mismatch on missing numeric values

| ID | Description | Assertions |
|---|---|---|
| TSP-001 | Same FeatureRow with None numerics produces identical feature vectors through `prepare_xy` (train/batch) and `OnlinePredictor._encode_value` (online) | Both paths yield `0.0` for missing numerics (after fix); vectors are element-wise equal |
| TSP-002 | Batch `predict_proba` and online `predict_bucket` on identical input produce identical raw probability arrays | `np.allclose(batch_proba, online_proba)` |
| TSP-003 | Regression guard: if online imputation diverges from training, test fails | Parametrize over each numeric column in `FEATURE_COLUMNS`; assert imputed value matches |

File: `tests/test_integration_train_infer_parity.py`
Existing partial coverage: `test_train_lgbm.py::test_nan_fill` (train side only), `test_infer_online.py::test_numerical_none_returns_nan` (documents current NaN behavior, will need updating after fix)

### Risk 1 — Live rolling context weaker than model expects

| ID | Description | Assertions |
|---|---|---|
| CTX-001 | Persistent online feature state preserves 15-minute history across predict_bucket calls | After 15+ buckets, `app_switch_count_15m` reflects full window, not just latest slice |
| CTX-002 | Delta features are non-zero when history is available | Second bucket's `delta_*` fields differ from first bucket's |
| CTX-003 | Session features reset correctly after idle gap | After idle gap exceeding threshold, `session_length_minutes` resets near zero |

File: `tests/test_infer_online.py` (extend `TestOnlineSessionTracking`)

### Risk 2 — Tray suggestions ignore most of the interval

| ID | Description | Assertions |
|---|---|---|
| AGG-001 | Interval aggregation over N buckets produces a label; last-bucket-only prediction may differ | Aggregated label uses information from all N buckets, not just `rows[-1]` |
| AGG-002 | Majority vote aggregation selects the most frequent label | Given 5 buckets: 3x Coding, 2x Writing → "Coding" |
| AGG-003 | Confidence-weighted vote weights by calibrated confidence | Higher-confidence minority can win over lower-confidence majority |
| AGG-004 | Single-bucket interval degrades gracefully to last-bucket behavior | Aggregation with 1 bucket equals direct prediction |

File: `tests/test_infer_aggregation.py` → planned `infer/aggregation.py`

### Risk 3 — Tray suggestions do not use input events

| ID | Description | Assertions |
|---|---|---|
| INP-001 | `_LabelSuggester.suggest` calls `fetch_aw_input_events` when available | Mock verifies `input_events` kwarg passed to `build_features_from_aw_events` |
| INP-002 | Feature rows built with input events have non-None input fields | `keyboard_events_per_sec`, `mouse_clicks_per_sec`, etc. are populated |
| INP-003 | Feature rows built without input events have None input fields | Backward compatibility when input source is unavailable |

File: INP-001 in `tests/test_ui_tray_suggest.py`; INP-002, INP-003 in `tests/test_features_build.py`

### Risk 4 — Stable user_id not passed through live suggestion feature building

| ID | Description | Assertions |
|---|---|---|
| UID-001 | `_LabelSuggester.suggest` passes config-backed `user_id` to `build_features_from_aw_events` | Mock captures `user_id` kwarg; equals config value, not `"default-user"` |
| UID-002 | OnlinePredictor receives correct `user_id` for per-user calibrator dispatch | `calibrator_store.get_calibrator(row.user_id)` called with stable user_id |
| UID-003 | Fallback to default user_id when config is absent | When no stable user_id configured, "default-user" is used (not crash) |

File: UID-001, UID-003 in `tests/test_ui_tray_suggest.py`; UID-002 in `tests/test_infer_online.py`

### Risk 5 — Reject threshold not tied to calibrated probabilities

| ID | Description | Assertions |
|---|---|---|
| REJ-001 | Batch inference rejects on calibrated (not raw) probabilities when calibrator is provided | Row with raw confidence > threshold but calibrated confidence < threshold is rejected |
| REJ-002 | Online inference rejects on calibrated probabilities | Same row, same result as batch |
| REJ-003 | `tune_reject_threshold` can operate on calibrated scores | Sweep results differ between raw and calibrated inputs |

File: REJ-001 in `tests/test_infer_batch_reject.py`; REJ-002 in `tests/test_infer_online.py`; REJ-003 in `tests/test_train_evaluate_reject.py`

### Risk 6 — Bundle metadata does not define active reject policy

| ID | Description | Assertions |
|---|---|---|
| POL-001 | `train tune-reject --write-policy` persists threshold in inference policy | `load_inference_policy` returns threshold matching tuned value |
| POL-002 | `resolve_inference_config` uses policy threshold as default | Resolved config `reject_threshold` equals policy value |
| POL-003 | CLI `--reject-threshold` override takes precedence over policy | Resolved config uses override, not policy |
| POL-004 | `ModelMetadata.reject_threshold` is advisory, not consumed at runtime | Online/batch paths do not read threshold from metadata when policy exists |

File: POL-001 in `tests/test_core_inference_policy.py`; POL-002, POL-003, POL-004 in `tests/test_infer_resolve.py`

### Risk 7 — Unknown-category handling (decision #5)

| ID | Description | Assertions |
|---|---|---|
| UNK-001 | Frequency thresholding collapses rare categories to `__unknown__` | Categories with < threshold occurrences become `__unknown__` in encoded output |
| UNK-002 | Random masking replaces a fraction of known categories with `__unknown__` during training | With seed, approximately `mask_rate` fraction of known values are masked |
| UNK-003 | `__unknown__` is a first-class encoder value | `le.classes_` contains `"__unknown__"` after fit |
| UNK-004 | Unseen values at inference map to `__unknown__` code, not -1 | `_encode_value("app_id", "never-seen-app")` returns the `__unknown__` code |
| UNK-005 | Model produces lower, broader probabilities for `__unknown__` inputs vs known | Mean max-probability for unknown rows < mean max-probability for known rows |
| UNK-006 | Reject threshold catches unknown-category inputs at higher rate | Reject rate on withheld categories > reject rate on known categories |
| UNK-007 | Masking rate and frequency threshold are recorded in bundle metadata | `metadata.json` contains both hyperparameters |

File: UNK-001 to UNK-003 in `tests/test_train_lgbm.py`; UNK-004 in `tests/test_infer_online.py`; UNK-005, UNK-006 in `tests/test_integration_train_infer_unknown.py`; UNK-007 in `tests/test_core_model_io.py`

### Risk 8 — Evaluation is bucket-level only

| ID | Description | Assertions |
|---|---|---|
| EVL-001 | Evaluation reports operational metrics beyond macro-F1 | Report includes flip rate, segment duration distribution, reject rate |
| EVL-002 | Post-smoothing evaluation differs from pre-smoothing | Smoothed macro-F1 >= raw macro-F1 (smoothing should help or be neutral) |
| EVL-003 | Interval-level suggestion evaluation produces per-interval accuracy | Each interval gets a single predicted label; accuracy computed against gold interval labels |

File: `tests/test_train_evaluate.py`

### Decision 1 — Personalization architecture

| ID | Description | Assertions |
|---|---|---|
| PER-001 | Schema-v1 bundles still receive user_id in feature vector | `FEATURE_COLUMNS` for v1 includes `"user_id"` |
| PER-002 | Per-user calibrator store dispatches correctly | Different user_ids get different calibrator objects from store |
| PER-003 | Per-user reject thresholds (when implemented) override global | User with personalized threshold uses it, others use global |

File: PER-001 in `tests/test_train_lgbm.py`; PER-002 in `tests/test_infer_calibration.py` (partially exists); PER-003 in `tests/test_infer_resolve.py` (when implemented)

### Decision 2 — Canonical runtime threshold location

| ID | Description | Assertions |
|---|---|---|
| THR-001 | Inference policy is the single source of truth for reject threshold | `resolve_inference_config` returns policy threshold when policy exists |
| THR-002 | CLI override > policy > active.json fallback > code default | Parametrize resolution with different combinations; verify precedence |
| THR-003 | Hot-reload swaps threshold alongside model | After `InferencePolicyReloader.check_reload()` with new policy, threshold updates |
| THR-004 | `CalibratorStore` records `model_bundle_id` and `model_schema_hash` | `store.json` contains both fields; `validate_policy` cross-checks them |

File: THR-001 to THR-003 in `tests/test_infer_resolve.py`; THR-004 in `tests/test_infer_calibration.py` (partially exists)

### Decision 3 — Tray suggestion semantics

| ID | Description | Assertions |
|---|---|---|
| SEM-001 | Transition suggestion targets the previous completed block, not current | Time range in suggestion spans `[transition_start, transition_end)`, not current bucket |
| SEM-002 | Live status targets current bucket only | Live status prediction uses only the latest bucket |
| SEM-003 | Gap-fill targets the full unlabeled interval since last confirmed label | Gap-fill interval spans from last label end to current time |

File: SEM-001 in `tests/test_ui_tray_suggest.py`; SEM-002 in `tests/test_ui_tray_surfaces.py`; SEM-003 in `tests/test_ui_tray_gap_fill.py`

### Decision 4 — Suggestion confidence strategy

| ID | Description | Assertions |
|---|---|---|
| CNF-001 | High reject threshold suppresses low-confidence suggestions | With threshold=0.9, most synthetic predictions are suppressed |
| CNF-002 | Suggestions-per-active-day tracking records events | After N predictions with K accepted, telemetry shows K suggestions |
| CNF-003 | Zero-suggestions-per-day triggers warning | Loaded model + full active day + zero suggestions → warning logged |
| CNF-004 | Reject threshold is not user-configurable via UI settings | Settings schema does not expose reject_threshold |

File: CNF-001 in `tests/test_infer_online.py`; CNF-002, CNF-003 in `tests/test_core_telemetry.py`; CNF-004 in `tests/test_ui_tray_surfaces.py`

### Decision 5 — Unknown-category handling

Covered by UNK-001 through UNK-007 above.

### Decision 6 — Tray/UI surface architecture

| ID | Description | Assertions |
|---|---|---|
| SRF-001 | Transition suggestion surface shows action-oriented copy with time range | Notification text matches pattern "Was this {label}? {start}–{end}" |
| SRF-002 | Live status surface shows present-tense label only | Display text matches pattern "Now: {label}" |
| SRF-003 | Transition and live status use separate code paths | `_LabelSuggester` and live-status display are distinct methods/classes |
| SRF-004 | No confidence percentages on transition or live surfaces | Notification payload does not contain numeric confidence |
| SRF-005 | Gap-fill passive indicator shows unlabeled duration | Badge text includes unlabeled time duration |
| SRF-006 | Gap-fill prompts only at idle return, session start, or post-acceptance | Prompt is not triggered at arbitrary times; only at defined trigger points |
| SRF-007 | Escalation when unlabeled time exceeds threshold | Tray icon state changes when unlabeled time > configurable threshold |
| SRF-008 | Copy strings are centralized in one module | All user-facing strings imported from single location |

File: SRF-001 to SRF-004, SRF-008 in `tests/test_ui_tray_surfaces.py`; SRF-005 to SRF-007 in `tests/test_ui_tray_gap_fill.py`

### Phase 0 — Lock the inference contract

| ID | Description | Assertions |
|---|---|---|
| P0-001 | Canonical inference order is: features → encode → impute → predict → calibrate → reject → smooth/aggregate → taxonomy | Each step's output feeds the next; no step is skipped or reordered in any path |
| P0-002 | All three paths (batch, online, tray) follow the same order | Instrument or trace the pipeline stages; assert same sequence |

File: `tests/test_integration_train_infer_parity.py`

### Phase 1 — Remove correctness mismatches

Covered by TSP-001 through TSP-003, INP-001 through INP-003, UID-001 through UID-003, UNK-001 through UNK-007 above.

### Phase 2 — Migrate personalization

| ID | Description | Assertions |
|---|---|---|
| P2-001 | Schema-v2 `FEATURE_COLUMNS` does not include `user_id` | New constant excludes it |
| P2-002 | Schema-v2 model + per-user calibrator matches or exceeds v1 hybrid model quality | Side-by-side evaluation metrics comparison |
| P2-003 | Schema-v1 bundles reject schema-v2 feature vectors | `load_model_bundle` with v2 hash raises on v1 expectations |
| P2-004 | Schema-v2 bundles reject schema-v1 feature vectors | Reverse direction also gated |

File: P2-001 in `tests/test_train_lgbm.py`; P2-002 in `tests/test_integration_train_infer.py`; P2-003, P2-004 in `tests/test_integration_train_infer.py` (extends existing `TestSchemaAlterationRefusesInference`)

### Phase 3 — Strengthen live context

Covered by CTX-001 through CTX-003 above.

### Phase 4 — Fix tray suggestion quality

Covered by AGG-001 through AGG-004, SRF-001 through SRF-004 above. Additional:

| ID | Description | Assertions |
|---|---|---|
| P4-001 | Aggregation strategies produce different results on heterogeneous intervals | Majority vote vs confidence-weighted vote vs highest-probability differ on mixed input |
| P4-002 | Input events included in tray suggestion features | After fix, tray feature rows have non-None input fields |

File: P4-001 in `tests/test_infer_aggregation.py`; P4-002 in `tests/test_ui_tray_suggest.py`

### Phase 4b — Gap-fill surface

Covered by SRF-005 through SRF-007 above. Additional:

| ID | Description | Assertions |
|---|---|---|
| P4b-001 | Idle return (>5 min) triggers gap-fill prompt | `ActivityMonitor` idle detection → prompt event published |
| P4b-002 | Session start triggers gap-fill prompt | New session with existing unlabeled time → prompt |
| P4b-003 | Post-acceptance triggers gap-fill prompt | After user accepts transition suggestion, if adjacent gap exists → prompt |
| P4b-004 | No popup on escalation, only icon state change | Escalation does not call notification API |

File: `tests/test_ui_tray_gap_fill.py`

### Phase 5 — Align evaluation with deployed behavior

Covered by EVL-001 through EVL-003 above. Additional:

| ID | Description | Assertions |
|---|---|---|
| P5-001 | Reject threshold tuned on calibrated scores persists via `--write-policy` | CLI command writes policy with calibrated-tuned threshold |
| P5-002 | Evaluation mode: raw vs calibrated shows different metrics | Two evaluation runs with same data, different modes, produce different F1 |
| P5-003 | Suggestions-per-active-day metric is nonzero for a working model | Simulated active day with loaded model produces > 0 suggestions |

File: P5-001 in `tests/test_train_evaluate_reject.py`; P5-002 in `tests/test_train_evaluate.py`; P5-003 in `tests/test_core_telemetry.py`

### Phase 6 — Improve features

| ID | Description | Assertions |
|---|---|---|
| P6-001 | Each new feature has a unit test for computation correctness | Feature builder produces expected values for known inputs |
| P6-002 | Schema hash changes when features are added | New feature → `FeatureSchemaV1.SCHEMA_HASH` differs from previous |
| P6-003 | New features are privacy-safe | No raw keystrokes, no raw titles stored in feature row |

File: P6-001 in relevant `tests/test_features_*.py`; P6-002 in `tests/test_core_schema.py`; P6-003 in `tests/test_security_privacy.py`

### Privacy invariants

| ID | Description | Assertions |
|---|---|---|
| PRV-001 | FeatureRow rejects `raw_keystrokes` field | Construction with raw keystrokes raises validation error |
| PRV-002 | Feature rows never contain raw window titles unless explicitly opted in | Default feature build has `raw_window_title=None` |
| PRV-003 | Persisted feature datasets contain only aggregate rates/counts | Parquet output columns do not include prohibited fields |

Existing coverage: `tests/test_security_privacy.py`, `tests/test_core_schema.py`

### Schema gating

| ID | Description | Assertions |
|---|---|---|
| SCH-001 | Inference refuses to run on schema hash mismatch | `load_model_bundle` with wrong hash raises |
| SCH-002 | Every feature row includes `schema_version` and `schema_hash` | Row construction without these fields raises |
| SCH-003 | Schema hash is deterministic for same field set | Same `FeatureSchemaV1` → same hash across runs |

Existing coverage: `tests/test_integration_train_infer.py::TestSchemaAlterationRefusesInference`, `tests/test_core_schema.py`

### Deterministic artifacts

| ID | Description | Assertions |
|---|---|---|
| DET-001 | Same training data + config + seed produces identical model bundle | Two runs with same inputs → same `metrics.json` values |
| DET-002 | `data/processed/` is reproducible from `data/raw/` + code + configs | Feature build on same raw data → identical parquet output |

File: DET-001 in `tests/test_integration_train_infer.py`; DET-002 in `tests/test_features_build.py`

### Measurement plan validation

| ID | Description | Assertions |
|---|---|---|
| MSR-001 | Flip rate computation is correct | `flap_rate(["A","B","A","B"])` returns expected value |
| MSR-002 | Segment duration distribution is computable from segments | Given segments, histogram of durations is non-empty |
| MSR-003 | Comparison table: raw vs calibrated can be generated | Two evaluation runs produce comparable metric dicts |
| MSR-004 | Comparison table: smoothed vs unsmoothed can be generated | Smoothing changes at least one label in a spiky sequence |
| MSR-005 | Comparison table: with-input vs without-input features differ | Feature rows with and without input events produce different predictions |
| MSR-006 | Known-category vs withheld-category reject rates are computable | Hold out one category; measure reject rate on held-out vs remaining |

File: MSR-001, MSR-002, MSR-004 in `tests/test_infer_smooth.py`; MSR-003, MSR-006 in `tests/test_train_evaluate.py`; MSR-005 in `tests/test_features_build.py`
Existing coverage: `tests/test_infer_smooth.py::TestFlapRate` covers MSR-001

### Experiment validation

Lightweight smoke tests that verify experiment infrastructure can run:

| ID | Description | Assertions |
|---|---|---|
| EXP-A | Train/serve parity experiment produces before/after metrics | Metrics dict has keys for both conditions |
| EXP-B | Interval aggregation experiment compares 3+ strategies | Result contains per-strategy accuracy |
| EXP-C | Calibrated threshold tuning experiment produces threshold on both raw and calibrated | Two `best_threshold` values returned |
| EXP-F | Unknown-category experiment compares 4 conditions | Result contains macro-F1 and reject rate for each condition |

File: EXP-A in `tests/test_integration_train_infer_parity.py`; EXP-B in `tests/test_infer_aggregation.py`; EXP-C in `tests/test_train_evaluate_reject.py`; EXP-F in `tests/test_integration_train_infer_unknown.py`

### Test file index

Consolidates where every test ID lands, grouped by file.

`tests/test_integration_train_infer_parity.py` (new — cross-module preprocessing parity):
TSP-001, TSP-002, TSP-003, P0-001, P0-002, EXP-A

`tests/test_integration_train_infer_unknown.py` (new — cross-module unknown-category):
UNK-005, UNK-006, EXP-F

`tests/test_integration_train_infer.py` (extend existing — schema migration, determinism):
P2-002, P2-003, P2-004, DET-001

`tests/test_train_lgbm.py` (extend existing):
UNK-001, UNK-002, UNK-003, PER-001, P2-001

`tests/test_train_evaluate.py` (extend existing):
EVL-001, EVL-002, EVL-003, P5-002, MSR-003, MSR-006

`tests/test_train_evaluate_reject.py` (renamed from `test_tune_reject.py`):
REJ-003, P5-001, EXP-C

`tests/test_infer_online.py` (extend existing):
CTX-001, CTX-002, CTX-003, UID-002, UNK-004, REJ-002, CNF-001

`tests/test_infer_batch_reject.py` (extend existing):
REJ-001

`tests/test_infer_smooth.py` (extend existing):
MSR-001, MSR-002, MSR-004

`tests/test_infer_calibration.py` (extend existing):
PER-002, THR-004

`tests/test_infer_resolve.py` (extend existing):
THR-001, THR-002, THR-003, POL-002, POL-003, POL-004, PER-003

`tests/test_infer_aggregation.py` (new — planned `infer/aggregation.py`):
AGG-001, AGG-002, AGG-003, AGG-004, P4-001, EXP-B

`tests/test_core_inference_policy.py` (renamed from `test_inference_policy.py`):
POL-001

`tests/test_core_model_io.py` (extend existing):
UNK-007

`tests/test_core_schema.py` (extend existing):
P6-002

`tests/test_core_telemetry.py` (renamed from `test_telemetry.py`):
CNF-002, CNF-003, P5-003

`tests/test_ui_tray_suggest.py` (new — `ui/tray.py`, suggestion domain):
INP-001, UID-001, UID-003, SEM-001, P4-002

`tests/test_ui_tray_surfaces.py` (new — `ui/tray.py`, surface architecture domain):
SEM-002, SRF-001, SRF-002, SRF-003, SRF-004, SRF-008, CNF-004

`tests/test_ui_tray_gap_fill.py` (new — `ui/tray.py`, gap-fill domain):
SEM-003, SRF-005, SRF-006, SRF-007, P4b-001, P4b-002, P4b-003, P4b-004

`tests/test_features_build.py` (extend existing):
INP-002, INP-003, MSR-005, DET-002

`tests/test_security_privacy.py` (extend existing):
PRV-001, PRV-002, PRV-003, P6-003

### Test implementation priorities

The tests should be implemented in dependency order matching the phases:

1. Immediate (Phase 0–1): TSP-001 to TSP-003, UID-001 to UID-003, INP-001 to INP-003
2. Next (Decision #5): UNK-001 to UNK-007
3. Then (Phase 4): AGG-001 to AGG-004, SRF-001 to SRF-008
4. Then (Phase 5): EVL-001 to EVL-003, REJ-001 to REJ-003, P5-001 to P5-003
5. Later (Phase 4b): P4b-001 to P4b-004, SRF-005 to SRF-007
6. Ongoing: PRV-001 to PRV-003, SCH-001 to SCH-003 (many already exist)

---

## Definition Of Done For "Inference Quality v1"

- All inference paths share the same preprocessing contract.
- Current schema-v1 inference paths preserve stable `user_id` where required for compatibility, and the next schema/model generation removes `user_id` from the core model contract.
- Personalization is applied through calibrators and user-specific post-processing rather than identity splits in the long-term base model.
- Suggestion-time inference uses the same feature families as training whenever the data source exists.
- Reject threshold is tuned on calibrated scores to optimize suggestion precision and persisted in the inference policy artifact so it travels with the model+calibration pair.
- Automatic tray suggestions default to the previous completed block rather than the last bucket, while last-minute live status and unlabeled-gap labeling remain explicit separate semantics.
- Live status and transition suggestions are implemented as clearly separate UI surfaces with distinct display and interaction logic (decision #6).
- Transition suggestion copy uses action-oriented framing with concrete time ranges. No confidence percentages are displayed to the user on live or transition surfaces.
- Automatic suggestions prefer precision over recall; coverage gaps are handled by the explicit gap-fill workflow, not by lowering the automatic threshold.
- Gap-fill surface provides a passive unlabeled-time indicator with contextual prompting at idle return, session start, and post-acceptance moments. Gap-fill is never a fixed-schedule automatic notification.
- Suggestions per active day is tracked; a loaded model that produces zero suggestions for a user triggers a warning.
- Unknown categoricals are handled through trained `__unknown__` encoding (decision #5), not inference-time `-1`. Frequency thresholding and random masking are applied during training, and the model produces appropriately uncertain predictions for unseen categories.
- Evaluation includes operational metrics, not only bucket-level macro-F1. Evaluation also includes held-out-category tests confirming that unknown-category confidence triggers the reject mechanism (decision #5).
- Docs and tests cover the new behavior.
