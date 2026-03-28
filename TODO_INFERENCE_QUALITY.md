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

1. Create the next schema/model contract that removes `user_id` from `FEATURE_COLUMNS` and categorical encoders.

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
- Evaluation includes operational metrics, not only bucket-level macro-F1.
- Docs and tests cover the new behavior.
