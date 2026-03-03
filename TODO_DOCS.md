# TODO — Documentation Gaps

Missing or thin documentation identified by auditing `docs/api/`
against `src/taskclf/` source modules.

Per `AGENTS.md` policy: every code change must include corresponding
doc updates; new modules, classes, and public functions require API
reference pages; doc drift is treated as a defect.

---

## 1. Thin Doc Pages (title + autodoc only, no narrative)

These pages contain only a heading and an mkdocstrings `:::`
directive. While the auto-generated API reference from docstrings is
useful, the project convention (see `online.md`, `monitor.md`,
`baseline.md`, `retrain.md` for examples) is to include narrative
context, usage examples, field/parameter tables, and architectural
notes above the autodoc directive.

### 1a. `docs/api/infer/calibration.md` — 5 lines
**Source:** `src/taskclf/infer/calibration.py` (327 lines, 7 public
classes/functions)

**What to add:**
- Overview of calibration pipeline: raw model probs → calibrate →
  reject decision
- Table of calibrator types with descriptions:

| Class | Description |
|---|---|
| `IdentityCalibrator` | No-op pass-through (default) |
| `TemperatureCalibrator` | Single-parameter temperature scaling |
| `IsotonicCalibrator` | Per-class isotonic regression |

- `CalibratorStore` section: purpose (per-user + global fallback),
  directory layout (`store.json`, `global.json`, `users/<uid>.json`),
  and usage example
- Serialization section: `save_calibrator` / `load_calibrator` and
  `save_calibrator_store` / `load_calibrator_store` usage
- Note on eligibility thresholds for per-user calibration (cross-ref
  `train calibrate` command)

---

### 1b. `docs/api/infer/prediction.md` — 5 lines
**Source:** `src/taskclf/infer/prediction.py` (45 lines, 1 model)

**What to add:**
- Field table for `WindowPrediction`:

| Field | Type | Constraints | Description |
|---|---|---|---|
| `user_id` | `str` | — | User identifier |
| `bucket_start_ts` | `datetime` | — | Bucket start time |
| `core_label_id` | `int` | `0–7` | Predicted core label index |
| `core_label_name` | `str` | — | Predicted core label name |
| `core_probs` | `list[float]` | length 8, sums to 1.0 | Per-class probabilities |
| `confidence` | `float` | `[0.0, 1.0]` | `max(core_probs)` |
| `is_rejected` | `bool` | — | Below reject threshold |
| `mapped_label_name` | `str` | — | Taxonomy-mapped label |
| `mapped_probs` | `dict[str, float]` | sums to 1.0 | Bucket-level probabilities |
| `model_version` | `str` | — | Schema hash of model bundle |
| `schema_version` | `str` | default `features_v1` | Feature schema version |
| `label_version` | `str` | default `labels_v1` | Label schema version |

- Validation rules: `core_probs` and `mapped_probs` must each sum
  to 1.0 (tolerance 1e-4); model is frozen (immutable after creation)
- Cross-ref to `docs/guide/model_io.md` section 6

---

### 1c. `docs/api/infer/batch.md` — 5 lines
**Source:** `src/taskclf/infer/batch.py` (290 lines, 6 public
functions/classes)

**What to add:**
- Overview: predict → calibrate → reject → smooth → segmentize →
  hysteresis merge
- `BatchInferenceResult` field table:

| Field | Type | Description |
|---|---|---|
| `raw_labels` | `list[str]` | Pre-smoothing labels |
| `smoothed_labels` | `list[str]` | Post-smoothing labels |
| `segments` | `list[Segment]` | Hysteresis-merged segments |
| `confidences` | `np.ndarray` | `max(proba)` per row |
| `is_rejected` | `np.ndarray` | Boolean rejection flags |
| `core_probs` | `np.ndarray` | `(N, 8)` probability matrix |
| `mapped_labels` | `list[str] \| None` | Taxonomy-mapped (if taxonomy provided) |
| `mapped_probs` | `list[dict] \| None` | Per-bucket probs (if taxonomy provided) |

- Functions overview: `predict_proba`, `predict_labels`,
  `run_batch_inference`, `write_predictions_csv`,
  `write_segments_json`, `read_segments_json`
- Usage example:
  ```python
  from taskclf.infer.batch import run_batch_inference, write_segments_json
  result = run_batch_inference(model, features_df, cat_encoders=cat_encoders)
  write_segments_json(result.segments, Path("artifacts/segments.json"))
  ```
- Note on optional taxonomy, calibrator, and calibrator_store params

---

### 1d. `docs/api/infer/smooth.md` — 5 lines
**Source:** `src/taskclf/infer/smooth.py` (254 lines, 4 public
functions + 1 dataclass)

**What to add:**
- `Segment` dataclass field table (`start_ts`, `end_ts`, `label`,
  `bucket_count`)
- `rolling_majority` section: algorithm description (centred
  sliding-window majority vote, tie-breaking preserves original
  label), window size parameter
- `segmentize` section: merges consecutive identical labels into
  `Segment` spans
- `flap_rate` section: definition (`label changes / total windows`),
  acceptance thresholds from `docs/guide/acceptance.md` section 5
  (raw ≤ 0.25, smoothed ≤ 0.15)
- `merge_short_segments` section: hysteresis rule
  (`MIN_BLOCK_DURATION_SECONDS` default 180s), merge strategy
  (same-label neighbour → longer neighbour → preceding on tie)

---

### 1e. `docs/api/infer/taxonomy.md` — 5 lines
**Source:** `src/taskclf/infer/taxonomy.py` (342 lines, 8 public
classes/functions)

**What to add:**
- Overview: core labels → user-defined buckets via YAML config
- Config model hierarchy: `TaxonomyConfig` → `TaxonomyBucket`,
  `TaxonomyDisplay`, `TaxonomyReject`, `TaxonomyAdvanced`
- `TaxonomyResolver` usage example:
  ```python
  config = load_taxonomy(Path("configs/user_taxonomy.yaml"))
  resolver = TaxonomyResolver(config)
  result = resolver.resolve(core_label_id, core_probs)
  ```
- Aggregation modes: `sum` (default) vs `max`
- Fallback bucket: unmapped core labels assigned to `"Other"`
- Reweighting: `advanced.reweight_core_labels` adjusts probs before
  mapping
- Cross-ref to `configs/user_taxonomy_example.yaml`

---

### 1f. `docs/api/train/lgbm.md` — 6 lines
**Source:** `src/taskclf/train/lgbm.py` (232 lines, 5 public
functions + 2 constants)

**What to add:**
- Overview: LightGBM multiclass trainer with class-weight support
- `FEATURE_COLUMNS` constant (32 features): list with descriptions
  of each feature and its dtype (categorical vs numeric)
- `CATEGORICAL_COLUMNS` constant: `app_id`, `app_category`,
  `domain_category`, `user_id` — explain that these are
  label-encoded to integers for LightGBM native categorical support
- `encode_categoricals(df, cat_encoders)` section: two modes
  (fit-new vs reuse), unknown-value mapping to `-1` at inference time
- `prepare_xy(df, label_encoder, cat_encoders)` section: feature
  extraction, NaN fill with 0, label encoding against `LABEL_SET_V1`
- `compute_sample_weights(y, method)` section: `"balanced"` uses
  inverse class frequency (`n_samples / (n_classes * count)`),
  `"none"` returns `None`
- `train_lgbm(train_df, val_df, ...)` section: return value tuple
  `(model, metrics, confusion_df, params, cat_encoders)`, default
  hyperparameters table:

| Parameter | Default | Description |
|---|---|---|
| `objective` | `multiclass` | LightGBM objective |
| `metric` | `multi_logloss` | Evaluation metric |
| `num_leaves` | `31` | Tree complexity |
| `learning_rate` | `0.1` | Step size |
| `num_boost_round` | `100` | Boosting iterations |

- Usage example showing train → evaluate flow

---

### 1g. `docs/api/train/evaluate.md` — 4 lines
**Source:** `src/taskclf/train/evaluate.py` (398 lines, 3 public
functions + 2 models)

**What to add:**
- Overview: full evaluation pipeline producing metrics, calibration
  curves, and acceptance-gate verdicts
- `EvaluationReport` model field table: `macro_f1`, `weighted_f1`,
  `per_class`, `confusion_matrix`, `label_names`, `per_user`,
  `calibration`, `stratification`, `seen_user_f1`, `unseen_user_f1`,
  `reject_rate`, `acceptance_checks`, `acceptance_details`
- Acceptance checks table (from `_check_acceptance` thresholds):

| Check | Threshold | Description |
|---|---|---|
| `macro_f1` | >= 0.65 | Overall macro-F1 |
| `weighted_f1` | >= 0.70 | Overall weighted-F1 |
| `breakidle_precision` | >= 0.95 | BreakIdle precision |
| `breakidle_recall` | >= 0.90 | BreakIdle recall |
| `no_class_below_50_precision` | >= 0.50 | Per-class floor |
| `reject_rate_bounds` | [0.05, 0.30] | Reject rate window |
| `seen_user_f1` | >= 0.70 | Seen-user macro-F1 |
| `unseen_user_f1` | >= 0.60 | Unseen-user macro-F1 |

- `tune_reject_threshold(model, val_df, ...)` section: sweep
  algorithm, `RejectTuningResult` model with `best_threshold` and
  `sweep` table, acceptance-rate bounds constraint
- `write_evaluation_artifacts(report, output_dir)` section: writes
  `evaluation.json`, `calibration.json`, `confusion_matrix.csv`,
  and optional `calibration.png` plot
- Usage example for evaluate → tune → write flow

---

### 1h. `docs/api/train/calibrate.md` — 4 lines
**Source:** `src/taskclf/train/calibrate.py` (268 lines, 4 public
functions + 1 model)

**What to add:**
- Overview: training-side logic for personalization pipeline
  (eligibility → predict → fit global → fit per-user)
- `PersonalizationEligibility` model field table: `user_id`,
  `labeled_windows`, `labeled_days`, `distinct_labels`, `is_eligible`
- `check_personalization_eligible(df, user_id, ...)` section:
  eligibility thresholds from `core.defaults`:

| Threshold | Default | Description |
|---|---|---|
| `min_windows` | `DEFAULT_MIN_LABELED_WINDOWS` | Minimum labeled window count |
| `min_days` | `DEFAULT_MIN_LABELED_DAYS` | Minimum distinct calendar days |
| `min_labels` | `DEFAULT_MIN_DISTINCT_LABELS` | Minimum distinct core labels |

- `fit_temperature_calibrator(y_true_indices, y_proba)` section:
  two-pass grid search (coarse 0.1–5.0 step 0.1, fine ±0.1 step
  0.01), returns `TemperatureCalibrator`
- `fit_isotonic_calibrator(y_true_indices, y_proba, n_classes)`
  section: per-class `IsotonicRegression`, returns
  `IsotonicCalibrator`
- `fit_calibrator_store(model, labeled_df, ...)` section:
  orchestrates the full flow (predict → fit global → check each
  user → fit per-user), returns `(CalibratorStore, list[PersonalizationEligibility])`
- Method comparison: temperature (single scalar, lightweight) vs
  isotonic (per-class non-parametric, more flexible but larger)
- Usage example:
  ```python
  store, reports = fit_calibrator_store(
      model, val_df, cat_encoders=cat_encoders, method="temperature",
  )
  for r in reports:
      print(f"{r.user_id}: eligible={r.is_eligible}")
  ```

---

### 1i. `docs/api/report/daily.md` — 5 lines
**Source:** `src/taskclf/report/daily.py`

**What to add:**
- `build_daily_report()` signature and parameters
- `DailyReport` model field table
- Flap rate metrics included in report
- Usage example

---

### 1j. `docs/api/report/export.md` — 5 lines
**Source:** `src/taskclf/report/export.py`

**What to add:**
- `export_report_json()` function overview
- Sensitive-field redaction rules
- Output format and file naming
- Usage example

---

## 2. Source Modules with No Doc Page

These source files have no corresponding page under `docs/api/`.

### 2a. `src/taskclf/labels/weak_rules.py` — no doc page
**Expected location:** `docs/api/labels/weak_rules.md`

Weak labeling rules module. Should document:
- Available weak label rule functions
- How weak labels interact with gold labels
- `provenance` marking for weak labels
- Configuration/usage examples

---

### 2b. `src/taskclf/ui/window.py` — no doc page
**Expected location:** `docs/api/ui/window.md` or section in
`docs/api/ui/labeling.md`

Desktop window UI module (~280 lines). Should document:

| Symbol | Description |
|---|---|
| `WindowAPI` | JS-bridge class exposed to the SolidJS frontend via `window.pywebview.api`. Methods: `toggle_window`, `show_label_grid`, `hide_label_grid`, `toggle_state_panel`, `bind`/`bind_label`/`bind_panel`. |
| `run_window(port, on_ready, window_api)` | Creates three frameless, always-on-top, draggable pywebview windows (compact pill, label grid, state panel) and starts the GUI loop. Blocks on main thread. |

- Window sizes: compact `(150, 30)`, label grid `(280, 330)`, panel `(280, 520)`
- Integration: used by `cli/main.py` (`ui` command) and `ui/tray.py`

---

### 2c. `src/taskclf/ui/events.py` — no doc page
**Expected location:** `docs/api/ui/events.md` or section in
`docs/api/ui/labeling.md`

Thread-safe asyncio event bus (~60 lines). Should document:

| Symbol | Description |
|---|---|
| `EventBus` | Pub/sub for broadcasting events to WebSocket clients. `publish()` (async), `publish_threadsafe()` (from threads), `subscribe()` (async context manager yielding `asyncio.Queue`), `bind_loop()` (bind to running event loop at startup). |

- Architecture: `ActivityMonitor` (background thread) → `publish_threadsafe` → `EventBus` → `subscribe` → WebSocket `/ws/predictions`
- Queue capacity: 256 per subscriber; full queues are evicted

---

## 3. CLI Commands Missing from `docs/api/cli/main.md`

These commands exist in `src/taskclf/cli/main.py` but are not
documented in the CLI reference.

| Command | Code location | Priority | Notes |
|---|---|---|---|
| `train evaluate` | `main.py:680-829` | High | Fully functional; evaluates model against labeled data with acceptance checks |
| `train tune-reject` | `main.py:832-935` | High | Fully functional; sweeps reject thresholds and recommends optimal value |
| `train calibrate` | `main.py:938-1056` | High | Fully functional; fits per-user probability calibrators |
| `ui` | `main.py:2293-2454` | Medium | Alternative to `tray`; launches browser-based UI with FastAPI server |

Each entry should follow the existing `docs/api/cli/main.md` format:
command name, description, all `--flags` with types/defaults/descriptions,
example invocation, and expected output.

---

## 4. Incomplete Content in `docs/api/ui/labeling.md`

### 4a. Undocumented WebSocket event types
**Doc file:** `docs/api/ui/labeling.md:61-65`
**Source:** `src/taskclf/ui/tray.py:453-483`, `src/taskclf/ui/server.py:387-392`

The doc's WebSocket section lists four event types (`status`,
`tray_state`, `prediction`, `suggest_label`) but the code publishes
two additional types that are not documented:

| Event type | Published by | Payload |
|---|---|---|
| `prompt_label` | `TrayLabeler._handle_transition` (`tray.py:455`) | `prev_app`, `new_app`, `block_start`, `block_end`, `duration_min`, `suggested_label`, `suggested_confidence` |
| `show_label_grid` | `POST /api/window/show-label-grid` (`server.py:391`) | `type: "show_label_grid"` (no other fields) |

**Fix:** Add both event types to the WebSocket event list in
`labeling.md` under the "Architecture" section, after the existing
`suggest_label` entry. Include payload field descriptions.

---

## 5. Cross-Reference Gaps

These docs reference concepts without linking to the relevant pages.

| Doc page | Missing cross-ref | Target |
|---|---|---|
| `docs/api/infer/online.md` | "Label queue integration" mentions `ActiveLabelingQueue` but doesn't link | `docs/api/labels/queue.md` |
| `docs/api/infer/online.md` | "model bundle" doesn't link to model IO docs | `docs/guide/model_io.md` |
| `docs/api/infer/monitor.md` | "pure drift statistics from `core.drift`" not linked | `docs/api/core/drift.md` |
| `docs/api/infer/monitor.md` | "telemetry store" not linked | `docs/api/core/telemetry.md` |
| `docs/api/infer/baseline.md` | Threshold constants reference `core.defaults` but no link | `docs/api/core/defaults.md` |
