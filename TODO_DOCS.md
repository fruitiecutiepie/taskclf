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

### ~~1a. `docs/api/infer/calibration.md`~~ — DONE

Expanded with calibration pipeline overview, calibrator types table,
`CalibratorStore` section with directory layout, serialization
examples, and eligibility cross-ref.

---

### ~~1b. `docs/api/infer/prediction.md`~~ — DONE

Expanded with `WindowPrediction` field table (12 fields), validation
rules, frozen-model note, and cross-ref to `docs/guide/model_io.md`.

---

### ~~1c. `docs/api/infer/batch.md`~~ — DONE

Expanded with pipeline overview, `BatchInferenceResult` field table,
function-by-function docs, usage example, and notes on optional
taxonomy/calibrator/calibrator_store params.

---

### ~~1d. `docs/api/infer/smooth.md`~~ — DONE

Expanded with `Segment` field table, `rolling_majority` algorithm
description, `segmentize` section, `flap_rate` with acceptance
thresholds, and `merge_short_segments` hysteresis rule.

---

### ~~1e. `docs/api/infer/taxonomy.md`~~ — DONE

Expanded with config model hierarchy (5 models with field tables),
`TaxonomyResolver` usage, aggregation modes, fallback bucket,
reweighting, and I/O helpers.

---

### ~~1f. `docs/api/train/lgbm.md`~~ — DONE

Expanded with pipeline overview, `FEATURE_COLUMNS` table (34 features
with type annotations), `CATEGORICAL_COLUMNS` list, default
hyperparameters table, function-by-function docs (`encode_categoricals`,
`prepare_xy`, `compute_sample_weights`, `train_lgbm` with return-value
table), usage example, and cross-refs to `core.defaults` and
`train.evaluate`.

---

### ~~1g. `docs/api/train/evaluate.md`~~ — DONE

Expanded with evaluation pipeline overview, `EvaluationReport` field
table (13 fields), `RejectTuningResult` field table, acceptance checks
table (8 checks with thresholds), function docs for `evaluate_model`,
`tune_reject_threshold`, and `write_evaluation_artifacts` (with
artifacts table), and usage example showing evaluate → tune → write
flow.

---

### ~~1h. `docs/api/train/calibrate.md`~~ — DONE

Expanded with personalization pipeline overview, `PersonalizationEligibility`
field table, eligibility thresholds table (from `core.defaults`),
function docs for `check_personalization_eligible`,
`fit_temperature_calibrator`, `fit_isotonic_calibrator`, and
`fit_calibrator_store`, temperature-vs-isotonic method comparison
table, usage example, and cross-refs to `infer.calibration` and
`guide/personalization`.

---

### ~~1i. `docs/api/report/daily.md`~~ — DONE

Expanded with aggregation pipeline overview, `ContextSwitchStats` field
table (5 fields), `DailyReport` field table (8 fields),
`build_daily_report` function section with signature/parameter table
and error behavior, usage example, and cross-refs to `infer.smooth`
and `core.defaults`.

---

### ~~1j. `docs/api/report/export.md`~~ — DONE

Expanded with overview of three export formats (JSON, CSV, Parquet),
sensitive-field redaction rules (4 forbidden keys with recursive check),
per-function sections with signatures, CSV/Parquet output schema table,
usage example, and cross-ref to `report.daily`.

---

## 2. Source Modules with No Doc Page

These source files have no corresponding page under `docs/api/`.

### ~~2a. `src/taskclf/labels/weak_rules.py`~~ — DONE

Implemented the module (was a placeholder) and created
`docs/api/labels/weak_rules.md` with overview, `WeakRule` field table,
three built-in rule maps (`APP_ID_RULES`, `APP_CATEGORY_RULES`,
`DOMAIN_CATEGORY_RULES`) with full value tables, function-by-function
docs (`build_default_rules`, `match_rule`, `apply_weak_rules` with
parameter table), provenance convention, usage examples with custom
rules, and cross-refs to `labels.store`, `labels.projection`, and
`core.types.LabelSpan`. Added to `mkdocs.yml` nav.

---

### ~~2b. `src/taskclf/ui/window.py`~~ — DONE

Created `docs/api/ui/window.md` with overview, window sizes table
(3 windows), `WindowAPI` public methods table (8 methods), window
positioning rules, `window_run` function section with parameter table,
macOS stderr redirect note, and integration cross-refs to CLI `ui`
command, `ui.tray`, and `EventBus`. Added to `mkdocs.yml` nav.

---

### ~~2c. `src/taskclf/ui/events.py`~~ — DONE

Created `docs/api/ui/events.md` with overview, architecture flow
diagram, `EventBus` methods table (4 methods), per-method sections
with signatures, queue behaviour notes (capacity 256, eviction on
full), usage example, and cross-refs to `ui.server` and `ui.window`.
Added to `mkdocs.yml` nav.

---

### ~~2d. `src/taskclf/infer/resolve.py`~~ — DONE

Created `docs/api/infer/resolve.md` with overview, resolution
precedence table (4 levels), `ModelResolutionError` attribute table,
`resolve_model_dir` function section with parameter table and
raise/return behaviour, `ActiveModelReloader` class section with
constructor parameter table and `check_reload` return semantics,
usage examples (startup resolve and online hot-reload loop), and
cross-refs to `model_registry`, `core.model_io`, and `infer.online`.
Added to `mkdocs.yml` nav under Inference.

Note: `src/taskclf/adapters/input/macos.py` is a single-line
placeholder (`# optional: HID events aggregator`) and has no
public API to document.

---

## 3. CLI Commands Missing from `docs/api/cli/main.md`

These commands exist in `src/taskclf/cli/main.py` but are not
documented in the CLI reference.

| Command | Code location | Priority | Status |
|---|---|---|---|
| ~~`train evaluate`~~ | `main.py:680-829` | High | **DONE** — added to commands table, section with options table and examples |
| ~~`train tune-reject`~~ | `main.py:832-935` | High | **DONE** — added to commands table, section with options table and examples |
| ~~`train calibrate`~~ | `main.py:938-1056` | High | **DONE** — added to commands table, section with options table and examples |
| ~~`ui`~~ | `main.py:2293-2454` | Medium | **DONE** — added to commands table, section with options table and examples |

---

## ~~4. Incomplete Content in `docs/api/ui/labeling.md`~~ — DONE

### ~~4a. Undocumented WebSocket event types~~ — DONE

Added `prompt_label` and `show_label_grid` event types to the
WebSocket event list in `labeling.md`.

---

## ~~5. Cross-Reference Gaps~~ — DONE

All five cross-reference gaps have been fixed:

| Doc page | Fix applied |
|---|---|
| `docs/api/infer/online.md` | Linked `ActiveLabelingQueue` to `docs/api/labels/queue.md` |
| `docs/api/infer/online.md` | Linked "model bundle" to `docs/guide/model_io.md` |
| `docs/api/infer/monitor.md` | Linked `core.drift` to `docs/api/core/drift.md` |
| `docs/api/infer/monitor.md` | Linked "telemetry store" to `docs/api/core/telemetry.md` |
| `docs/api/infer/baseline.md` | Linked `core.defaults` to `docs/api/core/defaults.md` |

---

## 6. Thin Doc Pages (round 2)

Audit found 13 additional pages that contain only a heading, one-line
description, and `:::` autodoc directive — no narrative, usage
examples, field tables, or architectural notes.

### Core

| Page | Status |
|------|--------|
| ~~`docs/api/core/schema.md`~~ | **DONE** — expanded with `FeatureSchemaV1` overview, 40-column registry grouped by role (6 tables), `validate_row` / `validate_dataframe` docs with dtype-kind mapping table, usage examples, cross-refs |
| ~~`docs/api/core/metrics.md`~~ | **DONE** — expanded with 10-function overview table, function-by-function sections for `compute_metrics`, `class_distribution`, `confusion_matrix_df`, `per_class_metrics`, `reject_rate`, `compare_baselines`, `per_user_metrics`, `calibration_curve_data`, `user_stratification_report`, `reject_rate_by_group`, return-value tables, usage examples, cross-refs |
| `docs/api/core/hashing.md` | Pending |
| `docs/api/core/store.md` | Pending |
| `docs/api/core/time.md` | Pending |
| `docs/api/core/logging.md` | Pending |
| `docs/api/core/defaults.md` | Pending |

### Features

| Page | Status |
|------|--------|
| ~~`docs/api/features/build.md`~~ | **DONE** — expanded with pipeline overview, data flow diagram, `build_features_from_aw_events` parameter table and dominant-app algorithm, sub-module invocation table, `generate_dummy_features` and `build_features_for_date` sections, cross-refs |
| ~~`docs/api/features/windows.md`~~ | **DONE** — expanded with `app_switch_count_in_window` parameter table, look-back window explanation, `compute_rolling_app_switches` batch helper, usage examples, cross-refs |
| `docs/api/features/sessions.md` | Pending |
| `docs/api/features/domain.md` | Pending |
| `docs/api/features/dynamics.md` | Pending |
| `docs/api/features/text.md` | Pending |
