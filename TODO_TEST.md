# TODO — Test Coverage

Missing tests identified by auditing `docs/api/` against
`tests/` and `src/taskclf/`.

---

## Part A — CLI Command Tests (CliRunner)

Commands that have **no CliRunner end-to-end test**. The underlying
library functions may be unit-tested, but the CLI wiring (argument
parsing, error handling, output formatting, file creation) is not.

Existing CLI tests live in:
- `tests/test_cli_main.py` — covers `ingest aw`, `features build`,
  `labels import`, `train lgbm`, `infer batch`, `report daily`
- `tests/test_cli_train_list.py` — covers `train list`
- `tests/test_cli_model_set_active.py` — covers `model set-active`
- `tests/test_infer_taxonomy.py` — covers `taxonomy validate/show/init`
  and `infer batch --taxonomy`

### A1. `labels add-block` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:184-275` (`labels_add_block_cmd`)
**Underlying tests:** `test_label_now.py`, `test_tray.py` test `append_label_span`

| Test case | What to verify |
|---|---|
| Basic block creation | Exit code 0, span appended to `labels_v1/labels.parquet` |
| Feature summary display | When features exist for the date range, block summary table renders |
| Model prediction display | When `--model-dir` provided, predicted label shown (or graceful error) |
| Overlap rejection | Exit code 1 when block overlaps existing span |
| Invalid label | Exit code ≠ 0 for a non-core label |
| `--confidence` persisted | Round-trip: value appears in read-back span |

### A2. `labels label-now` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:278-379` (`labels_label_now_cmd`)
**Underlying tests:** `test_label_now.py` tests span creation logic

| Test case | What to verify |
|---|---|
| Basic labeling | Exit code 0, span with correct time window created |
| `--minutes` respected | `end_ts - start_ts == timedelta(minutes=N)` |
| AW unreachable graceful fallback | Exit code 0, "not reachable" message in output |
| Overlap rejection | Exit code 1 when overlapping existing span |
| `--confidence` defaults to 1.0 | When omitted, stored confidence is 1.0 |

**Note:** Requires mocking `datetime.now()` for deterministic timestamps.

### A3. `labels show-queue` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:382-426` (`labels_show_queue_cmd`)
**Underlying tests:** `test_labels_queue.py`, `test_label_now.py` test `ActiveLabelingQueue`

| Test case | What to verify |
|---|---|
| Empty queue | Exit code 0, "No pending" or "No labeling queue" message |
| Populated queue | Exit code 0, table rendered with request ID, user, time range, reason |
| `--user-id` filter | Only matching user's items shown |
| `--limit` cap | At most N items in output |

### A4. `labels project` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:429-476` (`labels_project_cmd`)
**Underlying tests:** `test_labels_projection.py` tests `project_blocks_to_windows`

| Test case | What to verify |
|---|---|
| Synthetic round-trip | Exit code 0, `projected_labels.parquet` created |
| No labels file | Exit code 1 |
| No features in range | Exit code 1 |
| Projected row count | Output message matches actual parquet row count |

### A5. `train build-dataset` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:504-573` (`train_build_dataset_cmd`)
**Underlying tests:** `test_train_build_dataset.py` tests `build_training_dataset`

| Test case | What to verify |
|---|---|
| Synthetic dataset | Exit code 0, `X.parquet`, `y.parquet`, `splits.json` created |
| `--holdout-fraction` | Holdout users excluded from train/val splits |
| Custom `--train-ratio` / `--val-ratio` | Split sizes approximately match ratios |
| No features in range | Exit code 1 |

### A6. `train evaluate` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:680-829` (`train_evaluate_cmd`)
**Underlying tests:** `test_train_evaluate.py` tests `evaluate_model`

| Test case | What to verify |
|---|---|
| Synthetic evaluation | Exit code 0, metrics table rendered |
| Acceptance checks displayed | Output contains PASS/FAIL markers |
| Evaluation artifacts written | `evaluation_report.json` (or similar) created in `--out-dir` |
| `--reject-threshold` affects reject rate | Different thresholds → different reject rates |

### A7. `train tune-reject` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:832-935` (`train_tune_reject_cmd`)
**Underlying tests:** `test_tune_reject.py` tests `tune_reject_threshold`

| Test case | What to verify |
|---|---|
| Synthetic sweep | Exit code 0, sweep table rendered |
| JSON report written | `reject_tuning.json` created in `--out-dir` |
| Best threshold in output | "Recommended reject threshold" message present |

### A8. `train calibrate` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:938-1056` (`train_calibrate_cmd`)
**Underlying tests:** `test_calibration.py` tests calibrator fitting

| Test case | What to verify |
|---|---|
| Synthetic calibration | Exit code 0, calibrator store directory created |
| Eligibility table rendered | Output contains user eligibility info |
| `--method temperature` vs `--method isotonic` | Both complete without error |

### A9. `train retrain` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:1059-1175` (`train_retrain_cmd`)
**Underlying tests:** `test_retrain.py` tests `run_retrain_pipeline` (7+ tests)

| Test case | What to verify |
|---|---|
| `--synthetic --force` | Exit code 0, model bundle created or rejected |
| `--dry-run` | Model not promoted, "Dry run" in output |
| Gate table displayed | Output contains PASS/FAIL gate rows |
| Dataset hash in output | Dataset hash shown in summary table |

### A10. `train check-retrain` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:1178-1240` (`train_check_retrain_cmd`)
**Underlying tests:** `test_retrain.py` tests `check_retrain_due`

| Test case | What to verify |
|---|---|
| No models → DUE | Exit code 0, "DUE" in output |
| Fresh model → OK | Exit code 0, "OK" in output |
| `--calibrator-store` | Calibrator row appears in status table |

### A11. `infer baseline` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:1710-1772` (`infer_baseline_cmd`)
**Underlying tests:** `test_infer_baseline.py` tests `run_baseline_inference`

| Test case | What to verify |
|---|---|
| Synthetic baseline | Exit code 0, `baseline_predictions.csv` + `baseline_segments.json` created |
| Reject rate in output | "reject rate" message present |
| No features in range | Exit code 1 |

### A12. `infer compare` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:1775-1895` (`infer_compare_cmd`)
**Underlying tests:** `test_infer_baseline.py` tests `compare_baselines`

| Test case | What to verify |
|---|---|
| Synthetic comparison | Exit code 0, comparison table rendered |
| `baseline_vs_model.json` written | File created in `--out-dir` |
| Per-class F1 table present | Output contains per-class rows |

### A13. `infer online` — no CLI test (low priority)
**CLI function:** `src/taskclf/cli/main.py:1664-1707` (`infer_online_cmd`)
**Underlying tests:** `test_infer_online.py` tests the loop

**Note:** Difficult to test via CliRunner due to infinite poll loop.
Requires mocking the loop to run a single iteration or a stop signal.

| Test case | What to verify |
|---|---|
| Model resolution failure | Exit code 1, error message |
| `--label-queue` constructs queue path | Queue path is `data_dir/labels_v1/queue.json` |

### A14. `monitor drift-check` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:2037-2138` (`monitor_drift_check_cmd`)
**Underlying tests:** `test_monitor.py` tests `run_drift_check`

| Test case | What to verify |
|---|---|
| No drift | Exit code 0, "No drift detected" in output |
| Drift detected | Alert table rendered, `drift_report.json` written |
| `--auto-label` | "Auto-enqueued" message in output |

**Note:** Requires preparing reference + current parquet/CSV fixtures.

### A15. `monitor telemetry` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:2141-2183` (`monitor_telemetry_cmd`)
**Underlying tests:** `test_telemetry.py` tests `compute_telemetry` + `TelemetryStore`

| Test case | What to verify |
|---|---|
| Snapshot computed and stored | Exit code 0, snapshot file created in `--store-dir` |
| Output shows key metrics | "Windows", "Reject rate", "Confidence" in output |

**Note:** Requires preparing features parquet + predictions CSV fixtures.

### A16. `monitor show` — no CLI test
**CLI function:** `src/taskclf/cli/main.py:2186-2227` (`monitor_show_cmd`)
**Underlying tests:** `test_telemetry.py` tests `TelemetryStore.read_recent`

| Test case | What to verify |
|---|---|
| Empty store | Exit code 0, "No telemetry snapshots found" |
| Populated store | Table rendered with timestamps, windows, reject rate |
| `--user-id` filter | Only matching user's snapshots shown |
| `--last N` | At most N rows in table |

### A17. `tray` / `ui` — no CLI test (low priority)
**CLI functions:** `src/taskclf/cli/main.py:2233-2287` (`tray_cmd`),
`src/taskclf/cli/main.py:2293-2454` (`ui_serve_cmd`)
**Underlying tests:** `test_tray.py` (ActivityMonitor), `test_ui_server.py` (FastAPI endpoints)

**Note:** Interactive GUI commands — not feasible to test via CliRunner.
Component-level coverage is adequate.

---

## Part B — Core Module Tests

---

Missing tests identified by auditing `docs/api/core/` against
`tests/test_core_*.py` and `src/taskclf/core/*.py`.

## High Priority — Entire modules untested

### ~~1. `core.drift` — no test file exists~~ DONE

**Status:** Fully covered by `tests/test_drift.py` (18+ tests covering
`compute_psi`, `compute_ks`, `feature_drift_report`,
`detect_reject_rate_increase`, `detect_entropy_spike`, `detect_class_shift`).

---

### ~~2. `core.telemetry` — no test file exists~~ DONE

**Status:** Fully covered by `tests/test_telemetry.py` (12+ tests covering
`compute_telemetry`, `TelemetryStore.append`, `read_recent`, `read_range`,
per-user files, empty store, user_id propagation, window range, class distribution).

---

### ~~3. `core.logging` — no test file exists~~ DONE

**Status:** Covered by `tests/test_core_logging.py` (15 tests):
- `redact_message`: all 8 sensitive keys, `key=value`/`key: value`/quoted formats,
  non-sensitive passthrough, multiple keys, case-insensitive, empty/plain messages
- `SanitizingFilter`: always returns True, redacts with/without `record.args`
- `install_sanitizing_filter`: root logger, given logger, `handler_level=True`

---

### 4. `core.defaults` — no test file exists
**File:** `src/taskclf/core/defaults.py`
**Doc:** `docs/api/core/defaults.md`

Low priority — these are `Final` constants. A smoke test that imports all
public names and asserts expected types (`int`, `float`, `str`) would
catch accidental deletions or type changes.

---

## Medium Priority — Gaps within tested modules

### ~~5. `core.time` — `generate_bucket_range()` untested~~ DONE

**Status:** Covered by `tests/test_core_time.py` (TC-TIME-005 through TC-TIME-010):
basic range, start==end, start>end, unaligned inputs, timezone-aware, custom bucket_seconds=300.

---

### ~~6. `core.metrics` — 4 functions untested~~ DONE

**Status:**
- **6a. `reject_rate`**: Covered by `tests/test_core_metrics.py::TestRejectRate` (5 tests)
- **6b. `per_class_metrics`**: Covered by `tests/test_core_metrics.py::TestPerClassMetrics` (3 tests)
- **6c. `compare_baselines`**: Covered by `tests/test_core_metrics.py::TestCompareBaselines` (4 tests)
- **6d. `reject_rate_by_group`**: Covered by `tests/test_reject_rate_by_group.py` (8 tests)

---

### ~~7. `core.types` — 3 areas untested~~ DONE

**Status:** Covered by `tests/test_core_types.py`:
- **7a. Event protocol**: `TestEventProtocol` (2 tests: conforming/non-conforming)
- **7b. LabelSpan NaN coercion**: `TestLabelSpanConfidenceNaN` (3 tests: NaN→None, 0.8→0.8, None→None)
- **7c. LabelSpan.extend_forward**: `TestLabelSpanExtendForward` (2 tests: default False, explicit True)

---

### ~~8. `core.validation` — 2 documented checks untested~~ DONE

**Status:** Covered by `tests/test_core_validation.py`:
- **8a. Dominant-value warning**: `TestDistributionWarnings` (2 tests:
  `test_dominant_value_warns` with 95% identical, `test_no_dominant_value_at_80_percent`)
- **8b. Session boundary gap warning**: `TestSessionBoundary` (3 tests:
  small gap warns, large gap no warning, same session no warning)

---

## Low Priority

### ~~9. `core.model_io` — `generate_run_id()` untested~~ DONE

**Status:** Covered by `tests/test_core_model_io.py::TestGenerateRunId` (2 tests:
TC-MODEL-RUN-001 format matches regex, TC-MODEL-RUN-002 two calls differ).

---
---

# TODO — Features Test Coverage

Missing tests identified by auditing `docs/api/features/` against
`tests/test_features_*.py` and `src/taskclf/features/*.py`.

---

## High Priority — Completely untested public functions

### ~~10. `features.windows` — `compute_rolling_app_switches()` untested~~ DONE

**Status:** Covered by `tests/test_features_windows.py`:
- `TestComputeRollingAppSwitches` (6 tests: 3 buckets/4 apps, single bucket,
  empty buckets, empty events, custom `window_minutes`, custom `bucket_seconds`)
- `TestAppSwitchCountInWindow` extended with 4 new tests (events before window,
  events after window, custom `window_minutes`, custom `bucket_seconds`)

---

### ~~11. `features.build` — `generate_dummy_features()` untested~~ DONE

**Status:** Covered by `tests/test_features_build.py::TestGenerateDummyFeatures` (9 tests:
TC-FEAT-BUILD-001 through TC-FEAT-BUILD-009 — default row count, custom n_rows,
schema validation, timestamps on correct date, custom user_id/device_id,
schema_version/schema_hash, session_id consistency, dynamics fields populated,
n_rows=0 empty list).

---

### ~~12. `features.build` — `build_features_for_date()` untested~~ DONE

**Status:** Covered by `tests/test_features_build.py::TestBuildFeaturesForDate` (4 tests:
TC-FEAT-BUILD-010 through TC-FEAT-BUILD-013 — returns existing parquet path,
output path structure matches, parquet readable with correct columns,
row count matches DEFAULT_DUMMY_ROWS).

---

## Medium Priority — Edge cases missing in tested modules

### ~~13. `features.text` — `title_hash_bucket()` edge cases untested~~ DONE

**Status:** Covered by `tests/test_features_text.py`:
- `TestTitleHashBucket` (5 tests: `n_buckets=0` raises, `n_buckets=-1` raises,
  non-hex fallback, custom `n_buckets=10`, `n_buckets=1` always returns 0)
- `TestFeaturizeTitle` extended (2 new tests: empty string title, different
  salts produce different hashes)

---

### ~~14. `features.build` — `_aggregate_input_for_bucket()` no direct unit tests~~ DONE

**Status:** Covered by `tests/test_features_build.py::TestAggregateInputForBucket` (6 tests:
TC-FEAT-AGG-001 empty bucket returns all None, TC-FEAT-AGG-002 single active event
produces positive occupancy, TC-FEAT-AGG-003 mixed active/idle events max idle run,
TC-FEAT-AGG-004 keyboard only, TC-FEAT-AGG-005 mouse only,
TC-FEAT-AGG-006 event_density formula).

---

## Low Priority — Minor edge case gaps

### ~~15. `features.dynamics` — `compute_batch()` edge cases~~ DONE

**Status:** Covered by `tests/test_features_dynamics.py::TestComputeBatchEdgeCases` (3 tests:
TC-FEAT-DYN-001 empty lists return empty, TC-FEAT-DYN-002 single element all deltas None,
TC-FEAT-DYN-003 custom rolling_15 parameter respected).

---

### ~~16. `features.sessions` — `session_start_for_bucket()` edge case~~ DONE

**Status:** Covered by `tests/test_features_sessions.py::TestSessionStartForBucket::test_bucket_before_all_sessions`
(TC-FEAT-SESS-001: bucket_ts before all session starts returns first session start).

---

### ~~17. `features.domain` — `classify_domain()` edge cases~~ DONE

**Status:** Covered by `tests/test_features_domain.py::TestClassifyDomain` (2 new tests:
TC-FEAT-DOM-001 leading/trailing whitespace stripped before lookup,
TC-FEAT-DOM-002 deep subdomain 3+ levels matches parent domain).

---

### ~~18. `features.windows` — `app_switch_count_in_window()` edge cases~~ DONE

**Status:** Covered by `tests/test_features_windows.py::TestAppSwitchCountInWindow`
(2 new tests: events entirely before window, events entirely after window).
Also covered: custom `window_minutes` and `bucket_seconds` parameters.

---
---

# TODO — Train Module Test Coverage

Missing tests identified by auditing `docs/api/train/` against
`tests/test_train_*.py`, `tests/test_calibration.py`,
`tests/test_retrain.py`, `tests/test_tune_reject.py`, and
`src/taskclf/train/*.py`.

## Existing coverage summary

| Module | Test file(s) | Status |
|---|---|---|
| `train.dataset` | `test_train_dataset.py` (7 tests) | Fully covered — `split_by_time`, holdout, chronological ordering, validation errors |
| `train.build_dataset` | `test_train_build_dataset.py` (10 tests) | Well covered — artifact writing, exclusion rules, holdout users |
| `train.lgbm` | `test_train_evaluate.py` (6 tests for `compute_sample_weights` + `train_lgbm`) | `encode_categoricals`, `prepare_xy` untested directly |
| `train.calibrate` | `test_calibration.py` (12+ tests) | Well covered — eligibility, temperature/isotonic fitting, `fit_calibrator_store` |
| `train.evaluate` | `test_train_evaluate.py` (9 tests), `test_tune_reject.py` (4 tests) | Well covered — `evaluate_model`, acceptance checks, `tune_reject_threshold`, `write_evaluation_artifacts` |
| `train.retrain` | `test_retrain.py` (27+ tests) | Well covered — `check_calibrator_update_due` (6 tests) and `find_latest_model` (5 tests) now covered |

---

## Medium Priority — Direct unit tests missing

### ~~34. `train.lgbm` — `encode_categoricals()` no direct test~~ DONE

**Status:** Covered by `tests/test_train_lgbm.py::TestEncodeCategoricals` (5 tests:
TC-TRAIN-LGBM-001 fit new encoders for all 4 CATEGORICAL_COLUMNS,
TC-TRAIN-LGBM-002 pre-fitted encoder reuse produces identical codes,
TC-TRAIN-LGBM-003 unknown value maps to -1,
TC-TRAIN-LGBM-004 output shape preserved,
TC-TRAIN-LGBM-005 non-categorical columns untouched).

---

### ~~35. `train.lgbm` — `prepare_xy()` no direct test~~ DONE

**Status:** Covered by `tests/test_train_lgbm.py::TestPrepareXY` (6 tests:
TC-TRAIN-LGBM-006 output shapes correct,
TC-TRAIN-LGBM-007 NaN in numeric features filled with 0,
TC-TRAIN-LGBM-008 default label_encoder uses sorted(LABEL_SET_V1) with 8 classes,
TC-TRAIN-LGBM-009 pre-fitted encoders returned as same objects,
TC-TRAIN-LGBM-010 unknown label raises ValueError,
TC-TRAIN-LGBM-011 X dtype is float64).

---

### ~~36. `train.retrain` — `check_calibrator_update_due()` untested~~ DONE

**Status:** Covered by `tests/test_retrain.py::TestCheckCalibratorUpdateDue` (6 tests:
no store.json, fresh store, stale store, missing `created_at`, malformed JSON,
custom `cadence_days=1` with 2-day-old store).

---

## Low Priority — Indirect-only coverage

### ~~37. `train.retrain` — `find_latest_model()` no isolated test~~ DONE

**Status:** Covered by `tests/test_retrain.py::TestFindLatestModel` (5 tests:
empty directory, single bundle, multiple bundles returns latest, unreadable
metadata skipped, non-existent directory).

---
---

# TODO — UI Module Test Coverage

Missing tests identified by auditing `docs/api/ui/labeling.md` against
`tests/test_ui_server.py`, `tests/test_tray.py`, and `src/taskclf/ui/`.

**Existing test files:**
- `tests/test_ui_server.py` — covers label CRUD (GET/POST/PUT/DELETE), queue (GET empty,
  POST mark done), config/labels, config/user (GET/PUT), feature summary (empty),
  AW live (fallback), window toggle/state, show-label-grid, notification accept/skip,
  labels limit parameter, extend_forward (7 tests), WebSocket connect
- `tests/test_ui_events.py` — covers `EventBus` publish/subscribe delivery,
  multiple subscribers, dead subscriber eviction, publish_threadsafe variants,
  subscriber cleanup (7 tests)
- `tests/test_tray.py` — covers `ActivityMonitor.check_transition` (7 tests),
  label span creation via tray callbacks (3 tests)

---

## High Priority — Untested REST endpoints

### ~~34. `PUT /api/labels` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestUpdateLabel` (4 tests:
TC-UI-UPD-001 happy path update, TC-UI-UPD-002 404 no match,
TC-UI-UPD-003 422 invalid timestamp, TC-UI-UPD-004 updated label persisted via GET).

---

### ~~35. `DELETE /api/labels` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestDeleteLabel` (4 tests:
TC-UI-DEL-001 happy path delete, TC-UI-DEL-002 404 no match,
TC-UI-DEL-003 span removed from storage, TC-UI-DEL-004 422 invalid timestamp).

---

### ~~36. `POST /api/queue/{request_id}/done` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestMarkQueueDone` (4 tests:
TC-UI-QD-001 mark labeled, TC-UI-QD-002 mark skipped,
TC-UI-QD-003 non-existent request_id, TC-UI-QD-004 no queue file).

---

### ~~37. `GET /api/config/user` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestGetUserConfig` (2 tests:
TC-UI-CU-001 returns default config with UUID, TC-UI-CU-002 user_id stable across requests).

---

### ~~38. `PUT /api/config/user` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestUpdateUserConfig` (4 tests:
TC-UI-CU-003 update username, TC-UI-CU-004 persisted across requests,
TC-UI-CU-005 empty body no-op, TC-UI-CU-006 user_id unchanged after update).

---

### ~~39. `POST /api/notification/accept` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestNotificationAccept` (5 tests:
TC-UI-NA-001 accept valid suggestion with provenance="suggestion",
TC-UI-NA-002 invalid label 422, TC-UI-NA-003 invalid timestamps 422,
TC-UI-NA-004 overlap 409, TC-UI-NA-005 label persisted via GET).

---

### ~~40. `POST /api/notification/skip` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestNotificationSkip` (1 test:
TC-UI-NS-001 skip returns `{"status": "skipped"}`).

---

### ~~41. `POST /api/window/show-label-grid` — no test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestShowLabelGrid` (2 tests:
TC-UI-WS-001 without window_api returns ok,
TC-UI-WS-002 with window_api calls `show_label_grid()`).

---

### ~~42. `GET /api/labels` — `limit` parameter untested~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestLabelsLimit` (4 tests:
TC-UI-LL-001 limit=1, TC-UI-LL-002 limit=500,
TC-UI-LL-003 limit=0 rejected 422, TC-UI-LL-004 limit=501 rejected 422).

---

### 43. WebSocket event delivery — incomplete test
**Code:** `src/taskclf/ui/server.py:429-440` (`ws_predictions`)
**Current test:** `TestWebSocket.test_ws_connects` connects and publishes
but **never reads or asserts** the event was received by the client.

| Test case | What to verify |
|---|---|
| Event delivery round-trip | Connect WS, publish via EventBus, read from WS → matches published event |
| Multiple event types | Publish `status`, `prediction`, `tray_state`, `suggest_label` → all received in order |
| Multiple subscribers | Two WS clients, publish one event → both receive it |
| Disconnect cleanup | Connect WS, disconnect, publish → no error on server |

Suggested test IDs: `TC-UI-WS-003` through `TC-UI-WS-006`.

---

## Medium Priority — Untested supporting modules

### ~~44. `ui/events.py` — `EventBus` has no dedicated test file~~ DONE

**Status:** Covered by `tests/test_ui_events.py` (7 tests:
TC-UI-EB-001 publish/subscribe delivery, TC-UI-EB-002 multiple subscribers,
TC-UI-EB-003 dead subscriber eviction on full queue,
TC-UI-EB-004 publish_threadsafe with bound loop,
TC-UI-EB-005 publish_threadsafe with no loop (no-op),
TC-UI-EB-006 publish_threadsafe with closed loop (no-op),
TC-UI-EB-007 subscriber cleanup on context exit).

---

### 45. `ui/window.py` — `WindowAPI` methods mostly untested
**File:** `src/taskclf/ui/window.py`
**Doc:** `docs/api/ui/labeling.md:39-50` (live features section)
**Current tests:** Only `toggle_window` and `visible` are tested
(via `test_ui_server.py::TestWindowControl.test_toggle_with_window_api`).

All methods below are testable without a real GUI by using a mock
window object (only needs `show()`, `hide()`, `move()`, `x`, `y`
attributes).

| Method | Test case | What to verify |
|---|---|---|
| `show_label_grid` | With bound label window | `_label_visible` becomes `True`, `_label_window.show()` called |
| `show_label_grid` | Without label window (`None`) | No-op, no error |
| `hide_label_grid` | After showing | `_label_visible` becomes `False` after timer fires |
| `toggle_state_panel` (show) | Panel hidden → toggle | `_panel_visible` becomes `True`, `_panel_window.show()` called |
| `toggle_state_panel` (hide) | Panel visible → toggle | `_panel_visible` becomes `False` after timer fires |
| `_reposition_label` | Main window at known x/y | Label window `move()` called with `x + 150 - 280`, `y + 30 + 4` |
| `_position_panel` | Label hidden | Panel `move()` at `x + 150 - 280`, `y + 30 + 4` |
| `_position_panel` | Label visible | Panel `move()` at `x + 150 - 280`, `y + 30 + 4 + 330 + 4` |
| `bind` | Mock window | `_window` set, `moved` event bound |
| `bind_label` | Mock window | `_label_window` set |
| `bind_panel` | Mock window | `_panel_window` set |

Suggested test file: `tests/test_ui_window.py`.
Suggested test IDs: `TC-UI-WIN-001` through `TC-UI-WIN-011`.

---

### 46. `ui/tray.py` — `TrayLabeler` event publishing untested
**File:** `src/taskclf/ui/tray.py`
**Doc:** `docs/api/ui/labeling.md:91-98` (tray features section)
**Current tests:** `test_tray.py` covers `ActivityMonitor.check_transition`
(7 tests) and label span creation (3 tests). `TrayLabeler` itself and
its event bus integration have zero test coverage.

#### 46a. `TrayLabeler._handle_transition` — event publishing
**Lines:** `src/taskclf/ui/tray.py:425-483`

On transition, publishes `prompt_label` event to EventBus, plus either
`suggest_label` (when model suggests) or `prediction` (when no suggestion).

| Test case | What to verify |
|---|---|
| Transition without model | `prompt_label` event published with `suggested_label=None`; `prediction` event with `label="unknown"` |
| Transition with model suggestion | `prompt_label` event has `suggested_label` + `suggested_confidence`; `suggest_label` event published with `reason="app_switch"` |
| `_transition_count` incremented | Count increases by 1 per transition |
| `_last_transition` dict shape | Has keys: `prev_app`, `new_app`, `block_start`, `block_end`, `fired_at` |

**Setup:** Create `TrayLabeler` with a mock `EventBus` (capture published
events). For model tests, mock `_LabelSuggester.suggest()`.

#### 46b. `TrayLabeler._handle_poll` — tray_state event shape
**Lines:** `src/taskclf/ui/tray.py:407-423`

| Test case | What to verify |
|---|---|
| Poll publishes `tray_state` | Event has `type: "tray_state"`, `model_loaded`, `model_dir`, `transition_count`, `labels_saved_count`, `data_dir`, `ui_port`, `dev_mode` |
| `_current_app` updated | After `_handle_poll("com.apple.Safari")`, `_current_app == "com.apple.Safari"` |

#### 46c. `ActivityMonitor._publish_status` — status event shape
**Lines:** `src/taskclf/ui/tray.py:213-243`

| Test case | What to verify |
|---|---|
| Status event published | Event has `type: "status"`, `state: "collecting"`, `current_app`, `poll_seconds`, `poll_count`, `uptime_s`, `aw_connected`, `aw_host`, `last_event_count`, `last_app_counts` |
| `poll_count` increments | Two calls → `poll_count` increases |
| `candidate_app` included when present | After one candidate poll, `candidate_app` is non-null, `candidate_duration_s > 0` |

Suggested test IDs: `TC-UI-TRAY-001` through `TC-UI-TRAY-009`.

---

## Low Priority — Helpers and hard-to-test functions

### 47. `_send_desktop_notification` — no test
**File:** `src/taskclf/ui/tray.py:45-66`

Platform-specific (osascript on macOS, log-only elsewhere). Testable
by mocking `subprocess.run`.

| Test case | What to verify |
|---|---|
| macOS path | `subprocess.run` called with `osascript` args containing title and message |
| Non-macOS fallback | `logger.info` called with title and message |
| `subprocess.run` failure | No exception propagated, falls back to `logger.info` |

Suggested test IDs: `TC-UI-NOTIF-001` through `TC-UI-NOTIF-003`.

---

### 48. `_make_icon_image` — no test
**File:** `src/taskclf/ui/tray.py:317-326`

| Test case | What to verify |
|---|---|
| Default call | Returns `PIL.Image.Image`, size `(64, 64)`, mode `"RGBA"` |
| Custom color and size | `_make_icon_image("#FF0000", size=32)` → size `(32, 32)` |

Suggested test IDs: `TC-UI-ICON-001`, `TC-UI-ICON-002`.

---

### 49. `_LabelSuggester.suggest` — no test
**File:** `src/taskclf/ui/tray.py:271-309`

Requires mocking `fetch_aw_events`, `find_window_bucket_id`,
`build_features_from_aw_events`, and a loaded model. The
`OnlinePredictor.predict_bucket` return value determines the
suggestion.

| Test case | What to verify |
|---|---|
| Successful suggestion | Returns `(label_name, confidence)` tuple |
| No events from AW | Returns `None` |
| No features built | Returns `None` |
| Model prediction exception | Returns `None` (logged warning) |

Suggested test IDs: `TC-UI-SUG-001` through `TC-UI-SUG-004`.

---
---

# TODO — Infer Module Test Coverage

Missing tests identified by auditing `docs/api/infer/` against
`tests/test_infer_*.py` and `src/taskclf/infer/*.py`.

**Existing test files:**
- `tests/test_infer_smooth.py` — `rolling_majority` (6 tests), `segmentize` (6 tests), `flap_rate` (6 tests)
- `tests/test_infer_pipeline.py` — `WindowPrediction`, calibrators
  (identity + temperature), CSV output, `merge_short_segments`,
  `OnlinePredictor` with calibrator, batch inference with calibrator
- `tests/test_infer_calibration.py` — `IsotonicCalibrator` (6 tests),
  `CalibratorStore` (6 tests), `save_calibrator_store`/`load_calibrator_store` (5 tests)
- `tests/test_infer_online.py` — `OnlinePredictor` predict/segments,
  session tracking
- `tests/test_infer_batch_reject.py` — `predict_labels` reject,
  `run_batch_inference` reject, `write_predictions_csv` reject columns
- `tests/test_infer_taxonomy.py` — config validation, resolver
  single/batch, YAML round-trip, default taxonomy, CLI commands,
  batch inference integration
- `tests/test_infer_baseline.py` — all 4 rules, priority ordering,
  acceptance gates, `predict_baseline`, `run_baseline_inference`,
  metrics (reject rate, compare baselines)
- `tests/test_infer_resolve.py` — `resolve_model_dir` (5 scenarios),
  `ActiveModelReloader` (no-change + interval guard)

---

## High Priority — Entire module untested

### 19. `infer.monitor` — no test file exists
**File:** `src/taskclf/infer/monitor.py`
**Doc:** `docs/api/infer/monitor.md`

Three public functions, two Pydantic models, and one StrEnum — zero
test coverage.

#### 19a. `run_drift_check()` (lines 89–249)

Orchestrates all drift sub-checks from `core.drift` and returns a
consolidated `DriftReport`.

```python
def run_drift_check(
    ref_features_df: pd.DataFrame,
    cur_features_df: pd.DataFrame,
    ref_labels: Sequence[str],
    cur_labels: Sequence[str],
    *,
    ref_probs: np.ndarray | None = None,
    cur_probs: np.ndarray | None = None,
    cur_confidences: np.ndarray | None = None,
    ...
) -> DriftReport:
```

| Test case | Expected |
|---|---|
| No drift (identical ref/cur) | `DriftReport` with empty `alerts`, `any_critical=False`, `summary` contains "No drift detected" |
| Feature PSI drift | At least one `DriftAlert` with `trigger=DriftTrigger.feature_psi`, affected feature listed |
| Feature KS drift | Alert with `trigger=DriftTrigger.feature_ks` |
| Reject rate increase | Alert with `trigger=DriftTrigger.reject_rate_increase`, `severity="critical"` |
| Entropy spike (ref_probs + cur_probs provided) | Alert with `trigger=DriftTrigger.entropy_spike` when cur entropy >> ref |
| No entropy check when probs omitted | `entropy_drift` is `None` on report |
| Class distribution shift | Alert with `trigger=DriftTrigger.class_shift`, `shifted_classes` populated |
| Critical severity threshold | PSI > 2× threshold → `severity="critical"` |
| `telemetry_snapshot` populated | Always present on report |
| `any_critical` flag correct | `True` iff any alert has `severity="critical"` |
| Summary text includes specifics | Lists drifted features, reject rate numbers, entropy ratio |

#### 19b. `auto_enqueue_drift_labels()` (lines 257–308)

Selects lowest-confidence buckets from the current window and enqueues
them for labeling via `ActiveLabelingQueue`.

```python
def auto_enqueue_drift_labels(
    drift_report: DriftReport,
    cur_features_df: pd.DataFrame,
    queue_path: Path,
    *,
    cur_confidences: np.ndarray | None = None,
    limit: int = DEFAULT_DRIFT_AUTO_LABEL_LIMIT,
) -> int:
```

| Test case | Expected |
|---|---|
| No alerts → no enqueue | Returns `0`, queue file unchanged |
| Alerts present → enqueues | Returns count > 0, queue file contains items |
| `limit` respected | At most `limit` items enqueued |
| Lowest confidence selected first | When `cur_confidences` provided, enqueued buckets are the lowest-confidence ones |
| No `cur_confidences` → still enqueues | Falls back to first `limit` rows (no sorting) |

#### 19c. `write_drift_report()` (lines 316–328)

```python
def write_drift_report(report: DriftReport, path: Path) -> Path:
```

| Test case | Expected |
|---|---|
| Round-trip | Written JSON is valid, parseable back to `DriftReport` via `model_validate_json` |
| Parent dirs created | Non-existent parent path created automatically |
| Returns the path | Return value equals the input `path` |

#### 19d. Models: `DriftTrigger`, `DriftAlert`, `DriftReport`

| Test case | Expected |
|---|---|
| `DriftTrigger` values | All 5 values exist and are strings |
| `DriftAlert` construction | All fields populated, `timestamp` is datetime |
| `DriftReport.any_critical` | Computed from `alerts` severity |

Suggested test file: `tests/test_infer_monitor.py`
Suggested test IDs: `TC-MON-001` through `TC-MON-011`.

---

### ~~20. `infer.calibration` — `IsotonicCalibrator`, `CalibratorStore`, and store persistence untested~~ DONE

**Status:** Covered by `tests/test_infer_calibration.py` (17 tests):
- **IsotonicCalibrator** (6 tests): TC-CAL-001 1D calibrate sums to 1.0,
  TC-CAL-002 2D calibrate rows sum to 1.0, TC-CAL-003 empty regressors raises,
  TC-CAL-004 n_classes property, TC-CAL-005 satisfies Calibrator protocol,
  TC-CAL-006 save/load round-trip preserves predictions.
- **CalibratorStore** (6 tests): TC-CAL-007 get per-user calibrator,
  TC-CAL-008 fallback to global, TC-CAL-009 calibrate_batch per-user,
  TC-CAL-010 shape preserved, TC-CAL-011 user_ids sorted,
  TC-CAL-012 empty user_calibrators fallback.
- **Store persistence** (5 tests): TC-CAL-013 round-trip temperature + 2 per-user,
  TC-CAL-014 round-trip isotonic global, TC-CAL-015 directory layout,
  TC-CAL-016 store.json metadata, TC-CAL-017 empty per-user dict.

---

## Medium Priority — Gaps within tested modules

### 21. `infer.batch` — `predict_proba`, segment I/O, and batch+taxonomy untested
**File:** `src/taskclf/infer/batch.py`
**Doc:** `docs/api/infer/batch.md`
**Existing tests:** `tests/test_infer_batch_reject.py` (reject paths),
`tests/test_infer_pipeline.py` (CSV output, hysteresis, calibrator).

#### 21a. `predict_proba()` (lines 47–65) — no direct test

```python
def predict_proba(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> np.ndarray:
```

| Test case | Expected |
|---|---|
| Returns `(N, 8)` array | Shape matches `(len(df), 8)` |
| Rows sum to ~1.0 | `np.allclose(proba.sum(axis=1), 1.0)` |
| `cat_encoders=None` works | Falls back to integer encoding |

#### 21b. `write_segments_json()` / `read_segments_json()` round-trip (lines 249–289) — no test

```python
def write_segments_json(segments: Sequence[Segment], path: Path) -> Path:
def read_segments_json(path: Path) -> list[Segment]:
```

| Test case | Expected |
|---|---|
| Round-trip preserves data | Write then read → same `start_ts`, `end_ts`, `label`, `bucket_count` |
| Empty segments list | Writes `[]`, reads back as `[]` |
| Parent dirs created | Non-existent parent path created automatically |

#### 21c. `run_batch_inference()` with taxonomy (lines 107–198) — unit test missing

CLI integration tested in `test_infer_taxonomy.py`, but no direct unit
test for `run_batch_inference(..., taxonomy=config)`.

| Test case | Expected |
|---|---|
| Taxonomy populates `mapped_labels` | `result.mapped_labels` is not `None`, length matches input |
| Taxonomy populates `mapped_probs` | `result.mapped_probs` is not `None`, each dict sums to ~1.0 |
| Without taxonomy | `mapped_labels` and `mapped_probs` are `None` |

#### 21d. `run_batch_inference()` with `calibrator_store` — no test

| Test case | Expected |
|---|---|
| Per-user calibration applied | `result.core_probs` differs from uncalibrated run when store has per-user calibrators |
| `user_id` column required | DataFrame without `user_id` falls back to single calibrator |

Suggested test IDs: `TC-BATCH-001` through `TC-BATCH-009`.

---

### ~~22. `infer.smooth` — `flap_rate()` untested~~ DONE

**Status:** Covered by `tests/test_infer_smooth.py`:
- `TestFlapRate` (6 tests: all same, all different, single element, empty,
  alternating, two runs)
- `TestRollingMajority` extended (5 new tests: empty list, single element,
  all identical, tie-breaking preserves original, `window=1` no smoothing)
- `TestSegmentize` extended (3 new tests: mismatched lengths raises, empty
  inputs, single bucket)

---

### 23. `infer.resolve` — `ActiveModelReloader` reload path untested
**File:** `src/taskclf/infer/resolve.py:99-163`
**Doc:** `docs/api/infer/online.md`
**Existing tests:** `tests/test_infer_resolve.py` covers no-change and
interval-guard paths only.

`ActiveModelReloader.check_reload()` when mtime changes (lines 130–162):

| Test case | Expected |
|---|---|
| Mtime changes → successful reload | Returns `(model, metadata, cat_encoders)` tuple |
| Mtime changes → load fails | Returns `None`, logs warning, current model kept |
| `_last_mtime` updated after successful reload | Subsequent check with same mtime returns `None` |
| No `active.json` file | `_current_mtime()` returns `None`, no reload |

**Setup:** Write a model bundle to `tmp_path/models/`, create
`active.json`, construct `ActiveModelReloader` with
`check_interval_s=0`, then modify `active.json` mtime (e.g. write new
pointer) before calling `check_reload()`.

Suggested test IDs: `TC-RELOAD-001` through `TC-RELOAD-004`.

---

### 24. `infer.online` — `OnlinePredictor` with taxonomy and calibrator store untested
**File:** `src/taskclf/infer/online.py`
**Doc:** `docs/api/infer/online.md`
**Existing tests:** `tests/test_infer_online.py` (basic predict/segments),
`tests/test_infer_pipeline.py` (calibrator integration, CSV output).

| Test case | Expected |
|---|---|
| `OnlinePredictor` with `taxonomy` | `prediction.mapped_label_name` comes from taxonomy buckets, not core labels |
| `OnlinePredictor` with `taxonomy` — `mapped_probs` | `mapped_probs` keys are bucket names, values sum to ~1.0 |
| `OnlinePredictor` with `calibrator_store` | Per-user calibration applied (different `user_id` → potentially different confidence) |
| `_encode_value` for categorical column | Known value → encoded int; unknown value → `-1.0` |
| `_encode_value` for numerical `None` | Returns `0.0` |
| `get_segments` after rejected predictions | Segments use `MIXED_UNKNOWN` label |

Suggested test IDs: `TC-ONLINE-001` through `TC-ONLINE-006`.

---

### ~~25. `infer.prediction` — boundary validation untested~~ DONE

**Status:** Covered by `tests/test_infer_pipeline.py::TestWindowPrediction` (5 new tests:
TC-PRED-001 confidence<0, TC-PRED-002 confidence>1.0, TC-PRED-003 core_label_id<0,
TC-PRED-004 core_label_id>7, TC-PRED-005 frozen model immutability).

---

## Low Priority

### 26. `infer.baseline` — `_safe_float()` and custom thresholds untested
**File:** `src/taskclf/infer/baseline.py`
**Existing tests:** `tests/test_infer_baseline.py` (comprehensive rule
coverage, all using default thresholds).

#### 26a. `_safe_float()` (lines 35–45) — internal but non-trivial

| Test case | Expected |
|---|---|
| `None` | Returns `default` |
| `float('nan')` | Returns `default` |
| Valid float | Returns that float |
| Non-numeric string `"abc"` | Returns `default` |
| Integer `42` | Returns `42.0` |

#### 26b. Custom threshold parameters

| Test case | Expected |
|---|---|
| `classify_single_row(row, idle_active_threshold=100.0)` | Row with `active_seconds_any=50` is NOT idle (would be idle at default 5.0) |
| `run_baseline_inference(df, scroll_high=100.0)` | Browser+high-scroll row is NOT ReadResearch (threshold too high) |

Suggested test IDs: `TC-BASE-001` through `TC-BASE-007`.

---
---

# TODO — Report Test Coverage

Missing tests identified by auditing `docs/api/report/` against
`tests/test_report.py` and `src/taskclf/report/*.py`.

**Already tested:**
- `build_daily_report` basic flow (totals, core_breakdown, segments_count,
  optional-fields-None, empty-segments-raises)
- `build_daily_report` with `mapped_labels`, `raw_labels`,
  `smoothed_labels`, `app_switch_counts`
- `export_report_json` roundtrip, excludes-None, includes-full-data
- `export_report_csv` core-only rows, core+mapped rows, column names
- `export_report_parquet` roundtrip schema, core+mapped row counts
- `flap_rate` (6 cases, defined in `infer.smooth`)

---

## High Priority — Sensitive-field guard never triggered

### ~~19. `_check_no_sensitive_fields()` — rejection path untested~~ DONE

**Status:** Covered by `tests/test_report.py::TestCheckNoSensitiveFields` (4 tests:
top-level key, nested key, all 4 keys individually, clean dict passes).

---

### ~~20. Sensitive-field rejection via each export function~~ DONE

**Status:** Covered by `tests/test_report.py::TestExportSensitiveGuard` (3 tests:
`export_report_json`, `export_report_csv`, `export_report_parquet` each raise
`ValueError` when monkeypatched `model_dump` injects a sensitive key).

---

## Medium Priority — Missing edge cases

### ~~21. `build_daily_report()` — non-default `bucket_seconds`~~ DONE

**Status:** Covered by `tests/test_report.py::TestBuildDailyReportBucketSeconds` (2 tests:
TC-RPT-DAILY-001 bucket_seconds=300 scales total_minutes, TC-RPT-DAILY-002
bucket_seconds=120 scales core_breakdown).

### ~~22. `build_daily_report()` — `smoothed_labels` without `raw_labels`~~ DONE

**Status:** Covered by `tests/test_report.py::TestBuildDailyReportSmoothedOnly` (1 test:
TC-RPT-DAILY-003 smoothed_labels provided with raw_labels=None).

### ~~23. `_build_context_switch_stats()` — edge cases~~ DONE

**Status:** Covered by `tests/test_report.py::TestBuildContextSwitchStatsEdgeCases` (4 tests:
TC-RPT-CTX-001 empty list, TC-RPT-CTX-002 single element,
TC-RPT-CTX-003 float values truncated, TC-RPT-CTX-004 median even count).

### ~~24. Pydantic validation on report models~~ DONE

**Status:** Covered by `tests/test_report.py::TestReportModelValidation` (3 tests:
TC-RPT-VAL-001 negative ContextSwitchStats.mean, TC-RPT-VAL-002 negative
DailyReport.total_minutes, TC-RPT-VAL-003 negative DailyReport.segments_count).

---

## Low Priority — Minor gaps

### 25. Export functions — parent directory creation
**Files:** `src/taskclf/report/export.py` — all three export functions
call `path.parent.mkdir(parents=True, exist_ok=True)`.
**Tests:** All existing tests use `tmp_path` (already exists).

| Test case | Expected |
|---|---|
| `export_report_json` with nested non-existent parent (`tmp_path / "a/b/report.json"`) | Directories created, file written |
| Same for `export_report_csv` | Directories created |
| Same for `export_report_parquet` | Directories created |

Suggested test IDs: `TC-RPT-MKDIR-001` through `TC-RPT-MKDIR-003`.

### 26. `_breakdown_to_rows()` — row content correctness
**File:** `src/taskclf/report/export.py:54-74`
**Tests:** Row counts and label sets are checked; per-row `minutes`
values and sort order are not.

| Test case | Expected |
|---|---|
| Core rows sorted alphabetically by label | Rows in `sorted()` order |
| `minutes` values match `core_breakdown` | Each row's `minutes` equals `round(core_breakdown[label], 2)` |
| `date` propagated into every row | All rows have `report.date` |

Suggested test IDs: `TC-RPT-ROWS-001` through `TC-RPT-ROWS-003`.

### 27. Export value correctness in CSV and Parquet
**Tests:** `TestExportCsv` and `TestExportParquet` check row counts
and column names but not the actual `minutes` values or `date` field.

| Test case | Expected |
|---|---|
| CSV: `minutes` column values match `core_breakdown` | Float comparison per row |
| Parquet: `minutes` column values match `core_breakdown` | Same |
| CSV: `date` column matches `report.date` | All rows equal |
| Parquet: `date` column matches `report.date` | All rows equal |

Suggested test IDs: `TC-RPT-CSVVAL-001` through `TC-RPT-CSVVAL-004`.

---
---

# TODO — Labels Test Coverage

Missing tests identified by auditing `docs/api/labels/` against
`tests/test_labels_*.py`, `tests/test_label_now.py`, and
`src/taskclf/labels/*.py`.

**Already well-tested:**
- `write_label_spans` / `read_label_spans` — round-trip in `test_labels_store.py::TestLabelSpanRoundTrip`
- `append_label_span` — extensive coverage in `test_labels_store.py::TestAppendLabelSpan`,
  `TestExtendForward` (18 tests), and `test_label_now.py::TestLabelNowSpanCreation`
- `_same_user` — `test_labels_store.py::TestSameUser` (4 tests)
- `import_labels_from_csv` — `test_labels_store.py::TestImportLabelsFromCsvWithOptionalColumns`
  (happy paths) + `TestImportLabelsFromCsvErrors` (3 error-path tests)
- `generate_label_summary` — `test_labels_store.py::TestGenerateLabelSummary` (happy paths only)
- `generate_dummy_labels` — `test_labels_store.py::TestGenerateDummyLabels` (4 tests)
- `project_blocks_to_windows` — `test_labels_projection.py` (11 tests across 7 classes)
- `ActiveLabelingQueue` — `test_labels_queue.py` (13 tests) + `test_label_now.py::TestOnlineQueueEnqueue`

---

## High Priority — Public functions with zero test coverage

### ~~28. `labels.store.update_label_span()` — no tests~~ DONE

**Status:** Covered by `tests/test_labels_store.py::TestUpdateLabelSpan` (5 tests:
happy path, file not found, no matching span, invalid label, preserves other fields).

---

### ~~29. `labels.store.delete_label_span()` — no tests~~ DONE

**Status:** Covered by `tests/test_labels_store.py::TestDeleteLabelSpan` (5 tests:
delete one of two, delete only span, file not found, no matching span,
same start different end).

---

## Medium Priority — Error/edge paths missing in tested functions

### ~~30. `labels.store.import_labels_from_csv()` — missing error path~~ DONE

**Status:** Covered by `tests/test_labels_store.py::TestImportLabelsFromCsvErrors` (3 tests:
TC-LABEL-CSV-001 missing label column, TC-LABEL-CSV-002 missing multiple columns,
TC-LABEL-CSV-003 invalid label value raises ValidationError).

---

### 31. `labels.store.generate_label_summary()` — missing edge cases
**File:** `src/taskclf/labels/store.py:255-316`
**Tests:** `test_labels_store.py::TestGenerateLabelSummary` (2 tests:
empty window + populated window)

The function handles missing columns gracefully (`app_id` → empty
`top_apps`, `session_id` → `session_count=0`, input columns → `None`
means). These defensive branches are untested.

| Test case | Setup | Expected |
|---|---|---|
| No `app_id` column | DataFrame without `app_id` | `top_apps == []` |
| No `session_id` column | DataFrame without `session_id` | `session_count == 0` |
| No input rate columns | DataFrame without `keys_per_min`, `clicks_per_min`, `scroll_events_per_min` | All mean values are `None` |
| All columns present but all NaN in rate columns | Rate columns filled with `float('nan')` | Mean values are `None` (`.dropna()` produces empty series) |

Suggested test IDs: `TC-LABEL-SUM-001` through `TC-LABEL-SUM-004`.

---

### 32. `labels.queue.enqueue_drift()` — missing edge cases
**File:** `src/taskclf/labels/queue.py:148-186`
**Tests:** `test_labels_queue.py::TestEnqueueDrift` (1 test: basic add)

Missing deduplication test and missing-optional-fields test.

| Test case | Setup | Expected |
|---|---|---|
| Dedup: same bucket re-enqueued | Call `enqueue_drift` twice with same bucket | Second call returns `0`, total items still `1` |
| Missing optional fields | Dict without `predicted_label` or `confidence` | `LabelRequest` created with `confidence=None`, `predicted_label=None` |
| Mixed: some new, some existing | Enqueue 3, then enqueue 4 (2 overlap) | Returns `2` (only new ones added) |

Suggested test IDs: `TC-LABEL-QD-001` through `TC-LABEL-QD-003`.

---

### 33. `labels.projection.project_blocks_to_windows()` — missing branch coverage
**File:** `src/taskclf/labels/projection.py:19-99`
**Tests:** `test_labels_projection.py` (11 tests, thorough for main rules)

Two code branches have no dedicated test:

| Test case | Setup | Expected |
|---|---|---|
| Auto-derived `bucket_end_ts` | Features DataFrame **without** a `bucket_end_ts` column | Column auto-derived from `bucket_start_ts + bucket_seconds`; projection still works correctly |
| Same-label multi-block covering | Two blocks with **identical** labels both fully cover a window | Window is labeled (not dropped), since only **conflicting** labels cause drops |

Suggested test IDs: `TC-LABEL-PROJ-001`, `TC-LABEL-PROJ-002`.
