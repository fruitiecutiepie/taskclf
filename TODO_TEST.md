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
- `tests/test_cli_commands.py` — covers `labels add-block`,
  `labels label-now`, `labels show-queue`, `labels project`,
  `train build-dataset`, `train evaluate`, `train tune-reject`,
  `train calibrate`, `train retrain`, `train check-retrain`,
  `infer baseline`, `infer compare`, `monitor drift-check`,
  `monitor telemetry`, `monitor show`
- `tests/test_cli_train_list.py` — covers `train list`
- `tests/test_cli_model_set_active.py` — covers `model set-active`
- `tests/test_infer_taxonomy.py` — covers `taxonomy validate/show/init`
  and `infer batch --taxonomy`

### ~~A1. `labels add-block` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestLabelsAddBlock` (5 tests:
TC-CLI-AB-001 basic block creation exit 0 + span in parquet,
TC-CLI-AB-002 overlap rejection exit 1,
TC-CLI-AB-003 invalid label exit != 0,
TC-CLI-AB-004 `--confidence` round-trip persisted,
TC-CLI-AB-005 feature summary table rendered when features exist).

### ~~A2. `labels label-now` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestLabelsLabelNow` (5 tests:
TC-CLI-LN-001 basic labeling exit 0 + span persisted,
TC-CLI-LN-002 `--minutes 25` produces 25-minute span,
TC-CLI-LN-003 unreachable AW host exit 0 + "not reachable" in output,
TC-CLI-LN-004 overlap rejection exit 1,
TC-CLI-LN-005 omitting `--confidence` stores 1.0).

### ~~A3. `labels show-queue` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestLabelsShowQueue` (4 tests:
TC-CLI-SQ-001 no queue file shows "No labeling queue",
TC-CLI-SQ-002 populated queue renders table,
TC-CLI-SQ-003 `--user-id` filter shows only matching user,
TC-CLI-SQ-004 `--limit` caps visible items).

### ~~A4. `labels project` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestLabelsProject` (4 tests:
TC-CLI-LP-001 round-trip creates `projected_labels.parquet`,
TC-CLI-LP-002 no labels file exit 1,
TC-CLI-LP-003 no features in range exit 1,
TC-CLI-LP-004 projected row count mentioned in output).

### ~~A5. `train build-dataset` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainBuildDataset` (3 tests:
TC-CLI-BD-001 synthetic dataset creates X/y/splits artifacts,
TC-CLI-BD-002 custom `--train-ratio`/`--val-ratio` reflected in splits,
TC-CLI-BD-003 no features non-synthetic exit 1).

### ~~A6. `train evaluate` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainEvaluate` (4 tests:
TC-CLI-EV-001 synthetic evaluation exit 0 + "Overall Metrics" table rendered,
TC-CLI-EV-002 "Acceptance Checks" with PASS/FAIL markers in output,
TC-CLI-EV-003 `evaluation.json` created in `--out-dir`,
TC-CLI-EV-004 `--reject-threshold 0.99` produces higher reject rate than default).

### ~~A7. `train tune-reject` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainTuneReject` (3 tests:
TC-CLI-TR-001 synthetic sweep exit 0 + "Reject Threshold Sweep" table rendered,
TC-CLI-TR-002 `reject_tuning.json` created in `--out-dir`,
TC-CLI-TR-003 "Recommended reject threshold" message present in output).

### ~~A8. `train calibrate` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainCalibrate` (3 tests:
TC-CLI-CA-001 synthetic calibration exit 0 + `store.json` created,
TC-CLI-CA-002 "Eligibility" table rendered in output,
TC-CLI-CA-003 `--method isotonic` completes without error + `store.json` created).

### ~~A9. `train retrain` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainRetrain` (4 tests:
TC-CLI-RT-001 `--synthetic --force` exit 0 + "Retrain Result" table rendered,
TC-CLI-RT-002 `--dry-run` prevents promotion,
TC-CLI-RT-003 regression gates table with PASS/FAIL when champion exists,
TC-CLI-RT-004 "Dataset hash" row present in summary table).

### ~~A10. `train check-retrain` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestTrainCheckRetrain` (3 tests:
TC-CLI-CR-001 no models directory → "DUE" in output,
TC-CLI-CR-002 freshly trained model → "OK" in output,
TC-CLI-CR-003 `--calibrator-store` adds "Calibrator" row to status table).

### ~~A11. `infer baseline` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestInferBaseline` (3 tests:
TC-CLI-BL-001 synthetic baseline creates predictions CSV + segments JSON,
TC-CLI-BL-002 "reject rate" present in output,
TC-CLI-BL-003 no features in range exit 1).

### ~~A12. `infer compare` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestInferCompare` (3 tests:
TC-CLI-IC-001 synthetic comparison exit 0 + "Baseline vs Model" table rendered,
TC-CLI-IC-002 `baseline_vs_model.json` created in `--out-dir`,
TC-CLI-IC-003 "Per-Class F1" table present in output).

### ~~A13. `infer online` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestInferOnline` (2 tests:
TC-CLI-IO-001 `ModelResolutionError` → exit 1 with error message,
TC-CLI-IO-002 `--label-queue` constructs queue path `data_dir/labels_v1/queue.json`).

**Note:** The infinite poll loop is mocked out (`run_online_loop` patched)
so CliRunner can exercise argument parsing and error handling.

### ~~A14. `monitor drift-check` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestMonitorDriftCheck` (3 tests:
TC-CLI-DC-001 identical ref/cur → "No drift detected",
TC-CLI-DC-002 shifted `keys_per_min` triggers alert table + `drift_report.json`,
TC-CLI-DC-003 `--auto-label` with drift prints "Auto-enqueued").

### ~~A15. `monitor telemetry` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestMonitorTelemetry` (2 tests:
TC-CLI-TEL-001 snapshot appended to JSONL file in `--store-dir`,
TC-CLI-TEL-002 output contains "Windows", "Reject rate", "Confidence").

### ~~A16. `monitor show` — no CLI test~~ DONE

**Status:** Covered by `tests/test_cli_commands.py::TestMonitorShow` (4 tests:
TC-CLI-MS-001 empty store shows "No telemetry snapshots found",
TC-CLI-MS-002 populated store renders table with data,
TC-CLI-MS-003 `--user-id` filter shows only matching user,
TC-CLI-MS-004 `--last N` caps snapshots shown).

### A17. `tray` / `ui` — no CLI test (low priority)
**CLI functions:** `src/taskclf/cli/main.py:2233-2287` (`tray_cmd`),
`src/taskclf/cli/main.py:2293-2454` (`ui_serve_cmd`)
**Underlying tests:** `test_tray.py` (ActivityMonitor), `test_ui_server.py` (FastAPI endpoints)

**Note:** Interactive GUI commands — not feasible to test via CliRunner.
Component-level coverage is adequate. See **Part G** below for automated
tray testing strategies that don't require manual interaction.

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

### ~~4. `core.defaults` — no test file exists~~ DONE

**Status:** Covered by `tests/test_core_defaults.py::TestDefaultsTypes` (4 tests:
`test_int_constants` checks 26 int constants, `test_float_constants` checks 15 float
constants, `test_str_constants` checks 10 str constants, `test_all_public_names_covered`
ensures no public name is missed).

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

### ~~43. WebSocket event delivery — incomplete test~~ DONE

**Status:** Covered by `tests/test_ui_server.py::TestWebSocket` (4 new tests:
TC-UI-WS-003 event delivery round-trip via publish_threadsafe + receive_json,
TC-UI-WS-004 multiple event types received in publish order,
TC-UI-WS-005 two WS clients both receive the same published event,
TC-UI-WS-006 disconnect then publish causes no server error).

**Note:** Tests use `TestClient` as context manager to ensure lifespan runs
(binds EventBus loop), and publish from a background thread to avoid
blocking on `receive_json()`.

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

### ~~45. `ui/window.py` — `WindowAPI` methods mostly untested~~ DONE

**Status:** Covered by `tests/test_ui_window.py` (14 tests):
- `TestShowLabelGrid` (3 tests: TC-UI-WIN-001 label visible and show() called,
  TC-UI-WIN-002 no-op without label window, no-op without main window)
- `TestHideLabelGrid` (1 test: TC-UI-WIN-003 label hidden after timer fires)
- `TestToggleStatePanel` (4 tests: TC-UI-WIN-004 show panel, TC-UI-WIN-005
  hide panel, no-op without panel window, no-op without main window)
- `TestRepositionLabel` (1 test: TC-UI-WIN-006 label positioned below pill)
- `TestPositionPanel` (2 tests: TC-UI-WIN-007 panel below pill label hidden,
  TC-UI-WIN-008 panel below label when visible)
- `TestBind` (3 tests: TC-UI-WIN-009 bind sets window and event,
  TC-UI-WIN-010 bind_label sets label window,
  TC-UI-WIN-011 bind_panel sets panel window)

---

### ~~46. `ui/tray.py` — `TrayLabeler` event publishing untested~~ DONE

**Status:** Covered by `tests/test_tray.py` (9 tests):
- **46a. `TrayLabeler._handle_transition`**: `TestHandleTransition` (4 tests:
  TC-UI-TRAY-001 transition without model publishes `prompt_label` + `prediction`,
  TC-UI-TRAY-002 transition with suggestion publishes `prompt_label` + `suggest_label`,
  TC-UI-TRAY-003 `_transition_count` incremented per call,
  TC-UI-TRAY-004 `_last_transition` dict has expected keys)
- **46b. `TrayLabeler._handle_poll`**: `TestHandlePoll` (2 tests:
  TC-UI-TRAY-005 `tray_state` event published with all expected keys,
  TC-UI-TRAY-006 `_current_app` updated after poll)
- **46c. `ActivityMonitor._publish_status`**: `TestPublishStatus` (3 tests:
  TC-UI-TRAY-007 status event shape with all expected keys,
  TC-UI-TRAY-008 `poll_count` increments across calls,
  TC-UI-TRAY-009 `candidate_app` included when present)

---

## Low Priority — Helpers and hard-to-test functions

### ~~47. `_send_desktop_notification` — no test~~ DONE

**Status:** Covered by `tests/test_tray.py::TestSendDesktopNotification` (3 tests:
TC-UI-NOTIF-001 macOS path calls `osascript` with title and message,
TC-UI-NOTIF-002 non-macOS fallback calls `logger.info`,
TC-UI-NOTIF-003 `subprocess.run` failure falls back to logger).

---

### ~~48. `_make_icon_image` — no test~~ DONE

**Status:** Covered by `tests/test_tray.py::TestMakeIconImage` (2 tests:
TC-UI-ICON-001 default call returns RGBA 64x64 PIL.Image,
TC-UI-ICON-002 custom color `"#FF0000"` and `size=32` returns 32x32).

---

### ~~49. `_LabelSuggester.suggest` — no test~~ DONE

**Status:** Covered by `tests/test_tray.py::TestLabelSuggester` (4 tests:
TC-UI-SUG-001 successful suggestion returns `(label, confidence)` tuple,
TC-UI-SUG-002 no events from AW returns `None`,
TC-UI-SUG-003 no features built returns `None`,
TC-UI-SUG-004 prediction exception returns `None`).

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

### ~~19. `infer.monitor` — no test file exists~~ DONE

**Status:** Covered by `tests/test_infer_monitor.py` (17 tests):
- **Models** (3 tests): TC-MON-001 DriftTrigger 5 values all strings,
  TC-MON-002 DriftAlert construction with all fields,
  TC-MON-003 DriftReport.any_critical computed correctly.
- **run_drift_check** (8 tests): TC-MON-004 no drift identical ref/cur,
  TC-MON-005 feature PSI drift on shifted column,
  TC-MON-006 reject rate increase produces critical alert,
  TC-MON-007 entropy spike with ref/cur probs,
  TC-MON-008 entropy_drift None when probs omitted,
  TC-MON-009 class distribution shift detected,
  TC-MON-010 telemetry_snapshot always populated,
  TC-MON-011 any_critical True on reject rate increase.
- **auto_enqueue_drift_labels** (4 tests): TC-MON-012 no alerts returns 0,
  TC-MON-013 alerts present enqueues items,
  TC-MON-014 limit respected, TC-MON-014b lowest confidence selected.
- **write_drift_report** (2 tests): TC-MON-015 round-trip JSON,
  TC-MON-016 parent dirs created.

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

### ~~21. `infer.batch` — `predict_proba`, segment I/O, and batch+taxonomy untested~~ DONE
**File:** `src/taskclf/infer/batch.py`
**Doc:** `docs/api/infer/batch.md`
**Existing tests:** `tests/test_infer_batch_reject.py` (reject paths),
`tests/test_infer_pipeline.py` (CSV output, hysteresis, calibrator).

#### ~~21a. `predict_proba()` (lines 47–65) — no direct test~~ DONE

**Status:** Covered by `tests/test_infer_batch_segments.py::TestPredictProba` (3 tests:
TC-BATCH-001 shape is (N, 8),
TC-BATCH-002 rows sum to ~1.0,
TC-BATCH-003 cat_encoders=None falls back to integer encoding).

#### ~~21b. `write_segments_json()` / `read_segments_json()` round-trip (lines 249–289) — no test~~ DONE

**Status:** Covered by `tests/test_infer_batch_segments.py::TestSegmentJsonRoundTrip` (3 tests:
TC-BATCH-SEG-001 round-trip preserves all fields,
TC-BATCH-SEG-002 empty segments list writes/reads [],
TC-BATCH-SEG-003 non-existent parent dirs created).

#### ~~21c. `run_batch_inference()` with taxonomy (lines 107–198) — unit test missing~~ DONE

**Status:** Covered by `tests/test_infer_batch_segments.py::TestRunBatchInferenceTaxonomy` (3 tests:
TC-BATCH-004 taxonomy populates mapped_labels with correct length,
TC-BATCH-005 taxonomy populates mapped_probs with dicts summing to ~1.0,
TC-BATCH-006 without taxonomy mapped_labels and mapped_probs are None).

#### ~~21d. `run_batch_inference()` with `calibrator_store` — no test~~ DONE

**Status:** Covered by `tests/test_infer_batch_segments.py::TestRunBatchInferenceCalibratorStore` (2 tests:
TC-BATCH-007 per-user calibration changes core_probs vs identity,
TC-BATCH-008 user_id column dispatches per-user calibrators).

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

### ~~23. `infer.resolve` — `ActiveModelReloader` reload path untested~~ DONE

**Status:** Covered by `tests/test_infer_resolve.py::TestActiveModelReloader` (4 new tests):
- TC-RELOAD-001 mtime change triggers successful reload (mocked `load_model_bundle`)
- TC-RELOAD-002 mtime change + load failure returns None (no model file)
- TC-RELOAD-003 `_last_mtime` updated after reload (second check returns None)
- TC-RELOAD-004 missing `active.json` → `_current_mtime()` is None, no reload

---

### ~~24. `infer.online` — `OnlinePredictor` with taxonomy and calibrator store untested~~ DONE

**Status:** Covered by `tests/test_infer_online.py` (6 new tests):
- **TestOnlinePredictorTaxonomy** (2 tests: TC-ONLINE-001 mapped_label_name comes from
  taxonomy bucket names, TC-ONLINE-002 mapped_probs keys are bucket names summing to ~1.0)
- **TestOnlinePredictorCalibratorStore** (1 test: TC-ONLINE-003 per-user calibration
  via CalibratorStore changes confidence vs identity)
- **TestEncodeValue** (2 tests: TC-ONLINE-004 known categorical value returns encoded
  int and unknown returns -1.0, TC-ONLINE-005 non-categorical None returns 0.0)
- **TestOnlinePredictorRejectSegments** (1 test: TC-ONLINE-006 rejected predictions
  produce segments with MIXED_UNKNOWN label)

---

### ~~25. `infer.prediction` — boundary validation untested~~ DONE

**Status:** Covered by `tests/test_infer_pipeline.py::TestWindowPrediction` (5 new tests:
TC-PRED-001 confidence<0, TC-PRED-002 confidence>1.0, TC-PRED-003 core_label_id<0,
TC-PRED-004 core_label_id>7, TC-PRED-005 frozen model immutability).

---

## Low Priority

### ~~26. `infer.baseline` — `_safe_float()` and custom thresholds untested~~ DONE

**Status:** Covered by `tests/test_infer_baseline.py` (7 new tests):
- `TestSafeFloat` (5 tests: TC-BASE-001 None returns default,
  TC-BASE-002 NaN returns default, TC-BASE-003 valid float returned,
  TC-BASE-004 non-numeric string returns default,
  TC-BASE-005 integer coerced to float)
- `TestCustomThresholds` (2 tests: TC-BASE-006 lowered idle threshold
  prevents BreakIdle classification, TC-BASE-007 raised scroll_high
  prevents ReadResearch classification via `run_baseline_inference`)

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

### ~~25. Export functions — parent directory creation~~ DONE

**Status:** Covered by `tests/test_report.py::TestExportParentDirCreation` (3 tests:
TC-RPT-MKDIR-001 export_report_json with nested path creates dirs,
TC-RPT-MKDIR-002 export_report_csv with nested path creates dirs,
TC-RPT-MKDIR-003 export_report_parquet with nested path creates dirs).

### ~~26. `_breakdown_to_rows()` — row content correctness~~ DONE

**Status:** Covered by `tests/test_report.py::TestBreakdownToRows` (3 tests:
TC-RPT-ROWS-001 core rows sorted alphabetically by label,
TC-RPT-ROWS-002 minutes values match core_breakdown rounded to 2dp,
TC-RPT-ROWS-003 date propagated into every row).

### ~~27. Export value correctness in CSV and Parquet~~ DONE

**Status:** Covered by `tests/test_report.py::TestExportValueCorrectness` (4 tests:
TC-RPT-CSVVAL-001 CSV minutes match core_breakdown,
TC-RPT-CSVVAL-002 CSV date matches report.date in all rows,
TC-RPT-CSVVAL-003 Parquet minutes match core_breakdown,
TC-RPT-CSVVAL-004 Parquet date matches report.date in all rows).

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

### ~~31. `labels.store.generate_label_summary()` — missing edge cases~~ DONE

**Status:** Covered by `tests/test_labels_store.py::TestGenerateLabelSummaryEdgeCases` (4 tests:
TC-LABEL-SUM-001 no app_id column → top_apps == [],
TC-LABEL-SUM-002 no session_id column → session_count == 0,
TC-LABEL-SUM-003 no input rate columns → all means None,
TC-LABEL-SUM-004 rate columns all NaN → means None).

---

### ~~32. `labels.queue.enqueue_drift()` — missing edge cases~~ DONE

**Status:** Covered by `tests/test_labels_queue.py::TestEnqueueDriftEdgeCases` (3 tests:
TC-LABEL-QD-001 dedup same bucket re-enqueued returns 0,
TC-LABEL-QD-002 missing optional fields → confidence/predicted_label None,
TC-LABEL-QD-003 mixed new/existing returns only new count).

---

### ~~33. `labels.projection.project_blocks_to_windows()` — missing branch coverage~~ DONE

**Status:** Covered by `tests/test_labels_projection.py` (2 new tests:
TC-LABEL-PROJ-001 auto-derived `bucket_end_ts` when column is missing,
TC-LABEL-PROJ-002 same-label multi-block covering labels window correctly).

---
---

# TODO — Adapters Test Coverage

Missing tests identified by auditing `src/taskclf/adapters/` against
`tests/test_adapters_aw.py` and `tests/test_aw_input_events.py`.

**Already tested:**
- `parse_aw_export` — 7 tests in `test_adapters_aw.py::TestParseAWExport`
- `parse_aw_input_export` — 5 tests in `test_aw_input_events.py::TestParseAWInputExport`
- `normalize_app` — 8 tests in `test_adapters_aw.py::TestNormalizeApp`
- `_raw_event_to_aw_event` — 5 tests in `test_adapters_aw.py::TestRawEventToAWEvent`
- `AWEvent` protocol — `test_adapters_aw.py::TestEventProtocol`
- `AWInputEvent` — 3 tests in `test_aw_input_events.py::TestAWInputEvent`

---

## High Priority — REST client functions with zero coverage

### ~~50. `list_aw_buckets()` — no test~~ DONE

**Status:** Covered by `tests/test_adapters_aw.py::TestListAwBuckets` (3 tests:
TC-AW-REST-001 returns bucket dict from mocked API,
TC-AW-REST-002 URL trailing slash normalized,
TC-AW-REST-003 empty response returns empty dict).

---

### ~~51. `find_window_bucket_id()` — no test~~ DONE

**Status:** Covered by `tests/test_adapters_aw.py::TestFindWindowBucketId` (4 tests:
TC-AW-REST-004 found currentwindow bucket returns correct ID,
TC-AW-REST-005 no matching bucket raises ValueError,
TC-AW-REST-006 multiple buckets picks correct one,
TC-AW-REST-007 empty buckets dict raises ValueError).

---

### ~~52. `find_input_bucket_id()` — no test~~ DONE

**Status:** Covered by `tests/test_adapters_aw.py::TestFindInputBucketId` (3 tests:
TC-AW-REST-008 found os.hid.input bucket returns ID,
TC-AW-REST-009 no matching bucket returns None,
TC-AW-REST-010 multiple buckets picks correct one).

---

### ~~53. `fetch_aw_events()` — no test~~ DONE

**Status:** Covered by `tests/test_adapters_aw.py::TestFetchAwEvents` (4 tests:
TC-AW-REST-011 returns sorted AWEvent list from mocked API,
TC-AW-REST-012 naive-UTC timestamp builds ISO query string with Z suffix,
TC-AW-REST-013 TZ-aware timestamp uses isoformat without extra Z,
TC-AW-REST-014 empty response returns []).

---

### ~~54. `fetch_aw_input_events()` — no test~~ DONE

**Status:** Covered by `tests/test_adapters_aw.py::TestFetchAwInputEvents` (3 tests:
TC-AW-REST-015 returns sorted AWInputEvent list,
TC-AW-REST-016 empty response returns [],
TC-AW-REST-017 URL construction includes bucket_id and start/end params).

---
---

# TODO — Config Test Coverage

Missing edge cases identified by auditing `src/taskclf/core/config.py` against
`tests/test_core_config.py`.

**Already tested:** auto-generated UUID, stable across reloads, immutable via update,
default username, set username, empty username rejection, as_dict basic, update username,
update empty rejection, update no-op, corrupt config fallback, parent dir creation,
username change preserves user_id.

---

### ~~55. `UserConfig.update()` — custom keys not tested~~ DONE

**Status:** Covered by `tests/test_core_config.py` (2 tests:
TC-CFG-001 update with custom keys persists and survives reload,
TC-CFG-002 as_dict includes custom keys alongside user_id and username).

---

### ~~56. `UserConfig.update()` — whitespace-only username via update~~ DONE

**Status:** Covered by `tests/test_core_config.py` (1 test:
TC-CFG-003 update with whitespace-only username raises ValueError).

---
---

# TODO — Model IO Test Coverage

Missing tests identified by auditing `src/taskclf/core/model_io.py` against
`tests/test_core_model_io.py`.

**Already tested:**
- `save_model_bundle` — writes model, metadata, metrics, confusion matrix, categorical encoders
- `load_model_bundle` — missing metadata, schema hash mismatch, label set mismatch, validation flags
- `build_metadata` — data provenance default
- `generate_run_id` — format and uniqueness

---

### ~~57. `core.model_io` — `save_model_bundle()` FileExistsError path untested~~ DONE

**Status:** Covered by `tests/test_core_model_io.py::TestSaveModelBundleFileExistsError` (1 test:
TC-MODEL-006 pre-created run dir triggers FileExistsError).

---

### ~~58. `core.model_io` — `load_model_bundle()` cat_encoders round-trip untested~~ DONE

**Status:** Covered by `tests/test_core_model_io.py::TestLoadCatEncodersRoundTrip` (2 tests:
TC-MODEL-007 cat_encoders present in bundle are loaded with correct classes,
TC-MODEL-008 missing categorical_encoders.json returns None).

---

### ~~59. `core.model_io` — `build_metadata()` reject_threshold and dataset_hash untested~~ DONE

**Status:** Covered by `tests/test_core_model_io.py::TestBuildMetadataParams` (3 tests:
TC-MODEL-009 explicit reject_threshold stored in metadata,
reject_threshold defaults to None,
TC-MODEL-010 dataset_hash round-trips).

---
---

# TODO — Retrain Model Test Coverage

Missing constructor/validation tests identified by auditing
`src/taskclf/train/retrain.py` Pydantic models against `tests/test_retrain.py`.

**Already tested:**
- `RetrainConfig` — load_retrain_config YAML round-trip
- `RegressionResult` — via check_regression_gates / check_candidate_gates
- `RetrainResult` — via run_retrain_pipeline

---

### ~~60. `train.retrain` — `TrainParams`, `DatasetSnapshot`, `RegressionGate` no dedicated constructor tests~~ DONE

**Status:** Covered by `tests/test_retrain.py` (6 tests):
- `TestTrainParamsConstructor` (2 tests: TC-RETRAIN-031 defaults match DEFAULT_NUM_BOOST_ROUND,
  TC-RETRAIN-032 custom num_boost_round and class_weight)
- `TestDatasetSnapshotConstructor` (2 tests: TC-RETRAIN-033 all fields stored,
  TC-RETRAIN-034 frozen immutability)
- `TestRegressionGateConstructor` (2 tests: TC-RETRAIN-035 name/passed/detail stored,
  TC-RETRAIN-036 frozen immutability)

---
---

# TODO — Taxonomy Test Coverage

Missing tests identified by auditing `src/taskclf/infer/taxonomy.py` against
`tests/test_infer_taxonomy.py`.

**Already tested:**
- `TaxonomyBucket` — validation of core_labels and hex color
- `TaxonomyConfig` — YAML round-trip, resolver, batch, default taxonomy
- `TaxonomyReject` — used via TaxonomyConfig
- `TaxonomyResolver` — single/batch resolution
- CLI commands — validate/show/init

---

### ~~61. `infer.taxonomy` — `TaxonomyDisplay` no dedicated test~~ DONE

**Status:** Covered by `tests/test_infer_taxonomy.py::TestTaxonomyDisplay` (3 tests:
TC-TAX-008 defaults (show_core_labels=False, default_view="mapped", color_theme="default"),
TC-TAX-009 custom values (show_core_labels=True, default_view="core", color_theme="dark"),
TC-TAX-010 frozen immutability).

---
---

### ~~62. `infer.online` — `run_online_loop()` direct test~~ DONE

**Status:** Covered by `tests/test_infer_online.py::TestRunOnlineLoop::test_run_online_loop_single_poll_writes_outputs`
(TC-ONLINE-007: stubbed AW client + single poll → predictions.csv and segments.json written; session_start propagated).

---
---

# TODO — Part G: Automated Tray Testing (no manual interaction)

The pystray GUI loop (`Icon.run()`) is inherently interactive, but
the tray system can be tested automatically at several levels without
a human clicking the icon. Existing `tests/test_tray.py` (~3100 lines)
covers business logic extensively; the items below target the remaining
integration and GUI-adjacent gaps.

**Existing coverage (for reference):**
- `ActivityMonitor` transition detection (7+ tests)
- `TrayLabeler` event publishing: `_handle_transition`, `_handle_poll`,
  `_publish_status` (9 tests, TC-UI-TRAY-001 through TC-UI-TRAY-009)
- `_build_menu_items` / `_build_model_submenu` — including real
  `pystray.Menu(callable)` integration (TestDynamicModelMenuRefresh)
- `_send_desktop_notification` (3 tests, mocked `osascript`)
- `_make_icon_image` (2 tests)
- `_LabelSuggester.suggest` (4 tests)
- Crash handler, pause/resume, import/export, model switching, retrain
- Label creation via tray callbacks (3 tests)

**What is NOT tested (remaining):**
- End-to-end event flow with a real server: tray → EventBus →
  FastAPI WebSocket → client (full stack with real uvicorn)
- Real pystray icon start/stop (visual rendering on macOS/Linux)

---

## High Priority

### ~~63. `--no-tray` integration test — full stack without GUI~~ DONE

**Status:** Covered by `tests/test_tray.py::TestNoTrayIntegration` (5 tests):
- TC-TRAY-NOTRAY-001 `no_tray + browser` → `_start_ui_embedded` called
- TC-TRAY-NOTRAY-002 `no_tray + browser=False` → `_start_ui_subprocess` called
- TC-TRAY-NOTRAY-003 `monitor.run()` invoked from a daemon thread (verified
  via `threading.current_thread().name`)
- TC-TRAY-NOTRAY-004 `KeyboardInterrupt` triggers `monitor.stop()` and
  `_cleanup_ui()` (mocked `threading.Thread`/`threading.Event` to avoid
  C-level blocking in `Event.wait()`)
- TC-TRAY-NOTRAY-005 `_cleanup_ui` registered with `atexit`

**Note:** Tests mock out UI startup (`_start_ui_embedded`/`_start_ui_subprocess`),
logging, and `atexit` to focus on wiring correctness. AW polling is not exercised
(monitor.run is mocked); the existing `TestActivityTransitionDetection` and
`TestTrayEventPublishing` suites cover that path.

---

### ~~64. Mock `pystray.Icon.run()` smoke test — `_run_inner()` lifecycle~~ DONE

**Status:** Covered by `tests/test_tray.py::TestRunInnerSmoke` (4 tests):
- TC-TRAY-RUN-001 `pystray.Icon` constructed with name `"taskclf"`, PIL Image,
  title `"taskclf"`, and a non-None menu; `icon.run()` called exactly once
- TC-TRAY-RUN-002 `_cleanup_ui` called in the finally block after `icon.run()`
  returns (verified via call ordering list)
- TC-TRAY-RUN-003 `_cleanup_ui` called even when `icon.run()` raises
  `RuntimeError`
- TC-TRAY-RUN-004 menu passed to `pystray.Icon` is a `pystray.Menu` instance
  whose `.items` access resolves to non-empty items (verifies the callable
  pattern `pystray.Menu(self._build_menu_items)`)

**Technique:** Patches `pystray.Icon` at the module level (not on `tray.py`)
since `_run_inner()` imports pystray lazily. UI startup, monitor, logging,
and atexit are all mocked.

---

## Medium Priority

### ~~65. Menu structure snapshot test~~ DONE

**Status:** Covered by `tests/test_tray.py::TestMenuStructureSnapshot` (8 tests):
- TC-TRAY-MENU-001 top-level menu has all 14 expected items in correct order
  (Open Dashboard, Pause, separators, Label Stats, Import/Export, Model,
  Status, Open Data Folder, Edit Config, Report Issue, Quit)
- TC-TRAY-MENU-002 separators appear at positions 2, 6, 12
- TC-TRAY-MENU-003 "Model" is the only item with a submenu
- TC-TRAY-MENU-004 "Open Dashboard" has `default=True`
- TC-TRAY-MENU-005 Pause/Resume label changes dynamically with monitor state
  (pystray resolves callable text via `.text` property)
- TC-TRAY-MENU-006 Model submenu with empty `models_dir` shows "(no models
  found)", "Reload Model", "Check Retrain"
- TC-TRAY-MENU-007 Model submenu with bundles lists IDs plus "(No Model)",
  "Reload Model", "Check Retrain"
- TC-TRAY-MENU-008 total top-level item count is 14 (11 items + 3 separators)

---

### ~~66. Crash handler integration test~~ DONE (already covered)

**Status:** Already covered by `tests/test_tray.py::TestTrayCrashHandler` (5 tests):
- `test_run_writes_crash_report_on_exception` — `_run_inner` raises →
  crash file written with exception details
- `test_run_attempts_desktop_notification_on_crash` — notification called
  with `"taskclf crashed"` title and crash file path
- `test_run_reraises_original_exception` — original exception propagates
- `test_system_exit_passes_through` — `SystemExit` not caught
- `test_keyboard_interrupt_passes_through` — `KeyboardInterrupt` not caught

These tests mock `_run_inner` with `side_effect=RuntimeError("boom")` and
call `labeler.run()`, which IS the full `run()` → `_run_inner()` →
exception → `write_crash_report()` → `_send_desktop_notification()` chain.
No additional test needed.

---

## Low Priority

### 67. Real pystray start/stop smoke test (developer machines only)

**File:** `tests/test_tray.py` (new class `TestPystrayRealSmoke`)
**Covers:** `pystray.Icon.run(setup=...)` with immediate `icon.stop()`

Uses pystray's `setup` callback to stop the icon immediately after
it becomes visible. Validates that the icon renders without crashing
on the host OS. Skipped in CI (requires a display server).

**Test plan:**
1. Mark with `@pytest.mark.skipif(os.environ.get("CI"), reason="needs display")`.
2. Create a real `pystray.Icon` with `_make_icon_image()` and a
   minimal menu.
3. Pass `setup=lambda icon: icon.stop()` to `icon.run()`.
4. Assert the function returns without error.
5. Optionally verify `icon.visible` was `True` at some point
   (pystray sets this before calling setup).

**Why it matters:** Catches pystray/Pillow version incompatibilities
and OS-specific rendering issues that mocks cannot detect. Safe to
run locally; harmless to skip in CI.
