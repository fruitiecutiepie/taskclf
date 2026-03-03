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

### 1. `core.drift` — no test file exists
**File:** `src/taskclf/core/drift.py`
**Doc:** `docs/api/core/drift.md`

Six public functions and six result models have zero test coverage.

| Function | Signature | What to test |
|---|---|---|
| `compute_psi(reference, current, bins=10)` | Returns `float` (PSI) | Identical distributions → ~0.0; shifted distribution → >0.2; empty/constant arrays → 0.0; NaN handling (uses `np.isfinite` filter) |
| `compute_ks(reference, current)` | Returns `KsResult(statistic, p_value)` | Same distribution → high p-value; different distributions → low p-value; `is_significant(alpha)` helper; empty arrays → `KsResult(0.0, 1.0)` |
| `feature_drift_report(ref_df, cur_df, numerical_features)` | Returns `FeatureDriftReport` | Multi-feature report; `flagged_features` populated when PSI > threshold or KS significant; missing columns skipped; `timestamp` is set |
| `detect_reject_rate_increase(ref_labels, cur_labels, threshold=0.10)` | Returns `RejectRateDrift` | No increase → `is_flagged=False`; increase >= threshold → flagged; empty sequences → rate 0.0; custom `reject_label` |
| `detect_entropy_spike(ref_probs, cur_probs, spike_multiplier=2.0)` | Returns `EntropyDrift` | Confident ref + uncertain cur → flagged; same entropy → not flagged; zero-ref entropy → not flagged (guarded by `_EPS`); empty arrays |
| `detect_class_shift(ref_labels, cur_labels, threshold=0.15)` | Returns `ClassShiftResult` | Same distribution → not flagged; class disappears/appears → flagged; `shifted_classes` lists only shifted ones; `max_shift` correct |

Result models to validate: `KsResult.is_significant()`, `FeatureDriftResult`, `FeatureDriftReport`, `RejectRateDrift`, `EntropyDrift`, `ClassShiftResult`.

Suggested test IDs: `TC-DRIFT-001` through `TC-DRIFT-006`.

---

### 2. `core.telemetry` — no test file exists
**File:** `src/taskclf/core/telemetry.py`
**Doc:** `docs/api/core/telemetry.md`

| Function / Class | What to test |
|---|---|
| `compute_telemetry(features_df, *, labels, confidences, core_probs, user_id)` | Returns `TelemetrySnapshot`; `feature_missingness` computed from `NUMERICAL_FEATURES`; `confidence_stats` (mean/median/p5/p95/std) when `confidences` provided, `None` when omitted; `reject_rate` counts `MIXED_UNKNOWN` labels; `class_distribution` sums to ~1.0; `mean_entropy` from `core_probs`; `window_start`/`window_end` from `bucket_start_ts` column; empty DataFrame → `total_windows=0`, no crash |
| `TelemetryStore(store_dir)` | Creates directory on init; `append(snapshot)` writes JSONL, returns path; file named `telemetry_{user_id}.jsonl` or `telemetry_global.jsonl`; `read_recent(n, user_id=)` returns last N snapshots; `read_range(start, end)` filters by timestamp; round-trip: append then read back; empty store → empty list |
| `ConfidenceStats` | Pydantic model with `mean`, `median`, `p5`, `p95`, `std` — basic construction |
| `TelemetrySnapshot` | Model with defaults (`reject_rate=0.0`, `mean_entropy=0.0`, `schema_version="features_v1"`); serialization round-trip via `model_dump_json` / `model_validate_json` |

Suggested test IDs: `TC-TELEM-001` through `TC-TELEM-004`.

---

### 3. `core.logging` — no test file exists
**File:** `src/taskclf/core/logging.py`
**Doc:** `docs/api/core/logging.md`

| Function / Class | What to test |
|---|---|
| `redact_message(message)` | Redacts each key in `_SENSITIVE_KEYS`: `raw_keystrokes`, `raw_keys`, `window_title_raw`, `window_title`, `clipboard_content`, `typed_text`, `im_content`, `full_url`; handles `key=value`, `key: value`, `key="quoted value"` formats; non-sensitive keys pass through untouched; multiple sensitive keys in one message all redacted; case-insensitive matching |
| `SanitizingFilter` | `filter(record)` rewrites `record.msg` and clears `record.args`; when `record.args` is `None`, redacts `record.msg` directly; always returns `True` (never drops records) |
| `install_sanitizing_filter(logger, handler_level=False)` | Attaches filter to logger (default) or to each handler (`handler_level=True`); defaults to root logger when `logger=None`; returns the `SanitizingFilter` instance |

Suggested test IDs: `TC-LOG-001` through `TC-LOG-003`.

---

### 4. `core.defaults` — no test file exists
**File:** `src/taskclf/core/defaults.py`
**Doc:** `docs/api/core/defaults.md`

Low priority — these are `Final` constants. A smoke test that imports all
public names and asserts expected types (`int`, `float`, `str`) would
catch accidental deletions or type changes.

---

## Medium Priority — Gaps within tested modules

### 5. `core.time` — `generate_bucket_range()` untested
**File:** `src/taskclf/core/time.py:42-67`
**Tests:** `tests/test_core_time.py` (only covers `align_to_bucket`)

```python
def generate_bucket_range(
    start: datetime,
    end: datetime,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[datetime]:
```

| Test case | Expected |
|---|---|
| 10:00:00 → 10:05:00 (60s buckets) | 5 buckets: `[10:00, 10:01, 10:02, 10:03, 10:04]` |
| start == end | Empty list (exclusive end) |
| start > end | Empty list |
| Unaligned start/end (e.g. 10:00:37 → 10:03:12) | Aligns both, returns buckets within |
| Timezone-aware inputs | Converted to naive UTC |
| Custom `bucket_seconds` (e.g. 300) | Correct step size |

Suggested test IDs: `TC-TIME-005` through `TC-TIME-010`.

---

### 6. `core.metrics` — 4 functions untested
**File:** `src/taskclf/core/metrics.py`
**Tests:** `tests/test_core_metrics.py`

#### 6a. `reject_rate(labels, reject_label=MIXED_UNKNOWN)` → `float`
**Lines:** 97-112

| Test case | Expected |
|---|---|
| No rejects | 0.0 |
| All rejects | 1.0 |
| Empty sequence | 0.0 |
| Custom `reject_label` | Counts only that label |

#### 6b. `per_class_metrics(y_true, y_pred, label_names)` → `dict[str, dict[str, float]]`
**Lines:** 115-141

| Test case | Expected |
|---|---|
| Perfect predictions | `precision=1.0, recall=1.0, f1=1.0` per class |
| Missing class in predictions | `f1=0.0` for that class (zero_division=0) |
| All keys present | Each label has `precision`, `recall`, `f1` |

#### 6c. `compare_baselines(y_true, predictions, label_names, reject_label)` → `dict[str, dict]`
**Lines:** 144-186

| Test case | Expected |
|---|---|
| Two methods compared | Both keyed in output |
| Each entry has required keys | `macro_f1`, `weighted_f1`, `reject_rate`, `per_class`, `confusion_matrix`, `label_names` |
| `reject_label` appended to `label_names` if absent | `label_names` in output includes it |

#### 6d. `reject_rate_by_group(labels, user_ids, timestamps, reject_label, spike_multiplier)` → `dict`
**Lines:** 341-395

| Test case | Expected |
|---|---|
| No rejects anywhere | `global_reject_rate=0.0`, empty `drift_flags` |
| One user/day spikes | That group key appears in `drift_flags` |
| `per_group` keys format | `"user_id\|YYYY-MM-DD"` |
| Each group has `reject_rate`, `total`, `rejected` | Correct counts |

---

### 7. `core.types` — 3 areas untested
**File:** `src/taskclf/core/types.py`
**Tests:** `tests/test_core_types.py`

#### 7a. `Event` protocol
**Lines:** 14-38

`Event` is a `@runtime_checkable` `Protocol`. No test verifies that an object
satisfying the protocol passes `isinstance(obj, Event)`, or that a
non-conforming object fails.

| Test case | Expected |
|---|---|
| Object with all required properties | `isinstance(obj, Event)` is `True` |
| Object missing a property | `isinstance(obj, Event)` is `False` |

#### 7b. `LabelSpan.confidence` NaN coercion
**Lines:** 222-226

```python
@field_validator("confidence", mode="before")
@classmethod
def _nan_confidence_to_none(cls, v: object) -> object:
    if isinstance(v, float) and math.isnan(v):
        return None
    return v
```

| Test case | Expected |
|---|---|
| `confidence=float('nan')` | Stored as `None` |
| `confidence=0.8` | Stored as `0.8` |
| `confidence=None` | Stored as `None` |

#### 7c. `LabelSpan.extend_forward` field
**Line:** 237

`extend_forward: bool = Field(default=False)`. No test verifies the default
or explicit `True` construction.

---

### 8. `core.validation` — 2 documented checks untested
**File:** `src/taskclf/core/validation.py`
**Tests:** `tests/test_core_validation.py`

#### 8a. Dominant-value warning (`_check_distributions`)
**Lines:** 285-296

Documented: columns where >90% of values are identical should emit a
`dominant_value` warning. Not tested.

| Test case | Expected |
|---|---|
| Column with 95% identical values | `Finding(check="dominant_value")` warning |
| Column with 80% identical values | No warning |

#### 8b. Session boundary gap warning (`_check_session_boundaries`)
**Lines:** 244-268

Documented: session changes with very small gaps (<= 60s) emit a
`session_boundary` warning. Not tested.

| Test case | Expected |
|---|---|
| Session change with 60s gap | Warning emitted |
| Session change with 300s gap | No warning |
| Same session throughout | No warning |

---

## Low Priority

### 9. `core.model_io` — `generate_run_id()` untested
**File:** `src/taskclf/core/model_io.py:59-67`
**Tests:** `tests/test_core_model_io.py`

```python
def generate_run_id() -> str:
    now = datetime.now(UTC)
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{now.strftime('%Y-%m-%d_%H%M%S')}_run-{suffix}"
```

| Test case | Expected |
|---|---|
| Format matches `YYYY-MM-DD_HHMMSS_run-XXXX` | Regex: `r"\d{4}-\d{2}-\d{2}_\d{6}_run-\d{4}"` |
| Two calls produce different IDs (probabilistic) | Assert not equal |

---
---

# TODO — Features Test Coverage

Missing tests identified by auditing `docs/api/features/` against
`tests/test_features_*.py` and `src/taskclf/features/*.py`.

---

## High Priority — Completely untested public functions

### 10. `features.windows` — `compute_rolling_app_switches()` untested
**File:** `src/taskclf/features/windows.py:54-75`
**Tests:** `tests/test_features_windows.py` (only covers `app_switch_count_in_window`)

```python
def compute_rolling_app_switches(
    sorted_events: Sequence[Event],
    sorted_buckets: Sequence[dt.datetime],
    window_minutes: int = DEFAULT_APP_SWITCH_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[int]:
```

Convenience wrapper that calls `app_switch_count_in_window` for each bucket.
Zero test coverage.

| Test case | Expected |
|---|---|
| 3 buckets, 4 apps spread across them | List of 3 ints matching per-bucket switch counts |
| Single bucket | List with one element |
| Empty `sorted_buckets` | Empty list |
| Empty `sorted_events` | All-zero list (one 0 per bucket) |
| Custom `window_minutes` (e.g. 15) | Wider window captures more switches |
| Custom `bucket_seconds` (e.g. 300) | Correct window end (`bucket_ts + 300s`) |

Also missing from `app_switch_count_in_window` tests:
- Custom `window_minutes` parameter
- Custom `bucket_seconds` parameter

Suggested test IDs: `TC-FEAT-WIN-001` through `TC-FEAT-WIN-006`.

---

### 11. `features.build` — `generate_dummy_features()` untested
**File:** `src/taskclf/features/build.py:48-138`
**Tests:** none

```python
def generate_dummy_features(
    date: dt.date,
    n_rows: int = DEFAULT_DUMMY_ROWS,
    *,
    user_id: str = "dummy-user-001",
    device_id: str | None = None,
) -> list[FeatureRow]:
```

Generates synthetic `FeatureRow` instances for testing/demo.
Used by `build_features_for_date`.

| Test case | Expected |
|---|---|
| Default call | Returns `DEFAULT_DUMMY_ROWS` rows |
| Custom `n_rows=5` | Returns exactly 5 rows |
| All rows pass `FeatureSchemaV1` validation | No `ValidationError` |
| `bucket_start_ts` spans hours 9–17 of the given date | All timestamps on the correct date |
| Custom `user_id` and `device_id` | All rows carry those values |
| `schema_version` and `schema_hash` match `FeatureSchemaV1` | Correct on every row |
| `session_id` is a valid hash string | Non-empty, consistent within call |
| Dynamics fields are populated (not all None) | At least the rolling means after the first few rows |
| `n_rows=0` | Returns empty list |

Suggested test IDs: `TC-FEAT-BUILD-001` through `TC-FEAT-BUILD-009`.

---

### 12. `features.build` — `build_features_for_date()` untested
**File:** `src/taskclf/features/build.py:417-439`
**Tests:** none

```python
def build_features_for_date(date: dt.date, data_dir: Path) -> Path:
```

End-to-end pipeline: generates dummy features, validates against
`FeatureSchemaV1`, writes to parquet at
`data_dir/features_v1/date=YYYY-MM-DD/features.parquet`.

| Test case | Expected |
|---|---|
| Valid date + `tmp_path` | Returns a `Path` that exists and is a `.parquet` file |
| Output path matches expected structure | `data_dir/features_v1/date=YYYY-MM-DD/features.parquet` |
| Parquet readable with correct columns | `pd.read_parquet(path)` has all `FeatureRow` columns |
| Row count matches `DEFAULT_DUMMY_ROWS` | `len(df) == DEFAULT_DUMMY_ROWS` |

Suggested test IDs: `TC-FEAT-BUILD-010` through `TC-FEAT-BUILD-013`.

---

## Medium Priority — Edge cases missing in tested modules

### 13. `features.text` — `title_hash_bucket()` edge cases untested
**File:** `src/taskclf/features/text.py:31-56`
**Tests:** `tests/test_features_text.py` (only tests default `n_buckets=256` with valid hex)

```python
def title_hash_bucket(title_hash: str, n_buckets: int = 256) -> int:
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    try:
        return int(title_hash, 16) % n_buckets
    except ValueError:
        return abs(hash(title_hash)) % n_buckets
```

| Test case | Expected |
|---|---|
| `n_buckets=0` | Raises `ValueError` |
| `n_buckets=-1` | Raises `ValueError` |
| Non-hex input (e.g. `"not-hex-string"`) | Falls back to `hash()`, returns `int` in `[0, n_buckets)` |
| Custom `n_buckets=10` | Result in `[0, 10)` |
| `n_buckets=1` | Always returns `0` |

Missing from `featurize_title` tests:

| Test case | Expected |
|---|---|
| Empty string title | Returns 12-char hex (no crash) |
| Different salts, same title | Different hashes |

Suggested test IDs: `TC-FEAT-TEXT-001` through `TC-FEAT-TEXT-007`.

---

### 14. `features.build` — `_aggregate_input_for_bucket()` no direct unit tests
**File:** `src/taskclf/features/build.py:154-219`
**Tests:** only exercised indirectly via `build_features_from_aw_events` in `test_features_from_aw.py`

Private function, but complex enough to warrant direct testing for
occupancy calculation (`active_seconds_keyboard/mouse/any`),
`max_idle_run_seconds` tracking, and `event_density`.

| Test case | Expected |
|---|---|
| Empty `bucket_input` | All values `None` (matches `_INPUT_NULL_FIELDS`) |
| Single active input event | `active_seconds_any > 0`, `max_idle_run_seconds == 0.0` |
| Mixed active/idle events | `max_idle_run_seconds` equals longest idle run |
| Only keyboard activity (no mouse) | `active_seconds_keyboard > 0`, `active_seconds_mouse == 0.0` |
| Only mouse activity (no keyboard) | `active_seconds_mouse > 0`, `active_seconds_keyboard == 0.0` |
| `event_density` formula | `active_event_count / active_seconds_any` |

Suggested test IDs: `TC-FEAT-AGG-001` through `TC-FEAT-AGG-006`.

---

## Low Priority — Minor edge case gaps

### 15. `features.dynamics` — `compute_batch()` edge cases
**File:** `src/taskclf/features/dynamics.py:115-140`
**Tests:** `tests/test_features_dynamics.py` (tested with 5-element and 4-element lists)

| Test case | Expected |
|---|---|
| Empty lists `([], [], [])` | Returns empty list |
| Single-element lists | One dict, all deltas `None` |
| Custom `rolling_15` parameter | Rolling-15 window uses the custom value |

Suggested test IDs: `TC-FEAT-DYN-001` through `TC-FEAT-DYN-003`.

---

### 16. `features.sessions` — `session_start_for_bucket()` edge case
**File:** `src/taskclf/features/sessions.py:52-69`
**Tests:** `tests/test_features_sessions.py`

| Test case | Expected |
|---|---|
| `bucket_ts` before all session starts | Returns first session start (the `max(idx, 0)` guard) |

Suggested test ID: `TC-FEAT-SESS-001`.

---

### 17. `features.domain` — `classify_domain()` edge cases
**File:** `src/taskclf/features/domain.py:109-138`
**Tests:** `tests/test_features_domain.py`

| Test case | Expected |
|---|---|
| Leading/trailing whitespace on real domain (e.g. `" github.com "`) | `"code_hosting"` (`.strip()` handles it) |
| Deep subdomain (3+ levels, e.g. `"a.b.github.com"`) | `"code_hosting"` (parent match on `"github.com"`) |

Suggested test IDs: `TC-FEAT-DOM-001` through `TC-FEAT-DOM-002`.

---

### 18. `features.windows` — `app_switch_count_in_window()` edge cases
**File:** `src/taskclf/features/windows.py:18-51`
**Tests:** `tests/test_features_windows.py`

| Test case | Expected |
|---|---|
| Events entirely before the window | Returns `0` |
| Events entirely after the window | Returns `0` |

Suggested test IDs: `TC-FEAT-WIN-007` through `TC-FEAT-WIN-008`.

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
| `train.retrain` | `test_retrain.py` (16+ tests) | Well covered but `check_calibrator_update_due` has zero coverage |

---

## Medium Priority — Direct unit tests missing

### 34. `train.lgbm` — `encode_categoricals()` no direct test
**File:** `src/taskclf/train/lgbm.py:69-103`
**Tests:** only exercised indirectly through `train_lgbm` in
`test_train_evaluate.py`

```python
def encode_categoricals(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
```

Two code paths: (1) `cat_encoders=None` fits new `LabelEncoder`s for
each column in `CATEGORICAL_COLUMNS` (`app_id`, `app_category`,
`domain_category`, `user_id`); (2) pre-fitted encoders map known
values to integers and unknown values to `-1`.

| Test case | Expected |
|---|---|
| `cat_encoders=None` (fit new) | Returns encoders keyed by all 4 `CATEGORICAL_COLUMNS`; encoded columns are integer dtype |
| Pre-fitted encoder reuse | Same input → same integer codes as fitting run |
| Unknown value at inference | Value not seen during fit → encoded as `-1` |
| Output shape preserved | Encoded DataFrame has same number of rows and columns as input |
| Non-categorical columns untouched | Numeric feature columns are unchanged |

Suggested test IDs: `TC-TRAIN-LGBM-001` through `TC-TRAIN-LGBM-005`.

---

### 35. `train.lgbm` — `prepare_xy()` no direct test
**File:** `src/taskclf/train/lgbm.py:106-138`
**Tests:** only exercised indirectly through `train_lgbm`

```python
def prepare_xy(
    df: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[str, LabelEncoder]]:
```

Extracts feature matrix `X` (shape `(n, len(FEATURE_COLUMNS))`) and
encoded label vector `y` (shape `(n,)`).  Categorical columns are
integer-encoded via `encode_categoricals`, numeric `NaN`s are filled
with `0`, and labels default to `sorted(LABEL_SET_V1)` encoding.

| Test case | Expected |
|---|---|
| Output shapes | `X.shape == (n_rows, len(FEATURE_COLUMNS))`, `y.shape == (n_rows,)` |
| NaN fill | `NaN` in numeric features → `0` in `X` |
| Label encoding default | `label_encoder.classes_` == `sorted(LABEL_SET_V1)` (8 classes) |
| Pre-fitted encoder reuse | Passing `label_encoder` and `cat_encoders` → same objects returned |
| Unknown label raises | Label not in `LABEL_SET_V1` → `ValueError` from `LabelEncoder.transform` |
| `X` dtype | `np.float64` |

Suggested test IDs: `TC-TRAIN-LGBM-006` through `TC-TRAIN-LGBM-011`.

---

### 36. `train.retrain` — `check_calibrator_update_due()` untested
**File:** `src/taskclf/train/retrain.py:220-249`
**Tests:** not imported or tested in any test file (verified: not in
`test_retrain.py` imports)

```python
def check_calibrator_update_due(
    calibrator_store_dir: Path,
    cadence_days: int = DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS,
) -> bool:
```

Reads `store.json` in the calibrator directory, parses the
`created_at` timestamp, and returns `True` if the store is older
than `cadence_days`.  Gracefully returns `True` on missing file,
missing field, malformed JSON, or unparseable timestamps.

| Test case | Expected |
|---|---|
| No `store.json` file | Returns `True` (update due) |
| Fresh `store.json` (created now) | Returns `False` (not due) |
| Stale `store.json` (older than cadence) | Returns `True` |
| Missing `created_at` field in JSON | Returns `True` |
| Malformed JSON in `store.json` | Returns `True` (graceful error handling) |
| Custom `cadence_days=1` with 2-day-old store | Returns `True` |

**Setup note:** Create `store.json` with
`{"created_at": "<iso-timestamp>", ...}` in a `tmp_path` directory.
Backdate `created_at` to simulate staleness (same pattern as
`test_retrain.py::TestCheckRetrainDue.test_stale_model_is_due`).

Suggested test IDs: `TC-RETRAIN-020` through `TC-RETRAIN-025`.

---

## Low Priority — Indirect-only coverage

### 37. `train.retrain` — `find_latest_model()` no isolated test
**File:** `src/taskclf/train/retrain.py:152-184`
**Tests:** used inside `test_retrain.py::TestRetrainPipeline` pipeline
tests but never tested in isolation

```python
def find_latest_model(models_dir: Path) -> Path | None:
```

Scans `models_dir` for subdirectories with `metadata.json`, picks
the one with the latest `created_at` timestamp.  Skips unreadable
metadata files.

| Test case | Expected |
|---|---|
| Empty directory | Returns `None` |
| Single model bundle | Returns that path |
| Multiple bundles with different `created_at` | Returns the latest |
| Bundle with unreadable `metadata.json` | Skipped gracefully, other bundles still found |
| Directory does not exist | Returns `None` (the `is_dir()` guard) |

Suggested test IDs: `TC-RETRAIN-026` through `TC-RETRAIN-030`.

---
---

# TODO — UI Module Test Coverage

Missing tests identified by auditing `docs/api/ui/labeling.md` against
`tests/test_ui_server.py`, `tests/test_tray.py`, and `src/taskclf/ui/`.

**Existing test files:**
- `tests/test_ui_server.py` — covers label CRUD (GET/POST), queue (GET empty),
  config/labels, feature summary (empty), AW live (fallback), window toggle/state,
  extend_forward (7 tests), WebSocket connect
- `tests/test_tray.py` — covers `ActivityMonitor.check_transition` (7 tests),
  label span creation via tray callbacks (3 tests)

---

## High Priority — Untested REST endpoints

### 34. `PUT /api/labels` — no test
**Code:** `src/taskclf/ui/server.py:224-241` (`update_label`)
**Doc:** `docs/api/ui/labeling.md:59` — "Changes the label on an existing
span identified by `start_ts` + `end_ts`. Returns 404 if no matching
span exists."

| Test case | What to verify |
|---|---|
| Happy path (update existing label) | POST a span, then PUT with new label → 200, label changed in GET response |
| 404 when no matching span | PUT with timestamps that don't match any existing span → 404 |
| Invalid timestamp format | Malformed `start_ts` or `end_ts` → 422 |
| Verify updated label persisted | PUT, then GET /api/labels → updated label visible |

Suggested test IDs: `TC-UI-UPD-001` through `TC-UI-UPD-004`.

---

### 35. `DELETE /api/labels` — no test
**Code:** `src/taskclf/ui/server.py:245-256` (`delete_label`)
**Doc:** `docs/api/ui/labeling.md:60` — "Removes a span identified by
`start_ts` + `end_ts`. Returns 404 if no matching span exists."

| Test case | What to verify |
|---|---|
| Happy path (delete existing label) | POST a span, then DELETE → 200, `{"status": "deleted"}` |
| 404 when no matching span | DELETE with non-existent timestamps → 404 |
| Span removed from storage | POST, DELETE, then GET /api/labels → empty list |
| Invalid timestamp format | Malformed timestamps → 422 |

Suggested test IDs: `TC-UI-DEL-001` through `TC-UI-DEL-004`.

---

### 36. `POST /api/queue/{request_id}/done` — no test
**Code:** `src/taskclf/ui/server.py:280-288` (`mark_queue_done`)

| Test case | What to verify |
|---|---|
| Mark existing item "labeled" | Pre-populate queue, POST `/api/queue/<id>/done` with `{"status": "labeled"}` → `{"status": "labeled"}` |
| Mark existing item "skipped" | Same with `{"status": "skipped"}` → `{"status": "skipped"}` |
| Non-existent request_id | POST with unknown ID → `{"status": "not_found"}` |
| No queue file | POST without queue file → `{"status": "not_found"}` |

Suggested test IDs: `TC-UI-QD-001` through `TC-UI-QD-004`.

---

### 37. `GET /api/config/user` — no test
**Code:** `src/taskclf/ui/server.py:352-357` (`get_user_config`)

| Test case | What to verify |
|---|---|
| Returns default config | GET → 200, response has `user_id` (UUID) and `username` |
| `user_id` is stable across requests | Two GETs return the same `user_id` |

Suggested test IDs: `TC-UI-CU-001`, `TC-UI-CU-002`.

---

### 38. `PUT /api/config/user` — no test
**Code:** `src/taskclf/ui/server.py:359-370` (`update_user_config`)

| Test case | What to verify |
|---|---|
| Update username | PUT `{"username": "alice"}` → 200, response `username == "alice"` |
| Persisted across requests | PUT, then GET → same username |
| Empty body (no-op) | PUT `{}` → 200, config unchanged |
| `user_id` unchanged after update | PUT with `username`, `user_id` stays the same |

Suggested test IDs: `TC-UI-CU-003` through `TC-UI-CU-006`.

---

### 39. `POST /api/notification/accept` — no test
**Code:** `src/taskclf/ui/server.py:401-425` (`notification_accept`)
**Doc:** `docs/api/ui/labeling.md:51` — suggestion banner accept action.

| Test case | What to verify |
|---|---|
| Accept valid suggestion | POST with `block_start`, `block_end`, `label` → 201, provenance is `"suggestion"` |
| Invalid label | POST with `label: "NotReal"` → 422 |
| Invalid timestamps | POST with malformed ISO timestamps → 422 |
| Overlap with existing span | Create a span first, then accept overlapping suggestion → 409 |
| Label persisted | Accept, then GET /api/labels → accepted label visible |

Suggested test IDs: `TC-UI-NA-001` through `TC-UI-NA-005`.

---

### 40. `POST /api/notification/skip` — no test
**Code:** `src/taskclf/ui/server.py:396-399` (`notification_skip`)

| Test case | What to verify |
|---|---|
| Skip returns ok | POST → 200, `{"status": "skipped"}` |

Suggested test ID: `TC-UI-NS-001`.

---

### 41. `POST /api/window/show-label-grid` — no test
**Code:** `src/taskclf/ui/server.py:387-392` (`window_show_label_grid`)

| Test case | What to verify |
|---|---|
| Without window_api | POST → 200, `{"status": "ok"}` (broadcasts event only) |
| With window_api | POST → 200, `show_label_grid()` called on window_api |

Suggested test IDs: `TC-UI-WS-001`, `TC-UI-WS-002`.

---

### 42. `GET /api/labels` — `limit` parameter untested
**Code:** `src/taskclf/ui/server.py:166` — `limit: int = Query(50, ge=1, le=500)`

Existing tests never pass an explicit `limit` value.

| Test case | What to verify |
|---|---|
| `limit=1` with 3 labels | Response contains exactly 1 label |
| `limit=500` | No error, returns all (up to 500) |
| `limit=0` | 422 (violates `ge=1`) |
| `limit=501` | 422 (violates `le=500`) |

Suggested test IDs: `TC-UI-LL-001` through `TC-UI-LL-004`.

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

### 44. `ui/events.py` — `EventBus` has no dedicated test file
**File:** `src/taskclf/ui/events.py`
**Doc:** `docs/api/ui/labeling.md:61` (architecture section)

`EventBus` is used by `test_ui_server.py` as a fixture dependency but
its own behavior is never directly tested.

| Test case | What to verify |
|---|---|
| Publish → subscribe delivery | `publish()` an event, `subscribe()` context manager yields queue, `queue.get()` returns event |
| Multiple subscribers | Two `subscribe()` contexts, one publish → both queues receive event |
| Dead subscriber eviction | Fill a subscriber queue to capacity (256), publish one more → full queue evicted from `_subscribers` |
| `publish_threadsafe` with bound loop | Call from a thread, event appears in subscriber queue |
| `publish_threadsafe` with no loop bound | Call before `bind_loop()` → no error, no-op |
| `publish_threadsafe` with closed loop | Bind loop, close it, call → no error, no-op |
| Subscriber cleanup on context exit | Exit `subscribe()` context → queue removed from `_subscribers` |

Suggested test file: `tests/test_ui_events.py`.
Suggested test IDs: `TC-UI-EB-001` through `TC-UI-EB-007`.

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
- `tests/test_infer_smooth.py` — `rolling_majority`, `segmentize`
- `tests/test_infer_pipeline.py` — `WindowPrediction`, calibrators
  (identity + temperature), CSV output, `merge_short_segments`,
  `OnlinePredictor` with calibrator, batch inference with calibrator
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

### 20. `infer.calibration` — `IsotonicCalibrator`, `CalibratorStore`, and store persistence untested
**File:** `src/taskclf/infer/calibration.py`
**Doc:** `docs/api/infer/calibration.md`
**Existing tests:** `tests/test_infer_pipeline.py` covers
`IdentityCalibrator` (3 tests), `TemperatureCalibrator` (6 tests),
`save_calibrator`/`load_calibrator` for identity + temperature (3 tests).

#### 20a. `IsotonicCalibrator` (lines 82–115) — no tests

Per-class isotonic regression calibrator. Wraps one
`sklearn.isotonic.IsotonicRegression` per class.

| Test case | Expected |
|---|---|
| 1D calibrate | Input shape `(8,)` → output shape `(8,)`, sums to 1.0 |
| 2D calibrate | Input shape `(N, 8)` → output shape `(N, 8)`, each row sums to 1.0 |
| Empty regressors raises | `ValueError("regressors list must not be empty")` |
| `n_classes` property | Returns `len(regressors)` |
| Satisfies `Calibrator` protocol | `isinstance(IsotonicCalibrator(...), Calibrator)` is `True` |
| Save/load round-trip | `save_calibrator` → `load_calibrator` → same predictions as original |

**Setup:** Create fitted `IsotonicRegression` instances by calling
`.fit()` with synthetic `[0.0, 0.5, 1.0]` arrays.

#### 20b. `CalibratorStore` (lines 118–172) — no tests

Per-user calibrator registry with global fallback.

| Test case | Expected |
|---|---|
| `get_calibrator` returns per-user when present | For known user_id, returns that user's calibrator |
| `get_calibrator` falls back to global | For unknown user_id, returns global calibrator |
| `calibrate_batch` applies per-user row-by-row | Different users get different calibrations |
| `calibrate_batch` shape preserved | Input `(N, K)` → output `(N, K)` |
| `user_ids` property | Returns sorted list of per-user keys |
| Empty `user_calibrators` | All users fall back to global |

#### 20c. `save_calibrator_store()` / `load_calibrator_store()` (lines 265–326) — no tests

```python
def save_calibrator_store(store: CalibratorStore, path: Path) -> Path:
def load_calibrator_store(path: Path) -> CalibratorStore:
```

| Test case | Expected |
|---|---|
| Round-trip with temperature global + 2 per-user | Loaded store has same global + per-user calibrators |
| Round-trip with isotonic global | Isotonic calibrator survives serialization |
| Directory layout | `path/store.json`, `path/global.json`, `path/users/<uid>.json` |
| `store.json` metadata | Contains `method`, `user_count`, `user_ids` |
| Empty per-user dict | No `users/` directory created, still loads fine |

Suggested test file: extend `tests/test_infer_pipeline.py` or create
`tests/test_infer_calibration.py`.
Suggested test IDs: `TC-CAL-001` through `TC-CAL-017`.

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

### 22. `infer.smooth` — `flap_rate()` untested
**File:** `src/taskclf/infer/smooth.py:121-136`
**Doc:** `docs/api/infer/smooth.md`
**Existing tests:** `tests/test_infer_smooth.py` (covers `rolling_majority`
and `segmentize` only).

```python
def flap_rate(labels: Sequence[str]) -> float:
```

| Test case | Expected |
|---|---|
| All same labels | `0.0` |
| All different labels (e.g. `[A, B, C, D]`) | `3/4 = 0.75` |
| Single element | `0.0` |
| Empty sequence | `0.0` |
| Alternating `[A, B, A, B, A]` | `4/5 = 0.8` |
| Two labels `[A, A, A, B, B]` | `1/5 = 0.2` |

Also missing from `rolling_majority` and `segmentize` tests:

| Function | Test case | Expected |
|---|---|---|
| `rolling_majority` | Empty list | `[]` |
| `rolling_majority` | Single element | Same element returned |
| `rolling_majority` | All identical labels | All same label returned |
| `rolling_majority` | Tie-breaking (original label preserved) | When counts tie, original label kept |
| `rolling_majority` | `window=1` (no smoothing) | Output equals input |
| `segmentize` | Mismatched lengths | Raises `ValueError` |
| `segmentize` | Empty inputs | Returns `[]` |
| `segmentize` | Single bucket | One segment with `bucket_count=1` |

Suggested test IDs: `TC-SMOOTH-001` through `TC-SMOOTH-014`.

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

### 25. `infer.prediction` — boundary validation untested
**File:** `src/taskclf/infer/prediction.py`
**Doc:** `docs/api/infer/prediction.md`
**Existing tests:** `tests/test_infer_pipeline.py::TestWindowPrediction`
(4 tests: valid, sum-to-one for core/mapped probs, 8-element check).

| Test case | Expected |
|---|---|
| `confidence < 0` | Raises `ValidationError` (Field `ge=0.0`) |
| `confidence > 1.0` | Raises `ValidationError` (Field `le=1.0`) |
| `core_label_id < 0` | Raises `ValidationError` (Field `ge=0`) |
| `core_label_id > 7` | Raises `ValidationError` (Field `le=7`) |
| Frozen model (immutability) | Assigning to field raises error |

Suggested test IDs: `TC-PRED-001` through `TC-PRED-005`.

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

### 19. `_check_no_sensitive_fields()` — rejection path untested
**File:** `src/taskclf/report/export.py:21-29`
**Tests:** `tests/test_report.py` — only exercised on clean data;
no test passes data containing a forbidden key.

```python
_SENSITIVE_KEYS = frozenset({
    "raw_keystrokes", "window_title_raw",
    "clipboard_content", "clipboard",
})

def _check_no_sensitive_fields(data: dict) -> None:
    for key in data:
        if key in _SENSITIVE_KEYS:
            raise ValueError(...)
        if isinstance(data[key], dict):
            _check_no_sensitive_fields(data[key])
```

| Test case | Expected |
|---|---|
| Top-level sensitive key (e.g. `{"raw_keystrokes": "..."}`) | Raises `ValueError` mentioning the key name |
| Nested sensitive key (e.g. `{"meta": {"clipboard_content": "..."}}`) | Raises `ValueError` (recursive branch) |
| All 4 sensitive keys rejected individually | Each raises `ValueError` |
| Clean dict passes through | No exception |

Suggested test IDs: `TC-RPT-SENS-001` through `TC-RPT-SENS-004`.

---

### 20. Sensitive-field rejection via each export function
**Files:** `src/taskclf/report/export.py` — `export_report_json` (line 32),
`export_report_csv` (line 77), `export_report_parquet` (line 104)
**Tests:** `tests/test_report.py` — `TestExportJson.test_no_sensitive_fields`
only checks that clean output doesn't contain the keys; it never triggers
the `ValueError` guard.

| Test case | Expected |
|---|---|
| `export_report_json` with injected sensitive key in model dump | Raises `ValueError` before writing file |
| `export_report_csv` with injected sensitive key | Raises `ValueError` |
| `export_report_parquet` with injected sensitive key | Raises `ValueError` |

**How to test:** Monkeypatch `DailyReport.model_dump` to inject a
forbidden key, or construct a dict manually and call
`_check_no_sensitive_fields` directly.

Suggested test IDs: `TC-RPT-GUARD-001` through `TC-RPT-GUARD-003`.

---

## Medium Priority — Missing edge cases

### 21. `build_daily_report()` — non-default `bucket_seconds`
**File:** `src/taskclf/report/daily.py:79-144`
**Tests:** All existing tests use `bucket_seconds=60`.

| Test case | Expected |
|---|---|
| `bucket_seconds=300` (5-min buckets) | `total_minutes` = sum of `(bucket_count × 300 / 60)` across segments |
| `bucket_seconds=120` | `core_breakdown` minutes scale correctly |

Suggested test IDs: `TC-RPT-DAILY-001`, `TC-RPT-DAILY-002`.

### 22. `build_daily_report()` — `smoothed_labels` without `raw_labels`
**File:** `src/taskclf/report/daily.py:79-144`
**Tests:** `test_flap_rate_only_raw` tests raw-without-smoothed but
not the reverse.

| Test case | Expected |
|---|---|
| `smoothed_labels` provided, `raw_labels=None` | `flap_rate_smoothed` populated, `flap_rate_raw` is `None` |

Suggested test ID: `TC-RPT-DAILY-003`.

### 23. `_build_context_switch_stats()` — edge cases
**File:** `src/taskclf/report/daily.py:64-76`
**Tests:** Tested indirectly; empty-list and float-value paths not covered.

| Test case | Expected |
|---|---|
| Empty list `[]` (not all-None — no elements at all) | Returns `None` |
| Single-element list `[5]` | `mean=5.0, median=5.0, max_value=5, total_switches=5, buckets_counted=1` |
| Float values `[2.7, 3.1]` | Values truncated to `int` (`2, 3`); `total_switches=5` |
| `median` correctness (even count) | Median of `[1, 2, 3, 4]` → `2.5` |

Suggested test IDs: `TC-RPT-CTX-001` through `TC-RPT-CTX-004`.

### 24. Pydantic validation on report models
**Files:** `src/taskclf/report/daily.py:19-61`
**Tests:** No test verifies `Field(ge=0)` rejection.

| Test case | Expected |
|---|---|
| `ContextSwitchStats(mean=-1, ...)` | Raises `ValidationError` |
| `DailyReport(total_minutes=-1, ...)` | Raises `ValidationError` |
| `DailyReport(segments_count=-1, ...)` | Raises `ValidationError` |

Suggested test IDs: `TC-RPT-VAL-001` through `TC-RPT-VAL-003`.

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
  (happy paths only)
- `generate_label_summary` — `test_labels_store.py::TestGenerateLabelSummary` (happy paths only)
- `generate_dummy_labels` — `test_labels_store.py::TestGenerateDummyLabels` (4 tests)
- `project_blocks_to_windows` — `test_labels_projection.py` (11 tests across 7 classes)
- `ActiveLabelingQueue` — `test_labels_queue.py` (13 tests) + `test_label_now.py::TestOnlineQueueEnqueue`

---

## High Priority — Public functions with zero test coverage

### 28. `labels.store.update_label_span()` — no tests
**File:** `src/taskclf/labels/store.py:187-226`
**Doc:** `docs/api/labels/store.md` (listed in table)

```python
def update_label_span(
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    new_label: str,
    path: Path,
) -> LabelSpan:
```

Reads all spans from parquet, finds the one matching `start_ts`/`end_ts`,
replaces its label with `new_label` (validated via `LabelSpan` constructor
against `LABEL_SET_V1`), writes back, and returns the updated span.
Raises `ValueError` if file doesn't exist or no matching span is found.

| Test case | Setup | Expected |
|---|---|---|
| Happy path: update label | Write 2 spans, update the first's label from "Build" to "Debug" | Returns updated `LabelSpan` with `label="Debug"`; re-read file confirms change persisted; other span unchanged |
| File does not exist | Call with non-existent path | `ValueError("No labels file found")` |
| No matching span | Write spans, call with timestamps that don't match any | `ValueError("No label found for ...")` |
| Invalid new_label | Write a span, update with `new_label="InvalidLabel"` | `ValidationError` from `LabelSpan` constructor (label not in `LABEL_SET_V1`) |
| Preserves other fields | Write span with `user_id`, `confidence`, `extend_forward` | After update, all fields except `label` are unchanged |

Suggested test IDs: `TC-LABEL-UPD-001` through `TC-LABEL-UPD-005`.

---

### 29. `labels.store.delete_label_span()` — no tests
**File:** `src/taskclf/labels/store.py:229-252`
**Doc:** `docs/api/labels/store.md` (listed in table)

```python
def delete_label_span(
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    path: Path,
) -> None:
```

Reads all spans, removes the one matching both `start_ts` and `end_ts`,
writes the remaining spans back. Raises `ValueError` if file doesn't
exist or no matching span is found.

| Test case | Setup | Expected |
|---|---|---|
| Happy path: delete one of two spans | Write 2 non-overlapping spans, delete the first | File contains only the second span |
| Delete the only span | Write 1 span, delete it | File contains 0 spans (empty parquet) |
| File does not exist | Call with non-existent path | `ValueError("No labels file found")` |
| No matching span | Write spans, call with timestamps that don't match any | `ValueError("No label found for ...")` |
| Multiple spans with same start but different end | Write 2 spans with same `start_ts` but different `end_ts` (different users), delete one | Only the targeted span removed |

Suggested test IDs: `TC-LABEL-DEL-001` through `TC-LABEL-DEL-005`.

---

## Medium Priority — Error/edge paths missing in tested functions

### 30. `labels.store.import_labels_from_csv()` — missing error path
**File:** `src/taskclf/labels/store.py:76-121`
**Tests:** `test_labels_store.py::TestImportLabelsFromCsvWithOptionalColumns`
(2 happy-path tests)

The function raises `ValueError` when required columns
(`start_ts`, `end_ts`, `label`, `provenance`) are missing. This path
is untested.

| Test case | Setup | Expected |
|---|---|---|
| Missing `label` column | CSV with only `start_ts,end_ts,provenance` | `ValueError("CSV missing required columns: ['label']")` |
| Missing multiple columns | CSV with only `start_ts,end_ts` | `ValueError` listing both missing columns |
| Invalid label value in row | CSV with `label="NotALabel"` | `ValidationError` from `LabelSpan` constructor |

Suggested test IDs: `TC-LABEL-CSV-001` through `TC-LABEL-CSV-003`.

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
