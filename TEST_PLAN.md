# TEST_PLAN.md — taskclf

Test strategy and concrete test cases for `taskclf` (task-type classifier from local activity signals).
This plan prioritizes: correctness, privacy invariants, schema stability, reproducibility, and time-aware ML evaluation.

---

## Implementation Status

| Area | Status | Test file |
|------|--------|-----------|
| Core types / privacy (TC-CORE-001..004) | **Done** | `tests/test_core_types.py` |
| Core hashing (TC-CORE-010..012) | **Done** | `tests/test_core_hashing.py` |
| Schema hash stability (S1, S2) | **Done** | `tests/test_core_schema.py` |
| Schema validate_row / validate_dataframe | **Done** | `tests/test_core_schema.py` |
| Parquet I/O round-trip | **Done** | `tests/test_core_store.py` |
| Label span I/O + dummy generation | **Done** | `tests/test_labels_store.py` |
| Label-to-bucket assignment (TC-LABEL-001, 003, 004) | **Done** | `tests/test_train_dataset.py` |
| Time-based split (TC-EVAL-001) | **Done** | `tests/test_train_dataset.py` |
| Bucketization (TC-TIME-001..004) | **Done** | `tests/test_not_yet_implemented.py` |
| Smoothing / segmentization (TC-INF-001..004) | **Done** | `tests/test_not_yet_implemented.py` |
| Evaluation metrics (TC-EVAL-002..004) | **Done** | `tests/test_core_metrics.py` |
| Model IO bundle (TC-MODEL-001..004) | **Done** | `tests/test_core_model_io.py` |
| Security / privacy (TC-SEC-001..003) | **Done** | `tests/test_security_privacy.py` |
| Train -> infer integration (TC-INT-020..022) | **Done** | `tests/test_integration_train_infer.py` |
| Title opt-in policy (TC-CORE-005) | Blocked | `tests/test_not_yet_implemented.py` |
| Advanced features (TC-FEAT-002..005) | Blocked | `tests/test_not_yet_implemented.py` |
| Adapter ingest integration (TC-INT-001..003) | Not yet tested | -- |
| Ingest -> features -> labels (TC-INT-010..011) | Not yet tested | -- |
| Report generation (TC-INT-030..031) | Not yet tested | -- |
| E2E CLI tests (TC-E2E-*) | Not yet tested | -- |
| Performance / reliability (TC-PERF/REL-*) | Not yet tested | -- |

**Totals**: 114 passed, 5 skipped (blocked on stub modules / unimplemented features).

**Bug fixed during testing**: `train/dataset.py::assign_labels_to_buckets` crashed on empty `label_spans` (KeyError on empty DataFrame). Added early-return guard.

---

## 1. Scope & Goals

### Goals
- Ensure **privacy invariants** are enforced (no raw keystrokes / raw titles persisted by default).
- Ensure **feature schema stability** and **schema-hash gating** prevent silent skew.
- Ensure **time-bucketization**, **span labeling**, and **time-aware splits** are correct.
- Ensure pipelines are **reproducible** and produce **deterministic artifacts**.
- Ensure inference output is stable via smoothing and segmentization.
- Ensure CLI is a stable, reliable interface.

### Non-goals (initially)
- Perfect model quality tests (accuracy thresholds vary by user).
- Performance micro-optimizations (covered later with benchmarking).

---

## 2. Test Levels

### Unit Tests (fast, pure)
- Core contracts, validation, hashing, bucketization, schema hashing
- Feature computation from synthetic events
- Label span alignment logic
- Smoothing/segmentization logic

### Integration Tests (medium)
- Ingest (adapter) -> feature build -> dataset join
- Train -> model bundle -> infer (schema gates)
- Report generation from predictions

### End-to-End Tests (slow)
- CLI-driven run on a small fixture dataset producing artifacts
- Regression snapshots for outputs (hash/row counts/expected labels)

### Property-Based Tests (optional, high value)
- Time spans and bucket alignment invariants
- Segmentization invariants (coverage, ordering, merging)

---

## 3. Test Data Strategy

### Fixture Dataset Requirements
Create small deterministic fixtures (checked in under `tests/fixtures/`):
- `aw_export_minimal/` (or JSONL equivalent) representing:
  - foreground app changes
  - idle periods
  - window title variations
- `labels_minimal.csv` with label spans
- `expected_features.parquet` (optional) for golden tests
- `expected_reports.json` (optional) for snapshot tests

### Synthetic Event Generator
Provide helper that generates normalized events:
- controlled app switches
- controlled idle and active intervals
- controlled title changes
- randomized but seeded sequences for property tests

---

## 4. Core Invariants (must always hold)

### Privacy Invariants
P1. Persisted datasets must not contain:
- raw keystrokes
- raw window titles (unless explicitly enabled by config)
P2. Any title persistence must be hashed/tokenized per policy.
P3. Export/report formats must not include sensitive raw payloads.

### Schema Invariants
S1. Every FeatureRow contains `schema_version` and `schema_hash`.
S2. Schema hash is stable for a given ordered schema definition + feature config.
S3. Inference refuses to run if runtime schema hash != model bundle schema hash.

### Time Invariants
T1. Buckets are aligned to bucket size boundaries.
T2. All derived time values are in UTC internally (or explicitly stored tz-aware).
T3. Label spans cover time ranges; bucket labeling handles overlaps deterministically.

### Pipeline Invariants
R1. `data/processed/` must be reproducible from `data/raw/` + config + code.
R2. Model runs are immutable; new training creates a new run directory.
R3. Reports are derived artifacts and should be regenerable.

---

## 5. Unit Test Cases (Detailed)

### 5.1 `core/schema` and validation
- **TC-CORE-001**: FeatureRowV1 validates required fields exist. **[DONE]** `tests/test_core_types.py`
- **TC-CORE-002**: FeatureRowV1 rejects unknown fields (strict mode). **[DONE]** `tests/test_core_types.py`
- **TC-CORE-003**: Reject any persisted row containing `raw_keystrokes` field. **[DONE]** `tests/test_core_types.py`
- **TC-CORE-004**: Reject any persisted row containing `window_title_raw` by default policy. **[DONE]** `tests/test_core_types.py`
- **TC-CORE-005**: Allow raw title only when config `title_policy=raw_opt_in` and ensure it is never written to `data/processed/` (only `data/interim/` if allowed). **[BLOCKED: title_policy config not implemented]** `tests/test_not_yet_implemented.py`

### 5.2 `core/hashing`
- **TC-CORE-010**: Hashing is deterministic given fixed salt and input. **[DONE]** `tests/test_core_hashing.py`
- **TC-CORE-011**: Different salts yield different hashes. **[DONE]** `tests/test_core_hashing.py`
- **TC-CORE-012**: Hash function output length and charset constraints hold. **[DONE]** `tests/test_core_hashing.py`

### 5.3 `core/time` bucketization/sessionization
- **TC-TIME-001**: Bucket alignment (e.g., 12:00:37 -> 12:00:00 for 60s buckets). **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-TIME-002**: Boundary cases exactly on bucket boundary remain stable. **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-TIME-003**: Day rollovers (23:59:30 -> next day) handled correctly. **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-TIME-004**: DST transition safety (if local timestamps appear): conversion to UTC does not create duplicate buckets. **[DONE]** `tests/test_not_yet_implemented.py`

### 5.4 `features` computation
- **TC-FEAT-001**: Minimal events produce expected feature row counts. *(covered implicitly by `generate_dummy_features` usage in label/training tests)*
- **TC-FEAT-002**: App switch counts in last 5 minutes match expected. **[BLOCKED: `features/windows.py` rolling-window aggregations not implemented]** `tests/test_not_yet_implemented.py`
- **TC-FEAT-003**: Idle segments produce `active_seconds=0` and correct idle flags. **[BLOCKED: real idle event handling not implemented]** `tests/test_not_yet_implemented.py`
- **TC-FEAT-004**: Window title featurization uses hash/tokenization only. **[BLOCKED: `features/text.py` not implemented]** `tests/test_not_yet_implemented.py`
- **TC-FEAT-005**: Rolling-window features are consistent at start-of-day (insufficient history). **[BLOCKED: `features/windows.py` not implemented]** `tests/test_not_yet_implemented.py`

### 5.5 `labels` span alignment
- **TC-LABEL-001**: Bucket receives label if bucket_start_ts is inside `[start_ts, end_ts)`. **[DONE]** `tests/test_train_dataset.py` *(policy: first matching span wins; start inclusive, end exclusive)*
- **TC-LABEL-002**: Overlapping spans resolve by precedence rule (e.g., gold > weak, or longest overlap). *(current impl: first-match wins, tested in `test_first_matching_span_wins`)*
- **TC-LABEL-003**: Gaps in spans yield excluded rows (dropped). **[DONE]** `tests/test_train_dataset.py::test_drops_rows_outside_any_span`
- **TC-LABEL-004**: Invalid spans (end < start) rejected. **[DONE]** `tests/test_core_types.py`

### 5.6 `infer/smooth` and segmentization
- **TC-INF-001**: Rolling majority smoothing reduces short spikes. **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-INF-002**: Segmentization merges adjacent identical labels. **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-INF-003**: Segments are strictly ordered, non-overlapping, cover all predicted buckets. **[DONE]** `tests/test_not_yet_implemented.py`
- **TC-INF-004**: Segment durations match bucket counts * bucket_size. **[DONE]** `tests/test_not_yet_implemented.py`

### 5.7 `core/model_io` bundle checks
- **TC-MODEL-001**: Model bundle writes required files (model, metadata, metrics). **[DONE]** `tests/test_core_model_io.py`
- **TC-MODEL-002**: Load fails if metadata missing schema hash. **[DONE]** `tests/test_core_model_io.py`
- **TC-MODEL-003**: Load fails if schema hash mismatch. **[DONE]** `tests/test_core_model_io.py`
- **TC-MODEL-004**: Load fails if label set mismatch (optional strictness). **[DONE]** `tests/test_core_model_io.py`

---

## 6. Integration Test Cases

### 6.1 Adapter ingest -> normalized events
- **TC-INT-001**: Ingest fixture ActivityWatch export produces normalized events with expected fields.
- **TC-INT-002**: Unknown app ids are normalized to `app_id="unknown"` with provenance retained.
- **TC-INT-003**: Window titles are hashed/tokenized during normalization if required.

### 6.2 Ingest -> features -> labels join
- **TC-INT-010**: Pipeline produces `features_v1` parquet partition for date.
- **TC-INT-011**: Joining labels yields correct labeled training rows count.

### 6.3 Train -> infer (schema gate)
- **TC-INT-020**: Train baseline on fixture dataset produces model bundle. **[DONE]** `tests/test_integration_train_infer.py`
- **TC-INT-021**: Inference on same schema succeeds and produces valid predictions. **[DONE]** `tests/test_integration_train_infer.py`
- **TC-INT-022**: Alter schema (add/remove feature) causes inference to refuse. **[DONE]** `tests/test_integration_train_infer.py`

### 6.4 Report generation
- **TC-INT-030**: Daily report totals sum to total active time (within expected tolerance).
- **TC-INT-031**: Report does not leak raw titles or sensitive data.

---

## 7. End-to-End CLI Tests

Use `pytest` to invoke CLI commands on fixtures (temp dirs):
- **TC-E2E-001**: `taskclf ingest aw ...` -> creates `data/raw/...`
- **TC-E2E-002**: `taskclf features build --date ...` -> creates processed parquet
- **TC-E2E-003**: `taskclf labels import ...` -> creates labels parquet
- **TC-E2E-004**: `taskclf train lgbm ...` -> creates new model run dir
- **TC-E2E-005**: `taskclf infer batch ...` -> creates predictions and segments
- **TC-E2E-006**: `taskclf report daily ...` -> creates report outputs

Assertions:
- exit codes
- expected file existence
- no sensitive fields in outputs
- schema hash presence and consistency

---

## 8. Evaluation Tests (guardrails, not “accuracy locks”)

Because accuracy is user-dependent, focus on evaluation integrity:

- **TC-EVAL-001**: Train/val split is time-based (verify no date overlap). **[DONE]** `tests/test_train_dataset.py`
- **TC-EVAL-002**: Confusion matrix shape matches label set. **[DONE]** `tests/test_core_metrics.py`
- **TC-EVAL-003**: Macro-F1 computed without crashing on missing classes in a fold. **[DONE]** `tests/test_core_metrics.py`
- **TC-EVAL-004**: Class imbalance reported. **[DONE]** `tests/test_core_metrics.py`

Optional thresholds (only for fixtures):
- On the synthetic fixture dataset, expect > X macro-F1 (small, stable).

---

## 9. Performance & Reliability Tests

### 9.1 Batch performance smoke tests
- **TC-PERF-001**: Feature build processes 1 day under N seconds on fixture (loose bound).
- **TC-PERF-002**: DuckDB queries for last 7 days return under N seconds (if used).

### 9.2 Online loop reliability
- **TC-REL-001**: Online inference handles missing minute gracefully (no crash).
- **TC-REL-002**: Adapter temporary failure retries with backoff (if implemented).
- **TC-REL-003**: Writes are atomic (write temp -> rename).

---

## 10. Security & Privacy Tests

- **TC-SEC-001**: Scan produced parquet/csv/json for forbidden columns/keys. **[DONE]** `tests/test_security_privacy.py`
- **TC-SEC-002**: Ensure hashes are one-way (no reversible encoding used by default). **[DONE]** `tests/test_security_privacy.py`
- **TC-SEC-003**: Logs do not print sensitive raw payloads (sanitize logging). **[DONE]** `tests/test_security_privacy.py`

---

## 11. Tooling Setup (Recommended)

### Test runner
- `pytest`, `pytest-xdist`

### Lint/type
- `ruff`, `mypy`

### Pre-commit hooks
- ruff, formatting, trailing whitespace, basic secret scan (optional)

---

## 12. Ownership and “Definition of Done”

A feature is done when:
- unit tests exist for invariants it touches
- integration tests cover pipeline wiring (if applicable)
- schema changes are reflected in schema hash and metadata
- outputs are validated to be privacy-safe
- CLI remains stable or migration notes are added

---

## 13. Open Decisions (record before expanding scope)
- Label-to-bucket assignment policy (midpoint vs overlap fraction)
- Handling of unlabeled time (exclude vs “unknown” class)
- Title featurization method (hash-trick tokens vs salted hash only)
- Whether to store browser domain categories (hashed) and how

---

## Appendix A — Minimal Fixture Policies

To keep fixtures deterministic:
- Use fixed UTC timestamps
- Use seeded random generators
- Keep file sizes small (<1MB) and stable across OSes

---

## Appendix B — Forbidden Data Checklist
Never persist:
- raw keystrokes
- clipboard content
- full window titles (default)
- full URLs (default)
- typed text or IM content
