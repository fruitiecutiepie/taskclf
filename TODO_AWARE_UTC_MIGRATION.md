# TODO - Aware UTC Migration

Context dump and ordered plan for migrating `taskclf` from mixed internal
timestamp conventions to a single timezone-aware UTC contract.

This file is intended to be the working brief before code changes. It captures:

- what the current codebase does today
- where naive UTC is used
- where aware UTC is already used
- why the current mix is risky
- which files and artifacts are in scope
- the safest execution order
- what "done" looks like

---

## Goal

Move the repo to a single canonical internal timestamp convention:

- all domain-model datetimes are timezone-aware UTC
- all internal comparisons use timezone-aware UTC
- persisted timestamps are written in a form that round-trips as
  timezone-aware UTC
- REST and WebSocket outputs continue to include an explicit UTC suffix
- legacy naive timestamps are still readable during transition, but are
  treated as UTC and normalized immediately on input

This is an internal contract migration first. It is not a local-time feature.

---

## Why This Exists

Recent bug:

- `GET /api/labels` crashed with:
  - `TypeError: can't compare offset-naive and offset-aware datetimes`

Crash site:

- `src/taskclf/ui/server.py`
- label range filtering in `api_op_labels_get()`

Root cause:

- the repo currently mixes two UTC representations:
  - naive UTC (`tzinfo is None`, but semantically UTC)
  - aware UTC (`tzinfo=timezone.utc`)

That mix is fragile because Python refuses direct comparison between naive and
aware datetimes, even when both represent UTC.

---

## Non-Negotiables

- Do not reinterpret legacy naive timestamps as local time. They mean UTC.
- Do not silently mix naive UTC and aware UTC after the migration starts.
- Do not break privacy guarantees or schema gating while touching time logic.
- Do not leave docs stale; update spec/docs with code changes.
- Do not make artifact semantics ambiguous. Storage and API behavior must be
  explicit.

---

## Current State Summary

### 1) Core models now use aware UTC (Phase 0-1 done)

After Phase 0-1:

- `src/taskclf/core/time.py`
  - `ts_utc_aware_get()` is the canonical normalizer for all domain models.
  - `to_naive_utc()` is deprecated (still functional for transitional callers).
  - `align_to_bucket()` returns timezone-aware UTC (unchanged).
- `src/taskclf/core/types.py`
  - `FeatureRow.bucket_start_ts` and `FeatureRow.bucket_end_ts` are normalized
    to aware UTC (unchanged).
  - `LabelSpan.start_ts` and `LabelSpan.end_ts` are now normalized to
    aware UTC via `ts_utc_aware_get()`.
- `src/taskclf/labels/queue.py`
  - `LabelRequest.bucket_start_ts`, `.bucket_end_ts`, `.created_at` are
    normalized to aware UTC via field validator.
- `docs/guide/time_spec.md`
  - section 1.2 documents the canonical internal representation as aware UTC.

### 2) Label store boundary uses aware UTC (Phase 2 partially done)

- `src/taskclf/labels/store.py`
  - `update_label_span()` and `delete_label_span()` normalize inputs.
  - `read_label_spans()` reads legacy naive rows; `LabelSpan` normalizes them.
- `src/taskclf/labels/projection.py`
  - fixed Pandas `pd.Timestamp(ts, tz=...)` error for already-aware timestamps.

### 3) UI / API / CLI now use aware UTC internally (Phase 3 done)

- `src/taskclf/ui/server.py`
  - all request timestamps are parsed through `_ensure_utc` (alias for
    `ts_utc_aware_get()`); the old `_to_naive_utc` alias has been removed
  - `datetime.combine` date-filter boundaries use `tzinfo=dt.timezone.utc`
  - `train_data_check` label-range bounds use `tzinfo=dt.timezone.utc`
  - REST responses are still emitted with explicit UTC suffixes via `_utc_iso()`
- `src/taskclf/ui/tray.py`
  - `_label_stats()` simplified — redundant `hasattr` branch removed
    (all `LabelSpan` timestamps are guaranteed aware UTC)
- `src/taskclf/cli/main.py`
  - `labels_label_now_cmd` uses `ts_utc_aware_get()` instead of
    `to_naive_utc()`; aware-UTC timestamps are passed through to
    `LabelSpan`, `generate_label_summary`, and `append_label_span`

### 4) Feature / train / infer / report layers audited (Phase 4 partially done)

After Phase 4 audit:

- `src/taskclf/features/windows.py`
  - `_epoch()` now uses `ts_utc_aware_get()` instead of inline
    `replace(tzinfo=...)` workaround
- `src/taskclf/features/build.py`
  - already routes all timestamps through `align_to_bucket()` which
    returns aware UTC; no changes needed
- `src/taskclf/features/sessions.py`
  - relies on callers to pass consistent timestamp types; `build.py`
    always passes aligned (aware UTC) values; no changes needed
- `src/taskclf/infer/batch.py`
  - `read_segments_json()` normalizes parsed timestamps via
    `ts_utc_aware_get()` so legacy naive JSON is read as aware UTC
  - `run_batch_inference()` normalizes `bucket_starts` extracted from
    DataFrame via `ts_utc_aware_get()`
- `src/taskclf/train/retrain.py`
  - `check_retrain_due()` and `check_calibrator_update_due()` use
    `ts_utc_aware_get()` instead of inline `replace(tzinfo=UTC)`
  - `except` clauses fixed to use parenthesized tuples for PEP 8 style
- `src/taskclf/report/daily.py`
  - only uses `.date()` on segment timestamps — safe for aware UTC;
    no changes needed

Remaining artifacts to audit:

- `data/processed/labels_v1/labels.parquet`
- `data/processed/labels_v1/queue.json`
- `data/processed/features_v1/**/*.parquet`
- imported/exported labels CSVs

Important note:

- do not assume current on-disk behavior from type hints alone
- inspect representative real artifacts before changing write semantics

---

## Target State

Canonical internal contract after migration:

- all datetimes in Python domain models are aware UTC
- all comparisons and filters are against aware UTC values
- day boundaries are built as aware UTC datetimes
- serialized JSON/API timestamps keep explicit UTC offsets (`Z` or `+00:00`)
- legacy naive inputs are normalized to aware UTC at the boundary

Recommended boundary rule:

- read: accept both naive and aware, normalize immediately to aware UTC
- write: emit aware UTC only

---

## Decisions Made

### 1) Canonical in-memory representation -- DECIDED

- aware UTC everywhere in Python objects
- enforced via `ts_utc_aware_get()` in model validators

### 2) Canonical on-disk representation -- DECIDED

- aware UTC for new writes
- transitional dual-read for legacy naive files (LabelSpan normalizes on read)

### 3) Naive external inputs -- DECIDED

- accepted during transition
- interpreted as UTC and normalized immediately via `ts_utc_aware_get()`

### 4) Feature schema version/hash change -- DECIDED

- timestamp semantics on disk did not change for feature artifacts (FeatureRow
  was already aware UTC)
- no schema version/hash bump needed
- Phase 4 audit confirmed: `align_to_bucket()` returns aware UTC, `build.py`
  constructs UTC day boundaries explicitly, date partitioning is unchanged

### 5) Migration tooling -- OPEN

- not yet implemented (Phase 5)
- label store reads handle legacy naive artifacts automatically via LabelSpan
  normalization

---

## Migration Surface

### Core time helpers -- DONE

- `src/taskclf/core/time.py`
  - [x] Added `ts_utc_aware_get()` as canonical normalizer
  - [x] Deprecated `to_naive_utc()` in docstring
  - [x] Tests: TC-TIME-014..016

### Core models -- DONE

- `src/taskclf/core/types.py`
  - [x] `LabelSpan` normalizes to aware UTC
  - [x] `FeatureRow` was already on aware UTC
  - [x] `LabelRequest` now normalizes to aware UTC

### Label storage and import/export -- PARTIALLY DONE

- `src/taskclf/labels/store.py`
  - [x] `update_label_span()` / `delete_label_span()` normalize inputs
  - [x] Parquet read/write works (LabelSpan normalizes on read)
  - [x] CSV import/export round-trip verified
- `src/taskclf/labels/queue.py`
  - [x] `LabelRequest` normalizes timestamps via field validator
  - [x] `_bucket_key()` normalizes for dedup consistency
- `src/taskclf/labels/projection.py`
  - [x] Fixed Pandas tz-aware Timestamp construction

### REST / WebSocket UI surface -- DONE

- `src/taskclf/ui/server.py`
  - [x] replaced `_to_naive_utc` with `_ensure_utc` (alias for `ts_utc_aware_get`)
  - [x] aware UTC for visible-range filters
  - [x] aware UTC for date filters (`datetime.combine` with `tzinfo=utc`)
  - [x] aware UTC for stats filters (`.date()` on aware timestamps)
  - [x] aware UTC for `train_data_check` label-range bounds
  - [x] response serialization unchanged (`_utc_iso()` handles both)

### Tray/UI integrations -- DONE

- `src/taskclf/ui/tray.py`
  - [x] `_label_stats()` simplified (redundant `hasattr` branch removed)
- `src/taskclf/ui/window.py` — no datetime usage; no change needed
- `src/taskclf/ui/window_run.py` — no datetime usage; no change needed

### CLI -- DONE

- `src/taskclf/cli/main.py`
  - [x] `labels_label_now_cmd` uses `ts_utc_aware_get()` instead of
    `to_naive_utc()`
  - [x] aware-UTC timestamps passed through to `LabelSpan`,
    `generate_label_summary`, and `append_label_span`

### Features -- DONE

- `src/taskclf/features/windows.py`
  - [x] Replaced inline `_epoch()` workaround with `ts_utc_aware_get()`
  - [x] Tests: TC-FEAT-WIN-UTC-001..005

- `src/taskclf/features/build.py` — no change needed (uses `align_to_bucket`)
- `src/taskclf/features/sessions.py` — no change needed (callers pass aware UTC)
- `src/taskclf/features/dynamics.py` — no datetime usage
- `src/taskclf/features/domain.py` — no datetime usage
- `src/taskclf/features/text.py` — no datetime usage

### Train -- DONE

- `src/taskclf/train/retrain.py`
  - [x] `check_retrain_due()` uses `ts_utc_aware_get()` for timestamp
    normalization
  - [x] `check_calibrator_update_due()` uses `ts_utc_aware_get()` for
    timestamp normalization
  - [x] `except` clauses use parenthesized tuples (PEP 8 style fix)
  - [x] Tests: TC-RETRAIN-UTC-001..007

- `src/taskclf/train/build_dataset.py` — no timestamp normalization needed
  (sorts by `bucket_start_ts` column; types from upstream)
- `src/taskclf/train/dataset.py` — no timestamp normalization needed
  (sorts by `bucket_start_ts`; split by index fractions)
- `src/taskclf/train/calibrate.py` — uses `.dt.date` on DataFrame column;
  correct for aware-UTC series
- `src/taskclf/train/lgbm.py` — no datetime usage
- `src/taskclf/train/evaluate.py` — no datetime usage

### Infer -- PARTIALLY DONE

- `src/taskclf/infer/batch.py`
  - [x] `read_segments_json()` normalizes via `ts_utc_aware_get()`
  - [x] `run_batch_inference()` normalizes `bucket_starts` via
    `ts_utc_aware_get()`
  - [x] Tests: TC-BATCH-UTC-001..004

- `src/taskclf/infer/smooth.py` — only `timedelta` arithmetic; no
  normalization needed
- `src/taskclf/infer/baseline.py` — uses `pd.Timestamp(ts).to_pydatetime()`
  like batch; inherits whatever tz the DataFrame column carries
- `src/taskclf/infer/prediction.py` — `bucket_start_ts: datetime` field only
- `src/taskclf/infer/monitor.py` — `datetime.now(tz=timezone.utc)` already
  aware; `pd.Timestamp` arithmetic preserves tz
- `src/taskclf/infer/resolve.py` — uses `time.monotonic()` only
- `src/taskclf/infer/calibration.py` — `created_at` is ISO string, not parsed
- `src/taskclf/infer/taxonomy.py` — no datetime usage

Remaining:

- [ ] `src/taskclf/infer/online.py` — `ev.timestamp` comparisons depend on
  adapter output; needs review when adapter layer is audited
- [ ] `src/taskclf/infer/baseline.py` — could normalize `bucket_starts` like
  `batch.py` for consistency (not urgent, inherits from upstream)

### Report -- DONE (no changes needed)

- `src/taskclf/report/daily.py` — only uses `.date()` on first segment's
  `start_ts`; safe for both naive and aware timestamps
- `src/taskclf/report/export.py` — no datetime columns; only string `date`

### Docs and tests

Docs:

- `docs/guide/time_spec.md`
- `docs/api/core/types.md`
- `docs/api/ui/labeling.md`
- `docs/guide/usage.md`
- `README.md` if timestamp examples or guarantees need updating

Tests -- Phase 4 additions:

- `tests/test_features_windows.py` -- TC-FEAT-WIN-UTC-001..005
- `tests/test_infer_batch_segments.py` -- TC-BATCH-UTC-001..004
- `tests/test_retrain.py` -- TC-RETRAIN-UTC-001..007

Tests -- previously covered:

- `tests/test_core_time.py`
- `tests/test_core_types.py`
- `tests/test_labels_store.py`
- `tests/test_ui_server.py`
- `tests/test_cli_main.py`
- `tests/test_label_now.py`
- `tests/test_integration_features_labels.py`
- `tests/test_monitor.py`

---

## Known Risk Areas

- Python will error on naive/aware comparison, but Pandas may silently preserve
  timezone metadata through CSV/Parquet round-trips.
- Equality-based update/delete operations may fail if one side is naive UTC and
  the other is aware UTC.
- Day filters built with naive `datetime.combine()` will break once labels move
  to aware UTC.
- `.date()` on aware UTC is safe only if the value has already been normalized
  to UTC first.
- JSON serialization may produce different wire strings (`+00:00` vs `Z`);
  choose one and keep tests/docs aligned.
- Artifact rewriting can affect reproducibility if mixed formats are written in
  the same dataset tree.

---

## Proposed Execution Plan

### Phase 0 - Spec First -- DONE

- [x] Decided and documented the canonical rule in `docs/guide/time_spec.md`
  (section 1.2 "Internal Representation"):
  - internal = aware UTC
  - legacy naive inputs are interpreted as UTC
- [x] Added `ts_utc_aware_get()` helper to `src/taskclf/core/time.py`
- [x] Added helper tests (TC-TIME-014..016) that define:
  - naive input -> aware UTC
  - non-UTC aware input -> converted aware UTC
  - UTC aware input -> preserved as same instant

### Phase 1 - Core Helpers And Models -- DONE

- [x] Introduced `ts_utc_aware_get()` in `src/taskclf/core/time.py`
- [x] Marked `to_naive_utc()` as deprecated (docstring note)
- [x] Made `LabelSpan` normalize to aware UTC
- [x] Added aware-UTC validator to `LabelRequest`
- [x] `FeatureRow` was already on aware UTC (no change needed)

Exit criteria met:

- all core datetime-carrying models have one documented normalization rule

### Phase 2 - Label Boundary Migration -- PARTIALLY DONE

- [x] `update_label_span()` and `delete_label_span()` normalize inputs via
  `ts_utc_aware_get()` so equality comparisons work with either naive or
  aware caller timestamps
- [x] `read_label_spans()` reads legacy naive rows; `LabelSpan` normalizes
  them to aware UTC automatically
- [x] Label CSV import/export round-trips verified (existing tests pass)
- [x] `ActiveLabelingQueue._bucket_key()` normalizes timestamps so
  deduplication works across naive/aware boundaries
- [x] Fixed `labels/projection.py` Pandas `pd.Timestamp(ts, tz=...)` error
  for already-aware timestamps

Remaining:

- [ ] Verify real on-disk `labels.parquet` and `queue.json` load correctly
  (manual smoke test needed)
- [ ] Update `docs/api/core/types.md` and `docs/api/ui/labeling.md` to
  reflect aware-UTC contract

### Phase 3 - UI / API / CLI Migration -- DONE

- [x] Replaced `_to_naive_utc` with `_ensure_utc` (alias for `ts_utc_aware_get`)
  across all 16 call sites in `src/taskclf/ui/server.py`
- [x] Fixed `datetime.combine` date filter to produce aware UTC boundaries
- [x] Fixed `train_data_check` naive datetime bounds to use `tzinfo=utc`
- [x] Simplified `_label_stats()` in `src/taskclf/ui/tray.py`
- [x] Replaced `to_naive_utc` with `ts_utc_aware_get()` in
  `src/taskclf/cli/main.py` `labels_label_now_cmd`
- [x] Unskipped `test_range_filter_handles_aware_spans_from_storage`
- [x] Updated `TestUtcHelpers` to test `_ensure_utc` instead of `_to_naive_utc`
- [x] Added tests: date filter with aware spans, non-UTC span normalization,
  stats with aware-UTC timestamps, stats with non-UTC offset inputs

Exit criteria met:

- all label CRUD and stats paths work with aware UTC inputs and legacy naive
  inputs
- no naive datetime constructors remain in these flows unless immediately
  normalized
- 112 test_ui_server tests pass (was 86 with 1 skipped; now 112 with 0 skipped)

### Phase 4 - Feature / Train / Infer / Report Audit -- PARTIALLY DONE

- [x] Audited all time comparisons and partitioning in:
  - features build/join paths
  - dataset construction
  - batch inference
  - summaries and reports
- [x] `features/windows.py`: replaced inline `_epoch()` workaround with
  `ts_utc_aware_get()`
- [x] `infer/batch.py`: `read_segments_json()` and `run_batch_inference()`
  normalize timestamps via `ts_utc_aware_get()`
- [x] `train/retrain.py`: cadence checks use `ts_utc_aware_get()`;
  `except` clauses fixed to PEP 8 parenthesized tuples
- [x] `report/`: audited — no changes needed (`.date()` only)
- [x] Feature parquet semantics: no migration needed; `FeatureRow` was
  already aware UTC, `align_to_bucket` returns aware UTC
- [x] Date partitioning: `build.py` constructs UTC day boundaries with
  `tzinfo=dt.timezone.utc`; buckets land in the same UTC day
- [x] Tests: TC-FEAT-WIN-UTC-001..005, TC-BATCH-UTC-001..004,
  TC-RETRAIN-UTC-001..007
- [x] Updated existing tests using naive `Segment` timestamps in
  `test_infer_batch_segments.py` to use aware UTC

Remaining:

- [ ] `infer/online.py` event timestamp comparisons (depends on adapter
  output contract)
- [ ] `infer/baseline.py` could normalize `bucket_starts` for consistency

Exit criteria:

- end-to-end train/infer/report flows operate on a single aware-UTC contract

### Phase 5 - Data Migration Tooling

- Add a one-shot audit command or script to detect naive datetime artifacts
- Add an optional rewrite command for:
  - `labels.parquet`
  - `queue.json`
  - any confirmed mixed legacy artifacts
- Make rewrite behavior explicit and idempotent

Exit criteria:

- operators can inspect and normalize old local data safely

### Phase 6 - Cleanup

- remove transitional compatibility shims where appropriate
- decide whether naive API/CLI inputs remain supported
- remove obsolete docs and comments about naive-UTC storage

Exit criteria:

- there is only one documented internal UTC convention left in the repo

---

## PR History

### PR 1 - Core + labels boundary (Phase 0-1 + partial Phase 2) -- DONE

Files changed:

- `src/taskclf/core/time.py` -- added `ts_utc_aware_get()`, deprecated
  `to_naive_utc()`
- `src/taskclf/core/types.py` -- `LabelSpan` normalizes to aware UTC
- `src/taskclf/labels/queue.py` -- `LabelRequest` normalizes timestamps
- `src/taskclf/labels/store.py` -- `update/delete_label_span` normalize inputs
- `src/taskclf/labels/projection.py` -- fixed Pandas tz-aware construction
- `docs/guide/time_spec.md` -- added section 1.2 "Internal Representation"
- `tests/test_core_time.py` -- TC-TIME-014..016
- `tests/test_labels_store.py` -- updated assertions for aware UTC
- `tests/test_labels_weak_rules.py` -- updated `_ts` helper for aware UTC
- `tests/test_ui_server.py` -- skipped Phase 3 test with TODO

### PR 2 - UI / API / CLI migration (Phase 3) -- DONE

Files changed:

- `src/taskclf/ui/server.py` -- replaced `_to_naive_utc` with `_ensure_utc`
  (alias for `ts_utc_aware_get`); fixed `datetime.combine` date filter and
  `train_data_check` naive bounds
- `src/taskclf/ui/tray.py` -- simplified `_label_stats()` filter
- `src/taskclf/cli/main.py` -- replaced `to_naive_utc` with `ts_utc_aware_get`
  in `labels_label_now_cmd`
- `tests/test_ui_server.py` -- unskipped Phase 3 test; updated
  `TestUtcHelpers` for `_ensure_utc`; added date filter, non-UTC
  normalization, and stats tests

### PR 3 - feature/train/infer audit (Phase 4) -- DONE

Files changed:

- `src/taskclf/features/windows.py` -- replaced inline `_epoch()` naive-UTC
  workaround with `ts_utc_aware_get()`
- `src/taskclf/infer/batch.py` -- `read_segments_json()` and
  `run_batch_inference()` normalize timestamps via `ts_utc_aware_get()`
- `src/taskclf/train/retrain.py` -- `check_retrain_due()` and
  `check_calibrator_update_due()` use `ts_utc_aware_get()` instead of inline
  `replace(tzinfo=UTC)`; `except` clauses fixed to PEP 8 parenthesized tuples
- `tests/test_features_windows.py` -- added TC-FEAT-WIN-UTC-001..005
  (aware-UTC, mixed naive/aware, non-UTC offset events)
- `tests/test_infer_batch_segments.py` -- updated existing segment round-trip
  tests to use aware UTC; added TC-BATCH-UTC-001..004 (legacy naive JSON,
  non-UTC offset JSON, batch inference aware segments)
- `tests/test_retrain.py` -- added TC-RETRAIN-UTC-001..007 (cadence checks
  with naive/aware/non-UTC-offset metadata)

### Remaining PRs

### PR 4 - migration tooling + cleanup (Phase 5-6)

- online inference event timestamp audit (depends on adapter contract)
- artifact audit/rewrite tooling
- final doc cleanup

---

## Acceptance Criteria

- no production path compares naive and aware datetimes
- all domain-model timestamps are normalized to aware UTC
- REST and WebSocket outputs still include explicit UTC offsets
- legacy naive label artifacts still load as UTC
- date/range filters behave correctly across API, CLI, and reporting paths
- tests cover naive input, aware input, CSV import, Parquet read, queue JSON
  read, and range/date filters
- docs describe one internal contract, not two

---

## Open Questions

- Does feature parquet already round-trip as aware UTC everywhere in practice?
- Should JSON APIs continue emitting `+00:00` or switch to `Z`?
- Should naive API/CLI inputs remain accepted permanently or be deprecated?
- Does changing on-disk timestamp awareness require a schema version/hash bump?
- Should migration tooling rewrite in place or write side-by-side outputs?

---

## Useful Starting Searches

- `to_naive_utc(`
- `replace(tzinfo=None)`
- `datetime.combine(`
- `tzinfo=timezone.utc`
- `tzinfo=dt.timezone.utc`
- `pd.to_datetime(`

---

## Bottom Line

This should be treated as a contract migration, not a one-line bug fix.

The real goal is not merely "stop the crash"; it is:

- one UTC convention
- one normalization rule
- one documented storage/API story
- zero naive/aware ambiguity across labels, features, training, inference, and
  reporting
