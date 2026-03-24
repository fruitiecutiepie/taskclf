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

### 4) Persisted artifacts are likely mixed or at least ambiguous

Artifacts to audit:

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

### 4) Feature schema version/hash change -- OPEN

- timestamp semantics on disk did not change for feature artifacts (FeatureRow
  was already aware UTC)
- no schema version/hash bump needed so far

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

### Features / train / infer / report

Must audit even if not all need changes:

- `src/taskclf/features/`
- `src/taskclf/train/`
- `src/taskclf/infer/`
- `src/taskclf/report/`

Specific risk areas:

- label projection comparisons
- date partitioning via `.date()`
- joins between features and labels
- online inference windows
- report day boundaries

### Docs and tests

Docs:

- `docs/guide/time_spec.md`
- `docs/api/core/types.md`
- `docs/api/ui/labeling.md`
- `docs/guide/usage.md`
- `README.md` if timestamp examples or guarantees need updating

Tests:

- `tests/test_core_time.py`
- `tests/test_core_types.py`
- `tests/test_labels_store.py`
- `tests/test_ui_server.py`
- `tests/test_cli_main.py`
- `tests/test_label_now.py`
- `tests/test_integration_features_labels.py`
- `tests/test_monitor.py`
- any train/infer/report tests that compare datetimes directly

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

### Phase 4 - Feature / Train / Infer / Report Audit

- Audit all time comparisons and partitioning in:
  - features build/join paths
  - dataset construction
  - online and batch inference
  - summaries and reports
- Decide whether feature parquet semantics need a migration or only clearer docs
- Verify that date partitioning still lands rows in the same UTC day buckets

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

### Remaining PRs

### PR 3 - broader audit + migration tooling (Phase 4-6)

- feature/train/infer/report audit fixes
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
