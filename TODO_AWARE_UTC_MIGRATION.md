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

### 1) There is no single internal UTC convention today

Conflicting facts in the current code:

- `src/taskclf/core/time.py`
  - `to_naive_utc()` is documented as the canonical conversion for timestamps
    entering the feature pipeline or Parquet storage.
  - `align_to_bucket()` returns timezone-aware UTC.
- `src/taskclf/core/types.py`
  - `FeatureRow.bucket_start_ts` and `FeatureRow.bucket_end_ts` are normalized
    to aware UTC.
  - `LabelSpan.start_ts` and `LabelSpan.end_ts` are currently normalized to
    naive UTC as a compatibility hotfix.
- `src/taskclf/ui/server.py`
  - label request timestamps are parsed through `_to_naive_utc`
  - REST responses are emitted with explicit UTC suffixes via `_utc_iso()`
- `docs/guide/time_spec.md`
  - says serialized timestamps must be UTC ISO-8601 and shows explicit-UTC
    examples like `...Z`
  - does not pin down whether parsed Python datetimes should be naive UTC or
    aware UTC across all layers

Implication:

- the repo already has contradictory UTC conventions, not just one bad callsite

### 2) Labels currently sit on the naive-UTC side

Relevant paths:

- `src/taskclf/ui/server.py`
- `src/taskclf/labels/store.py`
- `src/taskclf/ui/tray.py`
- `src/taskclf/cli/main.py`

Current behavior:

- label CRUD request bodies are normalized to naive UTC before `LabelSpan`
- CSV imports use `pandas.to_datetime()` and then build `LabelSpan`
- label parquet reads hydrate `LabelSpan` from stored rows
- some CLI and tray flows explicitly convert to naive UTC for pipeline
  compatibility

### 3) Other parts of the repo already lean aware UTC

Relevant paths:

- `src/taskclf/core/types.py` (`FeatureRow`)
- `src/taskclf/core/time.py` (`align_to_bucket()`)
- `src/taskclf/adapters/activitywatch/client.py`
- `src/taskclf/features/build.py`
- `src/taskclf/features/windows.py`

Current behavior:

- several adapters/features helpers already produce or preserve aware UTC
- this makes the current label-side naive contract an outlier

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

## Decisions To Make Before The First PR

### 1) What is the canonical in-memory representation?

Recommended:

- aware UTC everywhere in Python objects

Reason:

- Python comparisons become safe
- it matches `FeatureRow` and `align_to_bucket()`
- it preserves UTC explicitly instead of by convention

### 2) What is the canonical on-disk representation?

Recommended:

- aware UTC for new writes
- transitional dual-read for legacy naive files

Needs confirmation:

- how Pandas/PyArrow round-trip timezone-aware columns in the project's current
  write/read helpers

### 3) Should naive external inputs still be accepted?

Recommended:

- yes, during transition only
- interpret them as UTC
- optionally log/deprecate later

Reason:

- existing CLI and API callers may still send naive timestamps

### 4) Does this require a feature schema version/hash change?

Open question:

- if timestamp semantics on disk change for feature artifacts, decide whether
  that is:
  - a docs-only clarification
  - or a material contract change requiring a version/hash update

Do not decide casually. This touches the repo's schema-stability rules.

### 5) Do we need an explicit migration tool?

Recommended:

- yes, at least for labels and queue state
- possibly for feature parquet if inspection shows mixed semantics on disk

---

## Migration Surface

### Core time helpers

- `src/taskclf/core/time.py`

Audit and likely change:

- `to_naive_utc()` usage sites
- helper naming and deprecation path
- add an `ensure_utc_aware()` helper or equivalent
- make helper behavior the single source of truth

### Core models

- `src/taskclf/core/types.py`

Audit and likely change:

- make `LabelSpan` normalize to aware UTC
- keep `FeatureRow` on aware UTC
- audit any other datetime-carrying models for consistency

### Label storage and import/export

- `src/taskclf/labels/store.py`
- `src/taskclf/labels/queue.py`

Audit and likely change:

- parquet read/write semantics
- CSV import/export semantics
- queue JSON serialization/deserialization
- update/read/delete matching behavior with aware UTC

### REST / WebSocket UI surface

- `src/taskclf/ui/server.py`

Audit and likely change:

- replace `_to_naive_utc` request parsing with aware-UTC normalization
- use aware UTC for visible-range filters
- use aware UTC for date filters and stats filters
- verify all response serialization remains explicit and stable

### Tray/UI integrations

- `src/taskclf/ui/tray.py`
- `src/taskclf/ui/window.py`
- `src/taskclf/ui/window_run.py`

Audit and likely change:

- auto-saved labels
- suggestion windows
- any event timestamps published to the frontend

### CLI

- `src/taskclf/cli/main.py`

Audit and likely change:

- label creation/edit/delete flows
- date-range filtering helpers
- commands that convert timestamps to naive UTC for compatibility

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

### Phase 0 - Spec First

- Decide and document the canonical rule:
  - internal = aware UTC
  - legacy naive inputs are interpreted as UTC
- Update `docs/guide/time_spec.md` first so implementation has a target
- Add helper tests that define:
  - naive input -> aware UTC
  - non-UTC aware input -> converted aware UTC
  - UTC aware input -> preserved as same instant

### Phase 1 - Core Helpers And Models

- Introduce an aware-UTC normalization helper in `src/taskclf/core/time.py`
- Mark `to_naive_utc()` as transitional/deprecated or remove it from active
  paths
- Make `LabelSpan` aware UTC
- Audit `LabelRequest` and any other datetime models for explicit normalization

Exit criteria:

- all core datetime-carrying models have one documented normalization rule

### Phase 2 - Label Boundary Migration

- Update `src/taskclf/labels/store.py` to read legacy naive rows and normalize
  to aware UTC
- Update label CSV import/export behavior and tests
- Update `src/taskclf/labels/queue.py` JSON round-trip behavior
- Verify old `labels.parquet` and `queue.json` still load

Exit criteria:

- legacy label artifacts load
- new label artifacts write aware UTC

### Phase 3 - UI / API / CLI Migration

- Replace `_to_naive_utc` parsing in `src/taskclf/ui/server.py`
- Make label filters, stats filters, and date-window helpers aware UTC
- Update tray paths that create `LabelSpan`
- Remove CLI calls that coerce to naive UTC for compatibility

Exit criteria:

- all label CRUD and stats paths work with aware UTC inputs and legacy naive
  inputs
- no naive datetime constructors remain in these flows unless immediately
  normalized

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

## Suggested PR Breakdown

### PR 1 - Spec + helper groundwork

- docs/spec updates
- new aware-UTC helper
- core time tests
- no behavior changes outside helper boundaries unless required

### PR 2 - labels + UI + CLI

- `LabelSpan`
- label store
- queue
- API handlers
- tray
- CLI label flows
- regression tests for legacy files and aware inputs

### PR 3 - broader audit + migration tooling

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
