# AGENTS.md — taskclf

Operational rules for humans and code-generators working in this repo.
This file is authoritative for project boundaries, invariants, and interfaces.

---

## Mission
Build a **local-first**, privacy-preserving classifier that infers a user's task type
(e.g., coding/writing/meetings) from computer activity signals (apps/windows + aggregated input stats).

Primary outcomes:
- stable feature pipeline
- reproducible training
- safe on-device inference
- clear daily reports

---

## Non-Negotiables
1. **Privacy**
   - Never store raw keystrokes.
   - Never store full window titles by default.
   - Persist only aggregate rates/counts and hashed/tokenized identifiers.
   - Any opt-in raw storage must be explicit, local-only, and gated by config.

2. **Schema Stability**
   - Features are a versioned contract (`FeatureSchemaV1`, etc.).
   - Every feature row must include `schema_version` and `schema_hash`.
   - Inference MUST refuse to run if schema hash mismatches the model bundle.

3. **Separation of Concerns**
   - Adapters = platform/tool specific integrations.
   - Core = stable validated primitives and IO.
   - Pipelines = composition; pure transforms where possible.
   - CLI = thin, stable interface; no business logic.

4. **Time-Aware Evaluation**
   - Default train/val split is by **day/week**, never random rows.
   - Report macro-F1 and confusion matrix.

5. **Deterministic Artifacts**
   - `data/processed/` must be reproducible from `data/raw/` + code + configs.
   - `models/` run directories are immutable and never overwritten.

---

## Repo Map (what goes where)
- `src/taskclf/core/` — schemas, invariants, hashing, bucketization, storage, model IO, metrics
- `src/taskclf/adapters/` — ActivityWatch, OS input collectors (counts only), normalization
- `src/taskclf/features/` — event -> bucketed feature rows, rolling windows, title featurization
- `src/taskclf/labels/` — label spans format, validation, import/export, weak-label rules (optional)
- `src/taskclf/train/` — dataset join/splits, training, calibration, model bundling
- `src/taskclf/infer/` — prediction, smoothing, segment creation
- `src/taskclf/report/` — summaries and exports
- `src/taskclf/cli/` — Typer commands

Data:
- `data/raw/` — append-only source snapshots
- `data/processed/` — versioned feature/label datasets
- `models/` — model bundles (one folder per run)
- `artifacts/` — predictions, segments, reports, eval outputs

---

## Stable Interfaces (do not break casually)

### CLI (public surface)
Commands should remain stable; additions are fine, breaking changes require migration notes.

Suggested command groups:
- `taskclf ingest ...`
- `taskclf features ...`
- `taskclf labels ...`
- `taskclf train ...`
- `taskclf infer ...`
- `taskclf report ...`

### FeatureRow Contract
All persisted feature rows must include:
- `bucket_start_ts` (UTC, minute-aligned)
- `schema_version`
- `schema_hash`
- `source_ids` (which collectors contributed)

### LabelSpan Contract
Gold labels are spans:
- `start_ts`, `end_ts`, `label`, `provenance`
Weak labels must be marked as such.

### Model Bundle Contract
Each model run dir contains:
- model file
- `metadata.json` with: schema version/hash, label set, training range, params, git commit
- `metrics.json` and confusion matrix

---

## Coding Standards
- Python >= 3.14
- Prefer pure functions in feature computation and dataset joins.
- Use `pydantic` (or dataclasses + explicit validators) for contracts.
- Avoid implicit globals; pass config explicitly or via typed config object.
- Fail fast: validation errors should surface early with actionable messages.

Lint/test:
- `ruff` for linting
- `pytest` for tests
- `mypy` for typing (incrementally adopted)

---

## Change Policy
### Adding a new feature
1. Add to schema definition (`core/schema.py`) with dtype and description.
2. Update feature builder (`features/`) and unit tests.
3. Ensure schema hash changes (automatic).
4. Record in `configs/features_vX.yaml` if feature selection is configurable.

### Changing labels
- Avoid changing `labels_v1` once training begins.
- If you must: create `labels_v2` and keep compatibility tooling.

### Adding a new adapter
- Must output normalized events compatible with core types.
- Must not leak sensitive payloads into persisted datasets.

### Documentation
- Every code change must include corresponding updates to `docs/`.
- New modules, classes, and public functions require API reference pages under `docs/api/`.
- Changed signatures, behaviors, or defaults must be reflected in existing doc pages.
- CLI changes must update `docs/api/cli/` and any relevant guides.
- Do not merge code that leaves docs stale; treat doc drift as a defect.

---

## Execution Modes
### Batch
- Ingest -> feature build -> label import -> train -> report

### Online
- Poll -> build latest bucket -> predict -> smooth -> append predictions -> segmentize -> report

Online mode must never retrain.

---

## Testing Requirements (minimum)
- `core/schema` validation tests (reject raw titles, raw keystrokes fields)
- bucketization correctness (timezones/UTC alignment)
- dataset join correctness (span -> bucket label assignment)
- train/infer schema hash gate (must reject mismatches)

---

## Defaults
- Bucket size: 60s
- Split: by day
- Baseline model: LightGBM multiclass
- Smoothing: rolling majority window (configurable)

---

## Notes for Code Generators
When asked to implement something:
1. Start by identifying which layer it belongs to (core/adapters/features/train/infer/report/cli).
2. Maintain the privacy rules and schema gating.
3. Prefer adding small, composable modules and tests over large scripts.
4. Never store sensitive raw inputs unless config explicitly enables it.

If a request conflicts with this file, this file wins.
