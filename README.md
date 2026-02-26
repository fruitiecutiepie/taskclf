# taskclf — Task Type Classifier from Local Activity Signals

Train and run a personal task-type classifier (e.g. coding / writing / meetings) using privacy-preserving computer activity signals such as foreground app/window metadata and aggregated input statistics (counts/rates only).

This project is intentionally scoped as a **personalized classifier** (single-user first). The architecture keeps:
- **Collectors** (platform/tool dependent) isolated behind adapters
- **Features** as a versioned, validated contract
- **Models** as bundled artifacts with schema checks
- **Inference** as a small, stable loop that emits task segments and daily summaries

## Goals
- Fast iteration: first useful model in < 1 week of data
- Privacy: no raw keystrokes, no raw window titles persisted
- Stability: feature schema versioning + schema hash gates
- Extensibility: add new collectors and models without breaking consumers

## Non-Goals
- Universal (multi-user) generalization out of the box
- Storing or analyzing raw typed content
- "Perfect" labeling UI (start minimal, iterate later)

---

## Labels (v1)

Eight core labels defined in `schema/labels_v1.json`:

| ID | Label | Description |
|----|-------|-------------|
| 0 | `Build` | Writing or implementing structured content in editor/terminal |
| 1 | `Debug` | Investigating issues, terminal-heavy troubleshooting |
| 2 | `Review` | Reviewing technical material or diffs with light edits |
| 3 | `Write` | Writing structured non-code content |
| 4 | `ReadResearch` | Consuming information with minimal production |
| 5 | `Communicate` | Asynchronous coordination (chat/email) |
| 6 | `Meet` | Synchronous meetings or calls |
| 7 | `BreakIdle` | Idle or break period |

Labels are stored as **time spans** (not per-keystroke events). Users can remap
core labels to personal categories via a **taxonomy config**
(see `configs/user_taxonomy_example.yaml`).

---

## Data Flow Overview

### Structures (pipelines)
* ETL pipeline reads raw → produces features parquet
* Training pipeline reads features + labels → produces model
* Inference pipeline reads new events → emits predictions + segments

### Batch (repeatable)
1. **Ingest**: pull ActivityWatch export → `data/raw/aw/`
2. **Feature build**: events → per-minute features → `data/processed/features_v1/`
3. **Label import**: label spans → `data/processed/labels_v1/`
4. **Build dataset**: join features + labels, split by time → training arrays
5. **Train**: fit model → `models/<run_id>/`
6. **Evaluate**: metrics, acceptance checks, calibration
7. **Report**: daily summaries → `artifacts/`

### Online (real-time)
Every N seconds:
- read the last minute(s) of events
- compute the latest feature bucket
- predict + smooth (with optional calibration and taxonomy mapping)
- append predictions → `artifacts/`

At end-of-day:
- produce report

---

## Privacy & Safety
This repo enforces the following:
- **No raw keystrokes** are stored (only aggregate counts/rates).
- **No raw window titles** are stored by default.
  - Titles are hashed or locally tokenized; you can keep a local mapping if you choose.
- Dataset artifacts stay **local-first**.

---

## Quick Start

### Requirements
- Python >= 3.14
- `uv` installed

### Setup
```bash
uv sync
uv run taskclf --help
```

### Ingest (ActivityWatch)

```bash
uv run taskclf ingest aw --input /path/to/activitywatch-export.json
```

This parses an ActivityWatch JSON export, normalizes app names to reverse-domain
identifiers, hashes window titles (never storing raw text), and writes
privacy-safe events to `data/raw/aw/<YYYY-MM-DD>/events.parquet` partitioned by
date.

Options:
- `--out-dir` — output directory (default: `data/raw/aw`)
- `--title-salt` — salt for hashing window titles (default: `taskclf-default-salt`)

### Build features

```bash
uv run taskclf features build --date 2026-02-16
```

### Import labels

```bash
uv run taskclf labels import --file labels.csv
```

Or add individual label blocks:

```bash
uv run taskclf labels add-block \
  --start 2026-02-16T09:00:00 --end 2026-02-16T10:00:00 --label Build
```

### Train

```bash
uv run taskclf train lgbm --from 2026-02-01 --to 2026-02-16
```

### Run batch inference

```bash
uv run taskclf infer batch --model-dir models/<run_id> --from 2026-02-01 --to 2026-02-16
```

### Run online inference

```bash
uv run taskclf infer online --model-dir models/<run_id>
```

Starts a polling loop that queries a running ActivityWatch server, builds
feature rows from live window events, predicts task types using a trained model,
smooths predictions, and writes running outputs to `artifacts/`. Press Ctrl+C
to stop; a final daily report is generated on shutdown.

Options:
- `--poll-seconds` — seconds between polls (default: 60)
- `--aw-host` — ActivityWatch server URL (default: `http://localhost:5600`)
- `--smooth-window` — rolling majority window size (default: 3)
- `--title-salt` — salt for hashing window titles (default: `taskclf-default-salt`)
- `--out-dir` — output directory (default: `artifacts`)

### Run baseline (no model needed)

```bash
uv run taskclf infer baseline --from 2026-02-01 --to 2026-02-16
```

Rule-based classifier useful for day-1 bootstrapping before you have a trained model.

### Produce report

```bash
uv run taskclf report daily --segments-file artifacts/segments.json
```

---

## CLI Reference

All commands: `uv run taskclf --help`

| Group | Commands | Purpose |
|-------|----------|---------|
| `ingest` | `aw` | Import ActivityWatch exports |
| `features` | `build` | Build per-minute feature rows |
| `labels` | `import`, `add-block`, `show-queue`, `project` | Manage label spans and labeling queue |
| `train` | `build-dataset`, `lgbm`, `evaluate`, `tune-reject`, `calibrate`, `retrain`, `check-retrain` | Training, evaluation, and retraining pipeline |
| `taxonomy` | `validate`, `show`, `init` | User-defined label groupings |
| `infer` | `batch`, `online`, `baseline`, `compare` | Prediction (ML, rule-based, comparison) |
| `report` | `daily` | Daily summaries (JSON/CSV/Parquet) |
| `monitor` | `drift-check`, `telemetry`, `show` | Feature drift and telemetry tracking |

Full CLI docs: `docs/api/cli/main.md`

---

## Repo Layout

* `src/taskclf/` — application code (adapters, core, features, labels, train, infer, report, ui)
* `schema/` — versioned JSON schemas for features and labels
* `configs/` — configuration files (model params, retrain policy, taxonomy examples)
* `docs/` — API reference and guides (served via `make docs-serve`)
* `data/` — raw and processed datasets (local, gitignored)
* `models/` — trained model bundles (one folder per run)
* `artifacts/` — predictions, segments, reports, evaluation outputs
* `tests/` — test suite

---

## Model Artifact Contract

Every saved model bundle (`models/<run_id>/`) contains:

* the model file
* `metadata.json`: feature schema version + hash, label set, training date range, params, dataset hash
* `metrics.json`: macro/weighted F1, per-class metrics
* `confusion_matrix.csv`
* categorical encoders (if applicable)

Inference refuses to run if the schema hash mismatches the model bundle.

---

## Development

Common tasks are in the `Makefile`:

```bash
make lint        # ruff check .
make test        # pytest
make typecheck   # mypy src
make docs-serve  # local preview at http://127.0.0.1:8000
make docs-build  # static site in site/
```

---

## License

TBD (local-first personal project by default).
