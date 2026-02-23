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
- “Perfect” labeling UI (start minimal, iterate later)

---

## Labels (v1)
Start with a small, consistent label set and don’t churn it:

- `coding`
- `writing_docs`
- `messaging_email`
- `browsing_research`
- `meetings_calls`
- `break_idle`

Labels are stored as **time spans** (not per-keystroke events).

---

## Data Flow Overview

### Structures (pipelines)
* ETL pipeline reads raw → produces features parquet
* Training pipeline reads features + labels → produces model
* Inference pipeline reads new events → emits predictions + segments

### Batch (repeatable)
1. **Ingest**: pull ActivityWatch export or API -> `data/raw/`
2. **Feature build**: events -> per-minute features -> `data/processed/features_v1/`
3. **Label import**: label spans -> `data/processed/labels_v1/`
4. **Train**: join features + labels (split by day) -> `models/<run_id>/`
5. **Report**: daily summaries -> `artifacts/reports/`

### Online (real-time)
Every N seconds:
- read the last minute(s) of events
- compute the latest feature bucket
- predict + smooth
- append predictions -> `artifacts/predictions/`
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
uv run taskclf labels import --file data/interim/labels.csv
```

### Train baseline model

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

### Produce report

```bash
uv run taskclf report daily --segments-file artifacts/segments.json
```

---

## Repo Layout

High-level directories:

* `src/taskclf/` — application code
* `configs/` — configuration (schema version, hashing policy, model params)
* `data/` — raw and processed datasets (local)
* `models/` — trained model bundles
* `artifacts/` — predictions, reports, evaluation outputs
* `scripts/` — one-off utilities

Each folder contains its own README describing scope and invariants.

---

## Model Artifact Contract

Every saved model bundle contains:

* the model file
* `metadata.json` including:

  * feature schema version + schema hash
  * label set
  * training date range
  * metrics summary
    Inference refuses to run if schema mismatches.

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
