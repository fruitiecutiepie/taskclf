# cli/

Human-facing command surface area (stable).

## Command Groups

### `ingest`
- `aw` — ingest ActivityWatch JSON export into privacy-safe events

### `features`
- `build` — build per-minute feature rows for a date

### `labels`
- `import` — import label spans from CSV
- `add-block` — create a manual label block for a time range
- `show-queue` — show pending items in the active labeling queue
- `project` — project label blocks onto feature windows

### `train`
- `build-dataset` — build training dataset (join features + labels, split)
- `lgbm` — train a LightGBM multiclass model
- `evaluate` — run full evaluation (metrics, calibration, acceptance checks)
- `tune-reject` — sweep reject thresholds and recommend best
- `calibrate` — fit per-user probability calibrators
- `retrain` — run full retrain pipeline (train, evaluate, gate-check, promote)
- `check-retrain` — check if retraining or calibrator update is due

### `taxonomy`
- `validate` — validate a taxonomy YAML file
- `show` — display taxonomy mapping as a Rich table
- `init` — generate default taxonomy YAML

### `infer`
- `batch` — batch inference (predict, smooth, segmentize)
- `online` — online inference loop (poll ActivityWatch)
- `baseline` — rule-based baseline inference (no ML model)
- `compare` — compare baseline vs ML model on labeled data

### `report`
- `daily` — generate daily report from segments JSON

### `monitor`
- `drift-check` — run drift detection comparing reference vs current
- `telemetry` — compute telemetry snapshot and append to store
- `show` — display recent telemetry snapshots

## Responsibilities
- Typer app entrypoint
- Commands call pipeline functions (thin wrapper)
- Provide consistent flags and defaults (via `core.defaults`)

## Invariants
- CLI should remain backward compatible whenever possible.
- Keep business logic out of CLI; delegate to packages.
