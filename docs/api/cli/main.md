# cli.main

Typer CLI entrypoint and commands.

## Commands

| Command | Description |
|---------|-------------|
| `taskclf ingest aw` | Ingest an ActivityWatch JSON export |
| `taskclf features build` | Build per-minute feature rows for a date |
| `taskclf labels import` | Import label spans from CSV |
| `taskclf labels add-block` | Create a manual label block for a time range |
| `taskclf labels label-now` | Label the last N minutes (no timestamps needed) |
| `taskclf labels show-queue` | Show pending items in the active labeling queue |
| `taskclf labels project` | Project label blocks onto feature windows |
| `taskclf train build-dataset` | Build training dataset (X/y/splits artifacts) |
| `taskclf train lgbm` | Train a LightGBM multiclass model |
| `taskclf train retrain` | Run full retrain pipeline (train, evaluate, gate-check, promote) |
| `taskclf train check-retrain` | Check whether retraining or calibrator update is due |
| `taskclf train list` | List model bundles with ranking metrics and status |
| `taskclf model set-active` | Manually set the active model pointer (rollback / override) |
| `taskclf taxonomy validate` | Validate a user taxonomy YAML file |
| `taskclf taxonomy show` | Display taxonomy mapping as a table |
| `taskclf taxonomy init` | Generate a default taxonomy YAML |
| `taskclf infer batch` | Run batch inference (supports `--taxonomy`) |
| `taskclf infer baseline` | Run rule-based baseline inference (no ML) |
| `taskclf infer compare` | Compare baseline vs ML model on labeled data |
| `taskclf infer online` | Run online inference loop (supports `--taxonomy`, `--label-queue`) |
| `taskclf report daily` | Generate a daily report |
| `taskclf monitor drift-check` | Run drift detection (reference vs current) |
| `taskclf monitor telemetry` | Compute and store a telemetry snapshot |
| `taskclf monitor show` | Display recent telemetry snapshots |
| `taskclf tray` | Run system tray labeling app with activity transition detection |

### labels add-block

Create a manual label block with a feature summary and optional predicted
label display.

```bash
taskclf labels add-block \
  --start 2025-06-15T10:00:00 --end 2025-06-15T11:00:00 \
  --label Build --user-id my-user --confidence 0.9
```

### labels label-now

Label the last N minutes without typing timestamps.  Queries a running
ActivityWatch server for a live summary of apps used in the window.
Falls back gracefully if AW is unreachable.

```bash
taskclf labels label-now --minutes 10 --label Build
```

With options:

```bash
taskclf labels label-now \
  --minutes 15 --label Debug \
  --user-id my-user --confidence 0.9 \
  --aw-host http://localhost:5600
```

### labels show-queue

Display pending labeling requests sorted by confidence (lowest first).

```bash
taskclf labels show-queue --user-id my-user --limit 10
```

### labels project

Run strict block-to-window projection and write per-window labels.

```bash
taskclf labels project --from 2025-06-10 --to 2025-06-15
```

### train build-dataset

Joins features with labels via strict block-to-window projection,
applies exclusion rules (short sessions, missing features), splits by
time (70/15/15 per user), and writes `X.parquet`, `y.parquet`, and
`splits.json`.

```bash
taskclf train build-dataset \
  --from 2025-06-10 --to 2025-06-15 \
  --synthetic \
  --holdout-fraction 0.1
```

### train retrain

Run the full retrain pipeline: build dataset, train a challenger model,
evaluate it, run regression gates against the current champion, and
promote if all gates pass.  Uses `configs/retrain.yaml` for cadence,
gate thresholds, and training parameters.

```bash
taskclf train retrain \
  --config configs/retrain.yaml \
  --from 2025-06-01 --to 2025-06-30 \
  --force --synthetic
```

Use `--dry-run` to evaluate without promoting:

```bash
taskclf train retrain --config configs/retrain.yaml --dry-run --synthetic
```

### train check-retrain

Check whether a global retrain or calibrator update is due (read-only).

```bash
taskclf train check-retrain \
  --models-dir models/ \
  --calibrator-store artifacts/calibrator_store
```

### train list

List all model bundles under `models/` with ranking metrics, eligibility
status, and active pointer.  Columns include macro F1, weighted F1,
BreakIdle precision (derived from confusion matrix), and minimum
per-class precision.

```bash
taskclf train list
```

Filter to eligible bundles only (compatible schema + label set):

```bash
taskclf train list --eligible
```

Sort by a different metric:

```bash
taskclf train list --sort weighted_f1
```

Output as JSON for automation:

```bash
taskclf train list --json
```

Filter to a specific schema hash:

```bash
taskclf train list --schema-hash 740b4db787e9 --eligible
```

| Option | Default | Description |
|---|---|---|
| `--models-dir` | `models` | Base directory for model bundles |
| `--sort` | `macro_f1` | Sort column: `macro_f1`, `weighted_f1`, or `created_at` |
| `--eligible` | off | Show only eligible bundles (compatible schema + label set) |
| `--schema-hash` | *(current runtime)* | Filter to bundles matching this schema hash |
| `--json` | off | Output as JSON instead of a table |

### model set-active

Manually set the active model pointer to a specific bundle.  Useful
for rollback (reverting to a known-good model) or manual override
after a bad promotion.  The bundle must be valid and compatible with
the current schema and label set.

```bash
taskclf model set-active --model-id run_20260215_120000
```

With a custom models directory:

```bash
taskclf model set-active --model-id run_20260215_120000 --models-dir /data/models
```

| Option | Default | Description |
|---|---|---|
| `--model-id` | *(required)* | Bundle directory name under `models/` |
| `--models-dir` | `models` | Base directory for model bundles |

### taxonomy validate

Validate a user taxonomy YAML file and report any errors.

```bash
taskclf taxonomy validate --config configs/user_taxonomy.yaml
```

### taxonomy show

Display the taxonomy mapping as a Rich table showing bucket names,
core labels, colors, and advanced settings.

```bash
taskclf taxonomy show --config configs/user_taxonomy.yaml
```

### taxonomy init

Generate a default taxonomy YAML with one bucket per core label
(identity mapping) as a starting point for customisation.

```bash
taskclf taxonomy init --out configs/user_taxonomy.yaml
```

### infer batch

Run batch inference with optional taxonomy mapping.  When `--taxonomy`
is provided, a `mapped_label` column is added to `predictions.csv`.

`--model-dir` is optional.  When omitted, the model is auto-resolved:
first from `models/active.json`, then by best-model selection from
`--models-dir` (default `models/`).  See `docs/guide/model_selection.md`
for the full resolution precedence.

```bash
# With explicit model directory:
taskclf infer batch \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 --synthetic \
  --taxonomy configs/user_taxonomy.yaml

# With auto-resolution (uses active.json or best model):
taskclf infer batch \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(auto-resolved)* | Path to a model bundle directory |
| `--models-dir` | `models` | Base directory for model bundles (used for auto-resolution) |

### infer online (with taxonomy, calibrator, and label queue)

Run online inference with optional taxonomy mapping and probability
calibration.  Each prediction is written as a full `WindowPrediction`
row (core label, core probs, confidence, rejection status, mapped
label, mapped probs).  Segments are hysteresis-merged so blocks shorter
than `MIN_BLOCK_DURATION_SECONDS` (3 min) are absorbed by neighbours.

`--model-dir` is optional (same auto-resolution as `infer batch`).
When `--models-dir` is provided, the online loop watches
`models/active.json` for changes and hot-reloads the model bundle
without restarting.  The swap only occurs after the new bundle loads
successfully; on failure the current model is kept.

When `--label-queue` is enabled, predictions with confidence below
`--label-confidence` (default 0.55) are auto-enqueued to the labeling
queue for manual review.  Enqueued items appear in `labels show-queue`
and the web UI.

```bash
# With auto-resolution and model reload:
taskclf infer online \
  --taxonomy configs/user_taxonomy.yaml \
  --calibrator calibrators/user_cal.json

# With explicit model directory:
taskclf infer online \
  --model-dir models/run_20250615_120000 \
  --taxonomy configs/user_taxonomy.yaml \
  --calibrator calibrators/user_cal.json
```

With label queue integration:

```bash
taskclf infer online \
  --label-queue \
  --label-confidence 0.50
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(auto-resolved)* | Path to a model bundle directory |
| `--models-dir` | `models` | Base directory for model bundles (used for auto-resolution and reload) |

### infer baseline

Run rule-based heuristic inference without a trained model.  Produces
`baseline_predictions.csv` and `baseline_segments.json`.

```bash
taskclf infer baseline \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

### infer compare

Compare rule baseline vs a trained ML model on labeled data.  Outputs a
Rich summary table and writes `baseline_vs_model.json`.

```bash
taskclf infer compare \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

### monitor drift-check

Run drift detection comparing reference vs current prediction windows.
Flags feature PSI/KS drift, reject-rate increases, entropy spikes, and
class-distribution shifts.  Optionally auto-creates labeling tasks.

```bash
taskclf monitor drift-check \
  --ref-features data/processed/features_ref.parquet \
  --cur-features data/processed/features_cur.parquet \
  --ref-predictions artifacts/predictions_ref.csv \
  --cur-predictions artifacts/predictions_cur.csv \
  --auto-label --auto-label-limit 50
```

### monitor telemetry

Compute a telemetry snapshot (feature missingness, confidence stats,
reject rate, entropy, class distribution) and append to the store.

```bash
taskclf monitor telemetry \
  --features data/processed/features.parquet \
  --predictions artifacts/predictions.csv \
  --user-id user-1 \
  --store-dir artifacts/telemetry
```

### monitor show

Display recent telemetry snapshots as a Rich table.

```bash
taskclf monitor show --store-dir artifacts/telemetry --last 10
```

### tray

Run a persistent system tray app that polls ActivityWatch, detects
activity transitions, and prompts for labels.  Automatically starts the
web UI server so the "Show/Hide Window" menu item opens the labeling
dashboard in a browser.  When `--model-dir` is provided, the app
suggests labels using the trained model.  Without a model, all 8 core
labels are presented for manual selection.

```bash
taskclf tray
```

With a model for label suggestions:

```bash
taskclf tray --model-dir models/run_20260226
```

With frontend hot reload for development:

```bash
taskclf tray --dev
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(none)* | Model bundle for label suggestions |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--poll-seconds` | `60` | Seconds between AW polls |
| `--transition-minutes` | `3` | Minutes a new app must persist before prompting |
| `--data-dir` | `data/processed` | Processed data directory |
| `--port` | `8741` | Port for the embedded web UI server |
| `--dev` | off | Pass `--dev` to the spawned `taskclf ui` subprocess for frontend hot reload |

::: taskclf.cli.main
