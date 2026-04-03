# cli.main

Typer CLI entrypoint and commands.

## Entry point

The console script (`taskclf`) is defined in `cli.entry.cli_entry()`.
It intercepts `--version` / `-v` **only when that is the first argument
after the executable** (for example `taskclf --version` or `entry -v`),
before importing the full command module, so that version queries return
instantly without loading the entire Typer command tree and its transitive
dependencies.  Invocations such as `taskclf tray … --title-salt -v` still
load the full CLI so `-v` is handled as a value, not a version shortcut.
All other invocations delegate to `cli_main()` in this module.

## Crash handler

The CLI entry point (`cli_main()`) wraps all commands in a top-level
crash handler.  If an unhandled exception escapes, a timestamped crash
report is written to `<TASKCLF_HOME>/logs/crash_<YYYYMMDD_HHMMSS>.txt`
and the file path plus issue URL are printed to stderr.  `SystemExit`
and `KeyboardInterrupt` pass through normally.

See [core.crash](../core/crash.md) for crash file contents and privacy
details.

## Release version bumps (Makefile)

Targets at the repo root:

- **`bump-patch` / `bump-minor` / `bump-major`** — bump `pyproject.toml` and `uv.lock`, commit, tag **`vX.Y.Z`**, and trigger **`.github/workflows/payload-release.yml`**. Use for a **Python / sidecar payload** release.
- **`bump-launcher-patch` / `bump-launcher-minor` / `bump-launcher-major`** — bump **`electron/package.json`**, commit, tag **`launcher-vX.Y.Z`**, and trigger **`.github/workflows/electron-release.yml`**. Use for a **desktop launcher** release (installers + payload zips + `manifest.json`).

You only need **`bump-patch`** when you want a **`v*`** tag; you only need **`bump-launcher-*`** when you want a **`launcher-v*`** tag. They are independent unless you choose to run both.

**Guards:** By default, a bump aborts if there are no changes since the last matching tag under the paths that affect that release (see `Makefile`: `PAYLOAD_BUMP_PATHS` / `LAUNCHER_BUMP_PATHS`). Set **`BUMP_FORCE=1`** to tag anyway (e.g. re-release the same tree or you only changed docs).

**CI:** On tag push, workflows compare `github.event.before` to the tagged commit and **skip** the job when only unrelated files changed; **`workflow_dispatch`** always runs the full workflow.

## Commands

| Command | Description |
|---------|-------------|
| `taskclf ingest aw` | Ingest an ActivityWatch JSON export |
| `taskclf features build` | Build per-minute feature rows by fetching from ActivityWatch (supports single date or date range) |
| `taskclf labels import` | Import label spans from CSV |
| `taskclf labels add-block` | Create a manual label block for a time range |
| `taskclf labels label-now` | Label the last N minutes (no timestamps needed) |
| `taskclf labels show-queue` | Show pending items in the active labeling queue |
| `taskclf labels export` | Export labels.parquet to CSV |
| `taskclf labels project` | Project label blocks onto feature windows |
| `taskclf train build-dataset` | Build training dataset (X/y/splits artifacts) |
| `taskclf train lgbm` | Train a LightGBM multiclass model |
| `taskclf train evaluate` | Evaluate a trained model: metrics, calibration, acceptance checks |
| `taskclf train tune-reject` | Sweep reject thresholds and recommend the best one |
| `taskclf train calibrate` | Fit per-user probability calibrators |
| `taskclf train retrain` | Run full retrain pipeline (train, evaluate, gate-check, promote) |
| `taskclf train check-retrain` | Check whether retraining or calibrator update is due |
| `taskclf train list` | List model bundles with ranking metrics and status |
| `taskclf model set-active` | Manually set the active model pointer (rollback / override) |
| `taskclf policy show` | Print the current inference policy |
| `taskclf policy create` | Create an inference policy binding model + calibrator + threshold |
| `taskclf policy remove` | Remove the inference policy (falls back to active.json) |
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
| `taskclf diagnostics` | Collect environment info for bug reports |
| `taskclf ui` | Launch the labeling UI as a native floating window |
| `taskclf electron` | Launch the Electron desktop shell with native tray + multi-window popups |
| `taskclf tray` | Run system tray labeling app with activity transition detection |

### features build

Build per-minute feature rows by fetching from a running ActivityWatch
server.  Supports a single `--date` or a `--date-from` / `--date-to`
range for backfilling multiple days.

```bash
# Single date
taskclf features build --date 2026-03-10

# Date range (backfill)
taskclf features build --date-from 2026-02-16 --date-to 2026-03-12

# Dummy/synthetic features for testing
taskclf features build --date 2026-03-10 --synthetic
```

| Option | Default | Description |
|--------|---------|-------------|
| `--date` | -- | Single date (YYYY-MM-DD) |
| `--date-from` | -- | Start of date range (inclusive) |
| `--date-to` | -- | End of date range (inclusive; defaults to `--date-from` if omitted) |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--title-salt` | `taskclf-default-salt` | Salt for hashing window titles |
| `--synthetic` | `False` | Generate dummy features instead of fetching from AW |
| `--data-dir` | `<TASKCLF_HOME>/data/processed` | Output directory for parquet files |

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

### labels export

Export label spans from `labels.parquet` to a CSV file.  Columns
written: `start_ts`, `end_ts`, `label`, `provenance`, `user_id`,
`confidence`, `extend_forward`.

```bash
taskclf labels export
```

With a custom output path:

```bash
taskclf labels export --out ~/Desktop/my_labels.csv
```

| Option | Default | Description |
|---|---|---|
| `--out`, `-o` | `labels.csv` | Destination CSV file path |
| `--data-dir` | `<TASKCLF_HOME>/data/processed` | Processed data directory (reads `labels_v1/labels.parquet`) |

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

### train evaluate

Run full evaluation of a trained model against labeled data.  Outputs
overall metrics, per-class precision/recall/F1, per-user macro-F1,
and acceptance-check verdicts.  Writes `evaluation.json`,
`calibration.json`, `confusion_matrix.csv`, and `calibration.png` to
the output directory.

```bash
taskclf train evaluate \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

With user holdout for seen/unseen evaluation:

```bash
taskclf train evaluate \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 \
  --holdout-fraction 0.1 --reject-threshold 0.55
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to a model run directory |
| `--from` | *(required)* | Start date (YYYY-MM-DD) |
| `--to` | *(required)* | End date (YYYY-MM-DD, inclusive) |
| `--synthetic` | off | Generate dummy features + labels |
| `--data-dir` | `data/processed` | Processed data directory |
| `--out-dir` | `artifacts` | Output directory for evaluation artifacts |
| `--holdout-fraction` | `0.0` | Fraction of users held out for unseen-user evaluation |
| `--reject-threshold` | `0.55` | Max-probability below which prediction is rejected |

### train tune-reject

Sweep reject thresholds on a validation set and recommend the optimal
value.  Outputs a Rich table showing accuracy, reject rate, coverage,
and macro-F1 at each threshold.  Writes `reject_tuning.json` to the
output directory.

```bash
taskclf train tune-reject \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to a model run directory |
| `--from` | *(required)* | Start date (YYYY-MM-DD) |
| `--to` | *(required)* | End date (YYYY-MM-DD, inclusive) |
| `--synthetic` | off | Generate dummy features + labels |
| `--data-dir` | `data/processed` | Processed data directory |
| `--out-dir` | `artifacts` | Output directory for tuning report |
| `--write-policy / --no-write-policy` | off | Write an inference policy binding model + calibrator + tuned threshold |
| `--calibrator-store` | *(none)* | Path to calibrator store (included in policy when `--write-policy`) |
| `--models-dir` | `models` | Base directory for model bundles (used for policy file location) |

### train calibrate

Fit per-user probability calibrators and save a calibrator store.
Reports each user's eligibility (labeled windows, days, distinct
labels) and writes the store (global + per-user calibrators) to disk.

```bash
taskclf train calibrate \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 --synthetic
```

With custom eligibility thresholds and isotonic method:

```bash
taskclf train calibrate \
  --model-dir models/run_20250615_120000 \
  --from 2025-06-10 --to 2025-06-15 \
  --method isotonic \
  --min-windows 100 --min-days 2 --min-labels 2
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to a model run directory |
| `--from` | *(required)* | Start date (YYYY-MM-DD) |
| `--to` | *(required)* | End date (YYYY-MM-DD, inclusive) |
| `--synthetic` | off | Generate dummy features + labels |
| `--data-dir` | `data/processed` | Processed data directory |
| `--out` | `artifacts/calibrator_store` | Output directory for calibrator store |
| `--method` | `temperature` | Calibration method: `temperature` or `isotonic` |
| `--min-windows` | `200` | Minimum labeled windows for per-user calibration |
| `--min-days` | `3` | Minimum distinct days for per-user calibration |
| `--min-labels` | `3` | Minimum distinct core labels for per-user calibration |

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

### policy show

Print the current inference policy or report that none exists.

```bash
taskclf policy show
```

| Option | Default | Description |
|---|---|---|
| `--models-dir` | `models` | Base directory for model bundles |

### policy create

Create an inference policy binding model + calibrator store + reject
threshold.  Validates all bindings before writing.

```bash
taskclf policy create \
  --model-dir models/run_001 \
  --calibrator-store artifacts/calibrator_store \
  --reject-threshold 0.55
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to a model run directory |
| `--reject-threshold` | `0.55` | Reject threshold for this model+calibration pair |
| `--calibrator-store` | *(none)* | Path to calibrator store directory |
| `--models-dir` | `models` | Base directory for model bundles |

### policy remove

Remove the inference policy file.  Inference falls back to
`active.json` resolution.

```bash
taskclf policy remove
```

| Option | Default | Description |
|---|---|---|
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

### diagnostics

Collect environment and runtime info for bug reports.  Prints a
human-readable summary by default, or JSON with `--json`.

```bash
taskclf diagnostics
```

Output as JSON:

```bash
taskclf diagnostics --json
```

Include log tail:

```bash
taskclf diagnostics --json --include-logs --log-lines 100
```

Write to a file:

```bash
taskclf diagnostics --json --out diagnostics.json
```

**Output includes:**
- `taskclf` version, Python version, OS and architecture
- Resolved `TASKCLF_HOME` path
- ActivityWatch reachability (graceful on failure)
- Available model bundles (from `models/`)
- Config summary (`user_id` always redacted)
- Disk usage of `data/`, `models/`, `logs/`
- Last N log lines (when `--include-logs` is passed)

| Option | Default | Description |
|---|---|---|
| `--json` | off | Output as JSON instead of human-readable text |
| `--include-logs` | off | Append the last N log lines to the output |
| `--log-lines` | `50` | Number of log lines to include (requires `--include-logs`) |
| `--out` | *(stdout)* | Write output to a file instead of stdout |

### tray

Run a persistent system tray app that polls ActivityWatch, detects
activity transitions, and prompts for labels.  Automatically starts the
web UI server.  Left-clicking the tray icon opens the web dashboard
where all labeling is done through the UI.  Right-clicking shows a
minimal menu with "Toggle Dashboard" and "Quit".  When `--model-dir` is
provided, the app suggests labels using the trained model.

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

Open in browser instead of native window (useful combined with `--dev`):

```bash
taskclf tray --dev --browser
```

Keep the browser server headless for another host shell (for example Electron):

```bash
taskclf tray --browser --no-open-browser --no-tray
```

Fully browser-based (no native tray icon, no pywebview):

```bash
taskclf tray --dev --browser --no-tray
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(none)* | Model bundle for label suggestions |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--poll-seconds` | `60` | Seconds between AW polls |
| `--aw-timeout` | `10` | Seconds to wait for AW API responses |
| `--transition-minutes` | `3` | Minutes a new app must persist before prompting |
| `--data-dir` | `data/processed` (ephemeral in `--dev`) | Processed data directory; omit with `--dev` for an auto-cleaned temp dir |
| `--port` | `8741` | Port for the embedded web UI server |
| `--dev` | off | Vite hot reload + ephemeral data dir (unless `--data-dir` is set) |
| `--browser` | off | Open UI in browser instead of native window |
| `--open-browser` / `--no-open-browser` | `--open-browser` | When `--browser` is set, open the dashboard automatically in the default browser |
| `--no-tray` | off | Skip the native tray icon (use with `--browser` for browser-only mode) |

### electron

Launch the optional Electron desktop shell.  Electron owns the native
tray icon and three frameless BrowserWindows (compact pill, label popup,
state panel popup), while the existing Python tray backend runs as a
sidecar process in browser mode without opening a separate browser tab.

```bash
taskclf electron
```

With a model for label suggestions:

```bash
taskclf electron --model-dir models/run_20260226
```

With frontend hot reload and an ephemeral data dir:

```bash
taskclf electron --dev
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | *(none)* | Model bundle for label suggestions |
| `--models-dir` | `models` | Directory containing model bundles |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--poll-seconds` | `60` | Seconds between AW polls |
| `--title-salt` | `taskclf-default-salt` | Salt for hashing window titles |
| `--data-dir` | `data/processed` (ephemeral in `--dev`) | Processed data directory; omit with `--dev` for an auto-cleaned temp dir |
| `--transition-minutes` | `3` | Minutes a new app must persist before prompting |
| `--port` | `8741` | Port for the embedded web UI server and Electron sidecar |
| `--dev` | off | Enable frontend hot reload inside Electron; uses an ephemeral data dir unless `--data-dir` is set |
| `--username` | *(none)* | Display name to persist in `config.json` |
| `--retrain-config` | *(none)* | Retrain YAML config path passed through to the tray backend |

### ui

Launch the labeling UI as a native floating window with live prediction
streaming.  Starts a FastAPI server, an `ActivityMonitor` background
thread, and a pywebview window.  When `--model-dir` is provided, the
app suggests labels on activity transitions using the trained model.

```bash
taskclf ui
```

With a model for live predictions:

```bash
taskclf ui --model-dir models/run_20260226
```

With frontend hot reload for development:

```bash
taskclf ui --dev
```

Open in browser instead of native window:

```bash
taskclf ui --browser
```

Combined dev + browser mode:

```bash
taskclf ui --dev --browser
```

| Option | Default | Description |
|---|---|---|
| `--port` | `8741` | Port for the web UI server |
| `--model-dir` | *(none)* | Model bundle for live predictions |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--poll-seconds` | `60` | Seconds between AW polling iterations |
| `--title-salt` | `taskclf-default-salt` | Salt for hashing window titles |
| `--data-dir` | `data/processed` (ephemeral in `--dev`) | Processed data directory; omit with `--dev` for an auto-cleaned temp dir |
| `--transition-minutes` | `3` | Minutes a new app must persist before suggesting a label |
| `--browser` | off | Open UI in browser instead of native window |
| `--dev` | off | Start Vite dev server for frontend hot reload; uses ephemeral data dir unless `--data-dir` is set |

::: taskclf.cli.main
