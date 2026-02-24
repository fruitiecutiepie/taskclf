# cli.main

Typer CLI entrypoint and commands.

## Commands

| Command | Description |
|---------|-------------|
| `taskclf ingest aw` | Ingest an ActivityWatch JSON export |
| `taskclf features build` | Build per-minute feature rows for a date |
| `taskclf labels import` | Import label spans from CSV |
| `taskclf labels add-block` | Create a manual label block for a time range |
| `taskclf labels show-queue` | Show pending items in the active labeling queue |
| `taskclf labels project` | Project label blocks onto feature windows |
| `taskclf train build-dataset` | Build training dataset (X/y/splits artifacts) |
| `taskclf train lgbm` | Train a LightGBM multiclass model |
| `taskclf infer batch` | Run batch inference |
| `taskclf infer online` | Run online inference loop |
| `taskclf report daily` | Generate a daily report |

### labels add-block

Create a manual label block with a feature summary and optional predicted
label display.

```bash
taskclf labels add-block \
  --start 2025-06-15T10:00:00 --end 2025-06-15T11:00:00 \
  --label Build --user-id my-user --confidence 0.9
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

::: taskclf.cli.main
