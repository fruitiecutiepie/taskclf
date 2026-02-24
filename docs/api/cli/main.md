# cli.main

Typer CLI entrypoint and commands.

## Commands

| Command | Description |
|---------|-------------|
| `taskclf ingest aw` | Ingest an ActivityWatch JSON export |
| `taskclf features build` | Build per-minute feature rows for a date |
| `taskclf labels import` | Import label spans from CSV |
| `taskclf train build-dataset` | Build training dataset (X/y/splits artifacts) |
| `taskclf train lgbm` | Train a LightGBM multiclass model |
| `taskclf infer batch` | Run batch inference |
| `taskclf infer online` | Run online inference loop |
| `taskclf report daily` | Generate a daily report |

### train build-dataset

Joins features with labels, applies exclusion rules (short sessions,
missing features), splits by time (70/15/15 per user), and writes
`X.parquet`, `y.parquet`, and `splits.json`.

```bash
taskclf train build-dataset \
  --from 2025-06-10 --to 2025-06-15 \
  --synthetic \
  --holdout-fraction 0.1
```

::: taskclf.cli.main
