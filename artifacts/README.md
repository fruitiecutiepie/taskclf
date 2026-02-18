# artifacts/

Outputs produced by running pipelines.

## Subfolders
- `predictions/` — per-minute predictions (parquet/csv)
- `segments/` — merged task segments (start/end + label)
- `reports/` — daily/weekly summaries (json/csv/html)
- `eval/` — plots and evaluation outputs from training runs

## Invariants
- Artifacts are derived outputs; safe to delete and regenerate.
- Prefer partitioned parquet for large timeseries.
