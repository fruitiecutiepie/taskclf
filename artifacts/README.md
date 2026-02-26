# artifacts/

Outputs produced by running pipelines. Files are written directly into this directory.

## Typical files
- `predictions.csv` — per-minute predictions with confidence and mapped labels
- `segments.json` — merged task segments (start/end + label)
- `report_<YYYY-MM-DD>.json` — daily summaries (also available as CSV/Parquet)
- `baseline_predictions.csv` / `baseline_segments.json` — rule-based baseline outputs
- `baseline_vs_model.json` — comparison report
- `drift_report.json` — feature/prediction drift analysis
- `reject_tuning.json` — reject threshold sweep results
- `calibrator_store/` — per-user probability calibrators

## Invariants
- Artifacts are derived outputs; safe to delete and regenerate.
- Prefer partitioned parquet for large timeseries.
