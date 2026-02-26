# src/taskclf/

Main package.

## Design principles
- **Adapters** isolate unstable platform/tool integrations.
- **Core** defines contracts, validation, and data/model IO.
- **Pipelines** compose pure transforms into repeatable runs.
- **CLI** is the stable interface for humans and automation.

## Subpackages
- `core/` — schemas, validation, storage primitives, model IO, metrics, drift detection, telemetry
- `adapters/` — ActivityWatch + input collectors
- `features/` — feature computation (event -> bucketed features, rolling windows, sessions)
- `labels/` — label span formats, import/export, projection onto feature windows, active labeling queue, weak label rules
- `train/` — dataset construction, splits, training, evaluation, calibration, retraining pipeline
- `infer/` — batch and online inference, rule-based baseline, smoothing, calibration, taxonomy mapping, drift monitoring
- `report/` — daily summaries and exports (JSON/CSV/Parquet)
- `cli/` — Typer entrypoint and commands
- `ui/` — labeling UI (Streamlit)
