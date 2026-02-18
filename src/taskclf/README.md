# src/taskclf/

Main package.

## Design principles
- **Adapters** isolate unstable platform/tool integrations.
- **Core** defines contracts, validation, and data/model IO.
- **Pipelines** compose pure transforms into repeatable runs.
- **CLI** is the stable interface for humans and automation.

## Subpackages
- `core/` — schemas, validation, storage primitives, model IO, metrics
- `adapters/` — ActivityWatch + input collectors
- `features/` — feature computation (event -> bucketed features)
- `labels/` — label span formats, import/export, weak label rules (optional)
- `train/` — dataset joins, splits, training, calibration
- `infer/` — online inference loop, smoothing
- `report/` — summaries and exports
- `cli/` — Typer entrypoint and commands
