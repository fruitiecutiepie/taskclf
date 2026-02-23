# cli/

Human-facing command surface area (stable).

## Commands
- `taskclf ingest aw` — ingest ActivityWatch JSON export into privacy-safe events
- `taskclf features build` — build per-minute feature rows for a date
- `taskclf labels import` — import label spans from CSV
- `taskclf train lgbm` — train a LightGBM multiclass model
- `taskclf infer batch` — batch inference: predict, smooth, segmentize
- `taskclf infer online` — real-time inference loop polling ActivityWatch
- `taskclf report daily` — generate daily report from segments

## Responsibilities
- Typer app entrypoint
- Commands call pipeline functions (thin wrapper)
- Provide consistent flags and defaults (via `core.defaults`)

## Invariants
- CLI should remain backward compatible whenever possible.
- Keep business logic out of CLI; delegate to packages.
