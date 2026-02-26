# data/

Local datasets. This directory is usually git-ignored.

- `processed/` must be reproducible from `raw/` + code + config.

## Subfolders
- `raw/` — immutable-ish source snapshots (ActivityWatch exports, etc.)
- `processed/` — clean, versioned datasets used for training/inference

## Recommended layout
- `raw/aw/<YYYY-MM-DD>/events.parquet` — ingested window events
- `raw/aw-input/<YYYY-MM-DD>/events.parquet` — ingested input events
- `processed/features_v1/date=<YYYY-MM-DD>/features.parquet` — per-minute features
- `processed/labels_v1/labels.parquet` — label spans (single file)
- `processed/labels_v1/projected_labels.parquet` — labels projected onto feature windows
- `processed/labels_v1/queue.json` — active labeling queue
- `processed/training_dataset/` — X/y/splits arrays for training

## Invariants
- `raw/` should be append-only (don't "fix" raw data).
- No raw keystrokes or raw window titles should be stored here.
