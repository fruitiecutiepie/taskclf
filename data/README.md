# data/

Local datasets and intermediate artifacts. This directory is usually git-ignored.
.
- `processed/` must be reproducible from `raw/` + code
## Subfolders
- `raw/` — immutable-ish source snapshots (ActivityWatch exports, JSONL, etc.)
- `interim/` — scratch outputs used during labeling or debugging
- `processed/` — clean, versioned datasets used for training/inference

## Recommended layout
- `processed/features_v1/date=YYYY-MM-DD/*.parquet`
- `processed/labels_v1/date=YYYY-MM-DD/*.parquet` (or a single spans file per day)

## Invariants
- `raw/` should be append-only (don’t “fix” raw data) + config.
- No raw keystrokes or raw window titles should be stored here.
