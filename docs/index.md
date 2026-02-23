# taskclf

Task Type Classifier from Local Activity Signals.

Train and run a personal task-type classifier (e.g. coding / writing / meetings)
using privacy-preserving computer activity signals such as foreground app/window
metadata and aggregated input statistics (counts/rates only).

## Quick links

- [CLI Reference](api/cli/main.md) — command-line interface
- [README](https://github.com/fruitiecutiepie/taskclf#readme) — setup, CLI usage, data flow

## API Reference

All modules are auto-generated from source docstrings.

### Core

- [core/types](api/core/types.md) — Pydantic models for data contracts
- [core/schema](api/core/schema.md) — feature schema versioning and validation
- [core/store](api/core/store.md) — parquet IO primitives
- [core/hashing](api/core/hashing.md) — deterministic hashing utilities
- [core/time](api/core/time.md) — time-bucket alignment and range generation
- [core/metrics](api/core/metrics.md) — evaluation metrics
- [core/model_io](api/core/model_io.md) — model bundle persistence
- [core/logging](api/core/logging.md) — sanitizing log filter

### Features

- [features/build](api/features/build.md) — feature computation pipeline

### Labels

- [labels/store](api/labels/store.md) — label span IO and validation

### Train

- [train/dataset](api/train/dataset.md) — dataset join and time-aware splits
- [train/lgbm](api/train/lgbm.md) — LightGBM training pipeline

### Infer

- [infer/batch](api/infer/batch.md) — batch inference pipeline
- [infer/smooth](api/infer/smooth.md) — smoothing and segmentization

### Report

- [report/daily](api/report/daily.md) — daily report generation
- [report/export](api/report/export.md) — report export with sensitive-field guards
