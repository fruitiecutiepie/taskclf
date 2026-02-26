# train.retrain

Retraining workflow: cadence scheduling, reproducible pipeline, regression gates.

## Overview

The retrain module provides automated retraining with safety gates:

- **Cadence checks** determine when a global retrain or calibrator update is due.
- **Dataset hashing** ensures each model can be traced to its exact training data.
- **Regression gates** prevent deploying a model that is worse than the champion.

## Configuration

Retrain configuration lives in `configs/retrain.yaml` (versioned in git):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_retrain_cadence_days` | int | 7 | Days between global model retrains |
| `calibrator_update_cadence_days` | int | 7 | Days between calibrator updates |
| `data_lookback_days` | int | 30 | Days of data to include in training |
| `regression_tolerance` | float | 0.02 | Max allowed macro-F1 drop vs champion |
| `require_baseline_improvement` | bool | true | Challenger must beat rule baseline |
| `auto_promote` | bool | false | Auto-promote when all gates pass |
| `train_params.num_boost_round` | int | 100 | LightGBM boosting rounds |
| `train_params.class_weight` | str | balanced | Class weight strategy |

## Promotion Gates

Both candidate gates and comparative gates must pass for promotion.

### Candidate Gates (`check_candidate_gates`)

Hard gates evaluated on the challenger model in isolation (no champion needed):

1. **breakidle_precision** — BreakIdle precision >= 0.95
2. **no_class_below_50_precision** — every class precision >= 0.50
3. **challenger_acceptance** — all acceptance checks from `docs/guide/acceptance.md` pass

### Comparative Gates (`check_regression_gates`)

Regression check comparing challenger against the champion:

1. **macro_f1_no_regression** — challenger macro-F1 within `regression_tolerance` of champion

Comparative gates are only evaluated when a champion exists. If no champion is available, only candidate gates apply.

## CLI

```bash
# Check if retraining is due
taskclf train check-retrain --models-dir models/

# Run full retrain pipeline
taskclf train retrain --config configs/retrain.yaml --force --synthetic

# Dry run (evaluate without promoting)
taskclf train retrain --config configs/retrain.yaml --dry-run --synthetic
```

::: taskclf.train.retrain
