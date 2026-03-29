# train.evaluate

Full model evaluation pipeline: metrics, calibration, acceptance checks.

## Overview

Evaluates a trained LightGBM model against labeled test data and
produces a comprehensive report with acceptance-gate verdicts.
Supports multiple evaluation modes so offline metrics align with
deployed inference behavior:

```
model + test_df → evaluate_model → EvaluationReport
                                       ├── overall metrics (macro/weighted F1)
                                       ├── per-class precision/recall/F1
                                       ├── per-user macro-F1
                                       ├── calibration curves
                                       ├── user stratification
                                       ├── reject rate
                                       ├── flip rate
                                       ├── segment duration distribution
                                       └── acceptance checks (pass/fail)
```

Predictions with max probability below the reject threshold are
classified as `Mixed/Unknown` (from [`core.defaults`](../core/defaults.md)).

## Models

### EvaluationReport

Frozen Pydantic model containing all evaluation artifacts.

| Field | Type | Description |
|-------|------|-------------|
| `macro_f1` | `float` | Overall macro-averaged F1 |
| `weighted_f1` | `float` | Overall weighted-averaged F1 |
| `per_class` | `dict[str, dict[str, float]]` | Per-class precision, recall, F1 |
| `confusion_matrix` | `list[list[int]]` | Confusion matrix as nested lists |
| `label_names` | `list[str]` | Ordered label names (rows/columns of confusion matrix) |
| `per_user` | `dict[str, dict[str, float]]` | Per-user macro-F1 and row count |
| `calibration` | `dict[str, dict[str, list[float]]]` | Per-class calibration curve data (`fraction_of_positives`, `mean_predicted_value`) |
| `stratification` | `dict[str, Any]` | User stratification report with optional warnings |
| `seen_user_f1` | `float \| None` | Macro-F1 on users seen during training (requires `holdout_users`) |
| `unseen_user_f1` | `float \| None` | Macro-F1 on held-out users (requires `holdout_users`) |
| `reject_rate` | `float` | Fraction of predictions below the reject threshold |
| `acceptance_checks` | `dict[str, bool]` | Named acceptance gates (pass/fail) |
| `acceptance_details` | `dict[str, str]` | Human-readable detail string per check |
| `flip_rate` | `float \| None` | Label-change rate (transitions / total windows) |
| `segment_duration_distribution` | `dict[str, int] \| None` | Histogram of segment durations by bucket (`"60s"`, `"120s"`, `"180s"`, `"300s"`, `"300s+"`) |
| `eval_mode` | `str` | Which evaluation pipeline was used (`"raw"`, `"calibrated"`, `"calibrated_reject"`, `"smoothed"`, `"interval"`) |

### RejectTuningResult

Result of sweeping reject thresholds on a validation set.

| Field | Type | Description |
|-------|------|-------------|
| `best_threshold` | `float` | Threshold maximizing accuracy on accepted windows within reject-rate bounds |
| `sweep` | `list[dict[str, float]]` | Per-threshold row with `threshold`, `accuracy_on_accepted`, `reject_rate`, `coverage`, `macro_f1` |

## Evaluation modes

The `eval_mode` parameter controls the evaluation pipeline:

| Mode | Calibrator | Reject | Smoothing | Description |
|------|-----------|--------|-----------|-------------|
| `"raw"` | No | Yes | No | Default; raw model probabilities with reject threshold |
| `"calibrated"` | Yes | No | No | Calibrated probabilities, no reject |
| `"calibrated_reject"` | Yes | Yes | No | Calibrated probabilities with reject threshold |
| `"smoothed"` | Yes | Yes | Yes | Calibrated + reject + rolling-majority smoothing |
| `"interval"` | Yes | Yes | Yes | Smoothed predictions aggregated into segments; interval-level accuracy |

Non-raw modes require a `calibrator` implementing the
[`Calibrator`](../infer/calibration.md) protocol.

## Acceptance checks

All checks must pass for a model to be promoted.  Thresholds are
defined in the module constants and align with
[`docs/guide/acceptance.md`](../../guide/acceptance.md):

| Check | Threshold | Description |
|-------|-----------|-------------|
| `macro_f1` | >= 0.65 | Overall macro-F1 |
| `weighted_f1` | >= 0.70 | Overall weighted-F1 |
| `breakidle_precision` | >= 0.95 | BreakIdle class precision |
| `breakidle_recall` | >= 0.90 | BreakIdle class recall |
| `no_class_below_50_precision` | >= 0.50 | Per-class precision floor |
| `reject_rate_bounds` | [0.05, 0.30] | Reject rate within window |
| `seen_user_f1` | >= 0.70 | Seen-user macro-F1 (when holdout users provided) |
| `unseen_user_f1` | >= 0.60 | Unseen-user macro-F1 (when holdout users provided) |

## Functions

### evaluate_model

```python
evaluate_model(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    holdout_users: Sequence[str] = (),
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    eval_mode: Literal["raw", "calibrated", "calibrated_reject", "smoothed", "interval"] = "raw",
    calibrator: Calibrator | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> EvaluationReport
```

Runs comprehensive evaluation: overall metrics, per-class and per-user
breakdowns, calibration curves, user stratification, and acceptance
checks.  When `holdout_users` is non-empty, computes separate
seen/unseen-user F1 scores.

The `eval_mode` parameter selects the evaluation pipeline (see table
above).  Non-raw modes require a `calibrator`.

### tune_reject_threshold

```python
tune_reject_threshold(
    model: lgb.Booster,
    val_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    thresholds: Sequence[float] | None = None,
    reject_rate_min: float = 0.05,
    reject_rate_max: float = 0.30,
    calibrator: Calibrator | None = None,
) -> RejectTuningResult
```

Sweeps candidate thresholds (default `np.arange(0.10, 1.00, 0.05)`)
and picks the one that maximizes accuracy on accepted windows while
keeping the reject rate within `[reject_rate_min, reject_rate_max]`.
Falls back to `DEFAULT_REJECT_THRESHOLD` (0.55) if no candidate
satisfies the bounds.

When `calibrator` is provided, raw probabilities are calibrated before
extracting confidences for the threshold sweep.  This ensures the
threshold is tuned on the same probability space used at inference time.

### write_evaluation_artifacts

```python
write_evaluation_artifacts(
    report: EvaluationReport,
    output_dir: Path,
) -> dict[str, Path]
```

Writes evaluation artifacts to disk:

| File | Content |
|------|---------|
| `evaluation.json` | Full report as JSON |
| `calibration.json` | Per-class calibration curve data |
| `confusion_matrix.csv` | Labeled confusion matrix |
| `calibration.png` | Per-class calibration plots (optional, requires matplotlib) |

Returns a dict mapping artifact name to its written path.

## Usage

```python
from taskclf.train.evaluate import (
    evaluate_model,
    tune_reject_threshold,
    write_evaluation_artifacts,
)
from taskclf.core.model_io import load_model_bundle
from taskclf.infer.calibration import TemperatureCalibrator

model, metadata, cat_encoders = load_model_bundle(Path("models/run_001"))

# Raw evaluation (default)
raw_report = evaluate_model(
    model, test_df,
    cat_encoders=cat_encoders,
    holdout_users=["user-X"],
)
print(f"Macro F1: {raw_report.macro_f1:.4f}")
print(f"Flip rate: {raw_report.flip_rate:.4f}")

# Calibrated evaluation
cal = TemperatureCalibrator(temperature=1.2)
cal_report = evaluate_model(
    model, test_df,
    cat_encoders=cat_encoders,
    eval_mode="calibrated",
    calibrator=cal,
)
print(f"Calibrated F1: {cal_report.macro_f1:.4f}")

# Tune reject threshold on calibrated scores
result = tune_reject_threshold(
    model, val_df,
    cat_encoders=cat_encoders,
    calibrator=cal,
)
print(f"Best threshold: {result.best_threshold}")

# Write artifacts
paths = write_evaluation_artifacts(raw_report, Path("artifacts/eval"))
```

::: taskclf.train.evaluate
