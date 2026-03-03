# train.calibrate

Per-user probability calibration: eligibility checks and calibrator fitting.

## Overview

Training-side logic for the personalization pipeline.  After a model
is trained, this module fits probability calibrators on validation data
so that predicted confidences better reflect true accuracy:

```
model + labeled_df → predict → fit global calibrator
                             → check each user's eligibility
                             → fit per-user calibrators (eligible users only)
                             → CalibratorStore + eligibility reports
```

The resulting [`CalibratorStore`](../infer/calibration.md) is used at
inference time to adjust raw model probabilities before the reject
decision.

## Models

### PersonalizationEligibility

Frozen Pydantic model reporting whether a user qualifies for per-user
calibration.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `str` | User identifier |
| `labeled_windows` | `int` | Number of labeled windows for this user |
| `labeled_days` | `int` | Number of distinct calendar days with labels |
| `distinct_labels` | `int` | Number of distinct core labels observed |
| `is_eligible` | `bool` | Whether all thresholds are met |

## Eligibility thresholds

A user must meet all three thresholds to receive a per-user calibrator.
Defaults are defined in [`core.defaults`](../core/defaults.md):

| Threshold | Default | Description |
|-----------|---------|-------------|
| `min_windows` | `DEFAULT_MIN_LABELED_WINDOWS` (200) | Minimum labeled window count |
| `min_days` | `DEFAULT_MIN_LABELED_DAYS` (3) | Minimum distinct calendar days |
| `min_labels` | `DEFAULT_MIN_DISTINCT_LABELS` (3) | Minimum distinct core labels |

Ineligible users fall back to the global calibrator at inference time.

## Functions

### check_personalization_eligible

```python
check_personalization_eligible(
    df: pd.DataFrame,
    user_id: str,
    *,
    min_windows: int = DEFAULT_MIN_LABELED_WINDOWS,
    min_days: int = DEFAULT_MIN_LABELED_DAYS,
    min_labels: int = DEFAULT_MIN_DISTINCT_LABELS,
) -> PersonalizationEligibility
```

Checks whether `user_id` has enough labeled data.  Returns a
`PersonalizationEligibility` report.  If the user is not present in
`df`, returns a report with all counts at 0 and `is_eligible=False`.

### fit_temperature_calibrator

```python
fit_temperature_calibrator(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
) -> TemperatureCalibrator
```

Finds the temperature scalar that minimizes negative log-likelihood
on validation data.  Uses a two-pass grid search:

1. **Coarse**: 0.1 to 5.0, step 0.1
2. **Fine**: best ± 0.1, step 0.01

Returns a `TemperatureCalibrator` with the optimal temperature.

### fit_isotonic_calibrator

```python
fit_isotonic_calibrator(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> IsotonicCalibrator
```

Fits per-class `sklearn.isotonic.IsotonicRegression` with
`y_min=0.0`, `y_max=1.0`, `out_of_bounds="clip"`.  Returns an
`IsotonicCalibrator` wrapping the fitted regressors.

### fit_calibrator_store

```python
fit_calibrator_store(
    model: lgb.Booster,
    labeled_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    method: Literal["temperature", "isotonic"] = DEFAULT_CALIBRATION_METHOD,
    min_windows: int = DEFAULT_MIN_LABELED_WINDOWS,
    min_days: int = DEFAULT_MIN_LABELED_DAYS,
    min_labels: int = DEFAULT_MIN_DISTINCT_LABELS,
) -> tuple[CalibratorStore, list[PersonalizationEligibility]]
```

Orchestrates the full calibration flow:

1. Predicts on `labeled_df` to get raw probabilities
2. Fits a global calibrator on all validation data
3. Checks each user's eligibility
4. Fits per-user calibrators for qualifying users

Returns a `(CalibratorStore, eligibility_reports)` tuple.  The
default calibration method is `"temperature"`
(from [`core.defaults`](../core/defaults.md)).

## Method comparison

| | Temperature | Isotonic |
|-|-------------|----------|
| **Parameters** | Single scalar `T` | Per-class non-parametric regression |
| **Size** | Lightweight (one float) | Larger (one `IsotonicRegression` per class) |
| **Flexibility** | Uniform scaling across all classes | Independent adjustment per class |
| **Best for** | Well-calibrated models needing minor adjustment | Models with class-specific miscalibration |
| **Risk** | Cannot fix per-class bias | Can overfit with small validation sets |

## Usage

```python
from taskclf.train.calibrate import fit_calibrator_store
from taskclf.infer.calibration import save_calibrator_store

store, reports = fit_calibrator_store(
    model, val_df,
    cat_encoders=cat_encoders,
    method="temperature",
)

for r in reports:
    print(f"{r.user_id}: eligible={r.is_eligible}")

save_calibrator_store(store, Path("artifacts/calibrator_store"))
```

See the [personalization guide](../../guide/personalization.md) for
end-to-end setup and the
[`infer.calibration`](../infer/calibration.md) page for runtime
calibrator usage.

::: taskclf.train.calibrate
