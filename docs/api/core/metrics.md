# core.metrics

Evaluation metrics for model assessment: macro-F1, per-class
precision/recall, confusion matrices, calibration curves, reject-rate
analysis, and per-user breakdowns.  All metric functions accept
string-typed labels and the ordered label vocabulary, returning plain
dicts suitable for JSON serialisation and artifact storage.

## Function overview

| Function | Purpose |
|----------|---------|
| `compute_metrics` | Macro-F1, weighted-F1, and confusion matrix |
| `class_distribution` | Per-class counts and fractions |
| `confusion_matrix_df` | Labelled confusion matrix as a DataFrame |
| `per_class_metrics` | Per-class precision, recall, F1, and optional support |
| `top_confusion_pairs` | Largest off-diagonal confusion counts (ranked) |
| `expected_calibration_error_multiclass` | OVR binary ECE, weighted by class support |
| `multiclass_brier_score` | One-hot vs predicted probability MSE |
| `multiclass_log_loss_score` | Multiclass log loss (clipped probs) |
| `slice_metrics_by_columns` | Macro/weighted F1 and per-class metrics per slice |
| `unknown_category_rates` | Share of rows with unseen categorical values vs encoders |
| `reject_rate` | Fraction of predictions equal to the reject label |
| `compare_baselines` | Side-by-side comparison of multiple prediction methods |
| `per_user_metrics` | Macro-F1 and per-class F1 grouped by user |
| `calibration_curve_data` | Per-class reliability diagram data |
| `user_stratification_report` | Training-set imbalance analysis per user |
| `reject_rate_by_group` | Reject rate by (user, date) with drift flags |

## compute_metrics

Primary evaluation entry point.  Returns aggregate scores and the full
confusion matrix for a single set of predictions.

| Return key | Type | Description |
|------------|------|-------------|
| `macro_f1` | `float` | Unweighted mean F1 across classes |
| `weighted_f1` | `float` | Support-weighted mean F1 |
| `confusion_matrix` | `list[list[int]]` | Row = true, column = predicted |
| `label_names` | `list[str]` | Label order matching matrix axes |

```python
from taskclf.core.metrics import compute_metrics
from taskclf.core.types import LABEL_SET_V1

result = compute_metrics(y_true, y_pred, sorted(LABEL_SET_V1))
print(f"Macro-F1: {result['macro_f1']:.4f}")
```

## class_distribution

Reports how many samples belong to each class, useful for detecting
label imbalance before training.

Returns a dict mapping each label to `{"count": int, "fraction": float}`.
Labels absent from `y_true` appear with count 0.

## confusion_matrix_df

Wraps `sklearn.metrics.confusion_matrix` into a `pd.DataFrame` with
`label_names` as both the row index (true labels) and column index
(predicted labels).  Convenient for CSV export or display.

## per_class_metrics

Returns per-class precision, recall, and F1 as a nested dict.  By default
each class also includes **`support`**: the number of true instances of
that class in `y_true` (same notion as sklearn).  Pass
`include_support=False` to omit support for callers that only need P/R/F1.

```python
{
    "Build": {"precision": 0.85, "recall": 0.90, "f1": 0.87, "support": 120},
    "Meet":  {"precision": 0.92, "recall": 0.88, "f1": 0.90, "support": 45},
    ...
}
```

Uses `zero_division=0` so classes with no predictions get 0.0 instead
of warnings.

## top_confusion_pairs

Takes a square confusion matrix and label order; returns up to `k`
off-diagonal pairs `{"true_label", "pred_label", "count"}` sorted by
count descending.  Used for bundle inspection and evaluation reports.

## expected_calibration_error_multiclass

Computes a **support-weighted mean** of one-vs-rest binary expected
calibration error (uniform probability bins) across classes.  Requires
integer-encoded true labels and a probability matrix `(n_samples,
n_classes)`.

## multiclass_brier_score / multiclass_log_loss_score

Probability-based scores aligned with the same `y_proba` used in
[`train.evaluate`](../train/evaluate.md).  Log loss uses clipped
probabilities to avoid log(0).

## slice_metrics_by_columns

Default slice columns are
`user_id`, `app_id`, `app_category`, `domain_category`, `hour_of_day`
(:data:`~taskclf.core.metrics.DEFAULT_SLICE_COLUMNS`), intersected with
columns present in the frame.  For each slice value (top groups by
frequency, capped per column), returns row count, macro/weighted F1,
reject rate, and per-class metrics.

## unknown_category_rates

For each evaluated categorical column, reports the fraction of rows
whose string value is **not** in the fitted `LabelEncoder.classes_`
(what becomes `__unknown__` / legacy `-1` at encode time).  Returns
`per_column`, `overall_rate` (mean of evaluated columns), and
`columns_evaluated`.  Column set should match the feature schema (see
[`train.lgbm.get_categorical_columns`](../train/lgbm.md)).

## reject_rate

Computes the fraction of predictions matching the reject label
(default `MIXED_UNKNOWN` from [`core.defaults`](defaults.md)).
Returns 0.0 for empty input.

## compare_baselines

Evaluates multiple prediction methods against the same ground truth in
a single call.  Each method receives its own `macro_f1`, `weighted_f1`,
`reject_rate`, `per_class` breakdown, and `confusion_matrix`.

```python
from taskclf.core.metrics import compare_baselines

results = compare_baselines(
    y_true,
    {"lgbm": lgbm_preds, "majority": majority_preds},
    label_names,
)
for name, m in results.items():
    print(f"{name}: F1={m['macro_f1']:.4f}  reject={m['reject_rate']:.2%}")
```

The label vocabulary is extended with `reject_label` if it is not
already present, so reject predictions are counted in the matrix.

## per_user_metrics

Groups predictions by `user_ids` and computes per-user macro-F1 plus
per-class F1 scores.  Useful for identifying users whose data the
model struggles with.

Each user entry contains `macro_f1`, `count`, and `{label}_f1` keys.

## calibration_curve_data

Generates per-class reliability diagram data using one-vs-rest
binarization.  Requires integer-encoded true labels and a probability
matrix `(n_samples, n_classes)`.

| Return key (per class) | Type | Description |
|------------------------|------|-------------|
| `fraction_of_positives` | `list[float]` | Observed positive fraction per bin |
| `mean_predicted_value` | `list[float]` | Mean predicted probability per bin |

Classes with zero positive samples return empty lists.

## user_stratification_report

Analyses per-user contribution to the training set.  Flags users whose
row fraction exceeds `dominance_threshold` (default 0.5) as dominant,
emitting human-readable warnings.

| Return key | Type | Description |
|------------|------|-------------|
| `per_user` | `dict` | Per-user `count`, `fraction`, `label_distribution` |
| `total_rows` | `int` | Total rows in the dataset |
| `user_count` | `int` | Number of distinct users |
| `warnings` | `list[str]` | Dominance warnings (empty if balanced) |

## reject_rate_by_group

Computes reject rate grouped by `(user_id, date)` for drift detection.
Groups whose reject rate exceeds `global_reject_rate * spike_multiplier`
(default 2.0) are added to `drift_flags`.

| Return key | Type | Description |
|------------|------|-------------|
| `global_reject_rate` | `float` | Overall reject fraction |
| `per_group` | `dict` | Keyed by `"user_id\|YYYY-MM-DD"` with `reject_rate`, `total`, `rejected` |
| `drift_flags` | `list[str]` | Group keys that exceed the spike threshold |

## See also

- [`train.evaluate`](../train/evaluate.md) -- model evaluation pipeline that calls these functions
- [`infer.baseline`](../infer/baseline.md) -- baseline comparisons using `compare_baselines`
- [`core.defaults`](defaults.md) -- `MIXED_UNKNOWN` reject label constant

::: taskclf.core.metrics
