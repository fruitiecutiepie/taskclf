# train.lgbm

LightGBM multiclass trainer with class-weight support and evaluation.

## Overview

Trains a LightGBM gradient-boosted tree model for 8-class task
classification.  The module handles categorical encoding, feature
extraction, class-imbalance weighting, and validation-set evaluation
in a single pipeline:

```
features_df → prepare_xy → train_lgbm → (model, metrics, confusion_df, params, cat_encoders)
```

Categorical columns (`app_id`, `app_category`, `domain_category`,
`user_id`) are label-encoded to integers so LightGBM can use them as
native categoricals.  During training, rare categories (below
`min_category_freq`) and a random fraction (`unknown_mask_rate`) of
known categories are replaced with `__unknown__` so the model learns
a meaningful embedding for unseen values.  At inference time, unseen
values map to `__unknown__` (or `-1` for legacy encoders without it).

## Constants

### FEATURE_COLUMNS

Ordered list of 34 feature names consumed by the model.  The first
four are categorical; the rest are numeric:

| # | Feature | Type |
|---|---------|------|
| 0 | `app_id` | categorical |
| 1 | `app_category` | categorical |
| 2 | `is_browser` | boolean |
| 3 | `is_editor` | boolean |
| 4 | `is_terminal` | boolean |
| 5 | `app_switch_count_last_5m` | numeric |
| 6 | `app_foreground_time_ratio` | numeric |
| 7 | `app_change_count` | numeric |
| 8 | `keys_per_min` | numeric |
| 9 | `backspace_ratio` | numeric |
| 10 | `shortcut_rate` | numeric |
| 11 | `clicks_per_min` | numeric |
| 12 | `scroll_events_per_min` | numeric |
| 13 | `mouse_distance` | numeric |
| 14 | `active_seconds_keyboard` | numeric |
| 15 | `active_seconds_mouse` | numeric |
| 16 | `active_seconds_any` | numeric |
| 17 | `max_idle_run_seconds` | numeric |
| 18 | `event_density` | numeric |
| 19 | `domain_category` | categorical |
| 20 | `window_title_bucket` | numeric |
| 21 | `title_repeat_count_session` | numeric |
| 22 | `keys_per_min_rolling_5` | numeric |
| 23 | `keys_per_min_rolling_15` | numeric |
| 24 | `mouse_distance_rolling_5` | numeric |
| 25 | `mouse_distance_rolling_15` | numeric |
| 26 | `keys_per_min_delta` | numeric |
| 27 | `clicks_per_min_delta` | numeric |
| 28 | `mouse_distance_delta` | numeric |
| 29 | `app_switch_count_last_15m` | numeric |
| 30 | `hour_of_day` | numeric |
| 31 | `day_of_week` | numeric |
| 32 | `session_length_so_far` | numeric |
| 33 | `user_id` | categorical |

### CATEGORICAL_COLUMNS

Subset of `FEATURE_COLUMNS` that are label-encoded to integers for
LightGBM native categorical support:

- `app_id`
- `app_category`
- `domain_category`
- `user_id`

### Default hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objective` | `multiclass` | LightGBM objective function |
| `metric` | `multi_logloss` | Evaluation metric |
| `num_leaves` | `31` | Maximum tree leaves (complexity control) |
| `learning_rate` | `0.1` | Gradient descent step size |
| `num_boost_round` | `100` | Boosting iterations (from [`core.defaults`](../core/defaults.md)) |
| `verbose` | `-1` | Suppress LightGBM training logs |

## Functions

### encode_categoricals

```python
encode_categoricals(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    *,
    min_category_freq: int = 5,
    unknown_mask_rate: float = 0.05,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]
```

Label-encodes the four categorical columns in-place.  Operates in
two modes:

- **Fit-new** (`cat_encoders=None`): counts value frequencies, replaces
  values with count below `min_category_freq` with `"__unknown__"`,
  randomly masks `unknown_mask_rate` of remaining known values to
  `"__unknown__"` (seeded by `random_state`), then fits a
  `LabelEncoder` per column.
- **Reuse** (`cat_encoders` provided): transforms using existing
  encoders; values not in the encoder map to `"__unknown__"` if
  present, otherwise to `-1` (legacy fallback).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_category_freq` | `5` | Minimum count for a category to keep its own code |
| `unknown_mask_rate` | `0.05` | Fraction of known-category rows randomly masked to `__unknown__` |
| `random_state` | `None` | Seed for reproducible masking |

### prepare_xy

```python
prepare_xy(
    df: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[str, LabelEncoder]]
```

Extracts a `(X, y, label_encoder, cat_encoders)` tuple from a labeled
DataFrame.  Encodes categoricals, fills missing numeric values with 0,
and encodes labels against `LABEL_SET_V1` (8 classes, sorted).

### compute_sample_weights

```python
compute_sample_weights(
    y: np.ndarray,
    method: Literal["balanced", "none"] = "balanced",
) -> np.ndarray | None
```

Maps encoded labels to per-sample weights.  `"balanced"` uses
inverse class frequency:

```
weight = n_samples / (n_classes * count_per_class)
```

`"none"` returns `None` (no weighting).

### train_lgbm

```python
train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    extra_params: dict[str, Any] | None = None,
    class_weight: Literal["balanced", "none"] = "balanced",
) -> tuple[lgb.Booster, dict, pd.DataFrame, dict[str, Any], dict[str, LabelEncoder]]
```

Trains a LightGBM multiclass model and evaluates on the validation
set.  Returns a 5-tuple:

| Element | Type | Description |
|---------|------|-------------|
| `model` | `lgb.Booster` | Trained model |
| `metrics` | `dict` | Macro/weighted F1 and per-class metrics |
| `confusion_df` | `pd.DataFrame` | Confusion matrix |
| `params` | `dict` | Merged hyperparameters (includes `class_weight_method`) |
| `cat_encoders` | `dict[str, LabelEncoder]` | Fitted categorical encoders |

## Usage

```python
from taskclf.train.lgbm import train_lgbm
from taskclf.train.dataset import split_by_time

labeled_df = ...  # DataFrame with FEATURE_COLUMNS + "label"
splits = split_by_time(labeled_df)
train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)

model, metrics, confusion_df, params, cat_encoders = train_lgbm(
    train_df, val_df,
    num_boost_round=100,
    class_weight="balanced",
)
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

After training, pass `model` and `cat_encoders` to
[`evaluate_model`](evaluate.md) for full evaluation with acceptance
checks, or to [`fit_calibrator_store`](calibrate.md) for per-user
probability calibration.

::: taskclf.train.lgbm
