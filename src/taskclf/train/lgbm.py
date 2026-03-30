"""LightGBM multiclass trainer with class-weight support and evaluation."""

from __future__ import annotations

from typing import Any, Final, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import DEFAULT_NUM_BOOST_ROUND
from taskclf.core.metrics import compute_metrics, confusion_matrix_df
from taskclf.core.types import LABEL_SET_V1

FEATURE_COLUMNS: Final[list[str]] = [
    "app_id",
    "app_category",
    "is_browser",
    "is_editor",
    "is_terminal",
    "app_switch_count_last_5m",
    "app_foreground_time_ratio",
    "app_change_count",
    "app_dwell_time_seconds",
    "app_entropy_5m",
    "app_entropy_15m",
    "top2_app_concentration_15m",
    "idle_return_indicator",
    "keys_per_min",
    "backspace_ratio",
    "shortcut_rate",
    "clicks_per_min",
    "scroll_events_per_min",
    "mouse_distance",
    "active_seconds_keyboard",
    "active_seconds_mouse",
    "active_seconds_any",
    "max_idle_run_seconds",
    "event_density",
    "domain_category",
    "window_title_bucket",
    "title_repeat_count_session",
    "keys_per_min_rolling_5",
    "keys_per_min_rolling_15",
    "mouse_distance_rolling_5",
    "mouse_distance_rolling_15",
    "keys_per_min_delta",
    "clicks_per_min_delta",
    "mouse_distance_delta",
    "app_switch_count_last_15m",
    "hour_of_day",
    "day_of_week",
    "session_length_so_far",
    "user_id",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "app_id",
    "app_category",
    "domain_category",
    "user_id",
]

FEATURE_COLUMNS_V2: Final[list[str]] = [c for c in FEATURE_COLUMNS if c != "user_id"]

CATEGORICAL_COLUMNS_V2: Final[list[str]] = [
    c for c in CATEGORICAL_COLUMNS if c != "user_id"
]


def get_feature_columns(schema_version: str) -> list[str]:
    """Return the feature column list for *schema_version*.

    Raises:
        ValueError: If *schema_version* is not ``"v1"`` or ``"v2"``.
    """
    if schema_version == "v1":
        return list(FEATURE_COLUMNS)
    if schema_version == "v2":
        return list(FEATURE_COLUMNS_V2)
    raise ValueError(f"Unknown schema version: {schema_version!r}")


def get_categorical_columns(schema_version: str) -> list[str]:
    """Return the categorical column list for *schema_version*.

    Raises:
        ValueError: If *schema_version* is not ``"v1"`` or ``"v2"``.
    """
    if schema_version == "v1":
        return list(CATEGORICAL_COLUMNS)
    if schema_version == "v2":
        return list(CATEGORICAL_COLUMNS_V2)
    raise ValueError(f"Unknown schema version: {schema_version!r}")


_DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "verbose": -1,
}


_UNKNOWN_TOKEN: Final[str] = "__unknown__"


def encode_categoricals(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    *,
    min_category_freq: int = 5,
    unknown_mask_rate: float = 0.05,
    random_state: int | None = None,
    schema_version: str = "v1",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns in-place and return fitted encoders.

    During training (``cat_encoders is None``), rare values (frequency
    below *min_category_freq*) are replaced with ``"__unknown__"`` and a
    random fraction (*unknown_mask_rate*) of known values are also masked
    to ``"__unknown__"`` so the model learns a meaningful embedding for
    unseen categories.

    During inference (``cat_encoders`` provided), values not present in
    the fitted encoder are mapped to ``"__unknown__"`` if it exists in
    the encoder's vocabulary, otherwise to ``-1`` for backward
    compatibility with legacy encoders.

    Args:
        df: DataFrame with the columns listed in ``CATEGORICAL_COLUMNS``.
        cat_encoders: Pre-fitted encoders keyed by column name.  When
            ``None``, new encoders are fitted on the data.
        min_category_freq: Minimum count for a value to be kept as its
            own category during training.  Values below this threshold
            are replaced with ``"__unknown__"``.
        unknown_mask_rate: Fraction of *known*-category rows to randomly
            mask to ``"__unknown__"`` during training (for robustness).
        random_state: Seed for the random masking (reproducibility).
        schema_version: ``"v1"`` or ``"v2"``.  Selects which categorical
            columns to encode.

    Returns:
        ``(encoded_df, cat_encoders)`` -- the DataFrame with categorical
        columns replaced by integer codes, and the encoder dict.
    """
    cat_cols = get_categorical_columns(schema_version)
    df = df.copy()
    if cat_encoders is None:
        rng = np.random.RandomState(random_state)
        cat_encoders = {}
        for col in cat_cols:
            vals = df[col].astype(str)
            freq = vals.value_counts()
            rare_mask = vals.isin(freq[freq < min_category_freq].index)
            vals = vals.copy()
            vals[rare_mask] = _UNKNOWN_TOKEN
            if unknown_mask_rate > 0:
                known_mask = vals != _UNKNOWN_TOKEN
                n_known = known_mask.sum()
                n_mask = int(round(n_known * unknown_mask_rate))
                if n_mask > 0:
                    mask_idx = rng.choice(
                        vals.index[known_mask], size=n_mask, replace=False
                    )
                    vals.iloc[vals.index.get_indexer(pd.Index(mask_idx))] = (
                        _UNKNOWN_TOKEN
                    )
            le = LabelEncoder()
            df[col] = le.fit_transform(vals)
            cat_encoders[col] = le
    else:
        for col in cat_cols:
            le = cat_encoders[col]
            known = set(le.classes_)
            has_unknown = _UNKNOWN_TOKEN in known

            def _encode(
                v: str, _k: set = known, _le: LabelEncoder = le, _hu: bool = has_unknown
            ) -> int:
                if v in _k:
                    return int(_le.transform([v])[0])
                if _hu:
                    return int(_le.transform([_UNKNOWN_TOKEN])[0])
                return -1

            df[col] = df[col].astype(str).apply(_encode)
    return df, cat_encoders


def prepare_xy(
    df: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    *,
    min_category_freq: int = 5,
    unknown_mask_rate: float = 0.05,
    random_state: int | None = None,
    schema_version: str = "v1",
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[str, LabelEncoder]]:
    """Extract feature matrix and encoded label vector from *df*.

    Categorical columns are label-encoded to integers so LightGBM can
    use them as native categoricals.  Missing numeric values are filled
    with 0.

    Args:
        df: Labeled feature DataFrame (must contain the feature columns
            for the selected *schema_version* and a ``label`` column).
        label_encoder: Pre-fitted encoder to reuse (e.g. the one returned
            from the training call).  If ``None``, a new encoder is fitted
            on the canonical ``LABEL_SET_V1``.
        cat_encoders: Pre-fitted categorical encoders.  If ``None``, new
            ones are fitted from *df*.
        min_category_freq: Forwarded to :func:`encode_categoricals`.
        unknown_mask_rate: Forwarded to :func:`encode_categoricals`.
        random_state: Forwarded to :func:`encode_categoricals`.
        schema_version: ``"v1"`` or ``"v2"``.

    Returns:
        A ``(X, y, label_encoder, cat_encoders)`` tuple.
    """
    feat_cols = get_feature_columns(schema_version)
    feat_df = df[feat_cols].copy()
    feat_df, cat_encoders = encode_categoricals(
        feat_df,
        cat_encoders,
        min_category_freq=min_category_freq,
        unknown_mask_rate=unknown_mask_rate,
        random_state=random_state,
        schema_version=schema_version,
    )
    x = feat_df.fillna(0).to_numpy(dtype=np.float64)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(sorted(LABEL_SET_V1))

    y = label_encoder.transform(df["label"].values)
    return x, y, label_encoder, cat_encoders


def _categorical_feature_indices(schema_version: str = "v1") -> list[int]:
    """Return the positional indices of categorical columns in feature columns."""
    feat_cols = get_feature_columns(schema_version)
    cat_cols = get_categorical_columns(schema_version)
    return [feat_cols.index(c) for c in cat_cols]


def compute_sample_weights(
    y: np.ndarray,
    method: Literal["balanced", "none"] = "balanced",
) -> np.ndarray | None:
    """Map encoded labels to per-sample weights using inverse class frequency.

    Args:
        y: Integer-encoded label array (output of ``LabelEncoder.transform``).
        method: ``"balanced"`` computes ``n_samples / (n_classes * count_per_class)``
            and maps each sample to its class weight.  ``"none"`` returns ``None``.

    Returns:
        Per-sample weight array with the same length as *y*, or ``None``
        when *method* is ``"none"``.
    """
    if method == "none":
        return None
    n_samples = len(y)
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = n_samples / (n_classes * counts)
    return class_weights[y]


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    extra_params: dict[str, Any] | None = None,
    class_weight: Literal["balanced", "none"] = "balanced",
    min_category_freq: int = 5,
    unknown_mask_rate: float = 0.05,
    random_state: int | None = None,
    schema_version: str = "v1",
) -> tuple[lgb.Booster, dict, pd.DataFrame, dict[str, Any], dict[str, LabelEncoder]]:
    """Train a LightGBM multiclass model and evaluate on the val set.

    Args:
        train_df: Training DataFrame with feature columns and a ``label``
            column.
        val_df: Validation DataFrame (same schema as *train_df*).
        num_boost_round: Number of boosting iterations.
        extra_params: Additional LightGBM parameters merged on top of the
            built-in defaults.
        class_weight: Strategy for handling class imbalance.
            ``"balanced"`` uses inverse-frequency sample weights;
            ``"none"`` disables weighting.
        min_category_freq: Minimum count for a category to keep its own
            code; rarer values become ``__unknown__``.
        unknown_mask_rate: Fraction of known-category rows randomly
            masked to ``__unknown__`` during training.
        random_state: Seed for the random unknown masking.
        schema_version: ``"v1"`` or ``"v2"``.

    Returns:
        A ``(model, metrics, confusion_df, params, cat_encoders)`` tuple
        where *cat_encoders* maps each categorical column name to its
        fitted ``LabelEncoder``.
    """
    feat_cols = get_feature_columns(schema_version)
    x_train, y_train, le, cat_encoders = prepare_xy(
        train_df,
        min_category_freq=min_category_freq,
        unknown_mask_rate=unknown_mask_rate,
        random_state=random_state,
        schema_version=schema_version,
    )
    x_val, y_val, _, _ = prepare_xy(
        val_df,
        label_encoder=le,
        cat_encoders=cat_encoders,
        schema_version=schema_version,
    )

    params = {**_DEFAULT_PARAMS, "num_class": len(le.classes_)}
    if extra_params:
        params.update(extra_params)

    cat_indices = _categorical_feature_indices(schema_version)
    sample_weights = compute_sample_weights(y_train, method=class_weight)

    train_ds = lgb.Dataset(
        x_train,
        label=y_train,
        weight=sample_weights,
        feature_name=feat_cols,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )
    val_ds = lgb.Dataset(x_val, label=y_val, reference=train_ds, free_raw_data=False)

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[val_ds],
        valid_names=["val"],
    )

    y_pred_idx = model.predict(x_val).argmax(axis=1)  # type: ignore[union-attr]
    y_pred_labels = le.inverse_transform(y_pred_idx)
    y_true_labels = le.inverse_transform(y_val)

    label_names = list(le.classes_)
    metrics = compute_metrics(y_true_labels, y_pred_labels, label_names)
    cm_df = confusion_matrix_df(y_true_labels, y_pred_labels, label_names)

    params["class_weight_method"] = class_weight
    params["unknown_category_freq_threshold"] = min_category_freq
    params["unknown_category_mask_rate"] = unknown_mask_rate
    return model, metrics, cm_df, params, cat_encoders
