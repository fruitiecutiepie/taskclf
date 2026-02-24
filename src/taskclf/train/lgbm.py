"""LightGBM multiclass baseline trainer with evaluation."""

from __future__ import annotations

from typing import Any, Final

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
    "hour_of_day",
    "day_of_week",
    "session_length_so_far",
]

CATEGORICAL_COLUMNS: Final[list[str]] = ["app_id", "app_category"]

_DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "verbose": -1,
}


def encode_categoricals(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns in-place and return fitted encoders.

    Unknown values at inference time are mapped to a reserved ``-1``
    code so the model degrades gracefully on unseen apps.

    Args:
        df: DataFrame with the columns listed in ``CATEGORICAL_COLUMNS``.
        cat_encoders: Pre-fitted encoders keyed by column name.  When
            ``None``, new encoders are fitted on the data.

    Returns:
        ``(encoded_df, cat_encoders)`` -- the DataFrame with categorical
        columns replaced by integer codes, and the encoder dict.
    """
    df = df.copy()
    if cat_encoders is None:
        cat_encoders = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            cat_encoders[col] = le
    else:
        for col in CATEGORICAL_COLUMNS:
            le = cat_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v, _k=known, _le=le: (
                    int(_le.transform([v])[0]) if v in _k else -1
                )
            )
    return df, cat_encoders


def prepare_xy(
    df: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[str, LabelEncoder]]:
    """Extract feature matrix and encoded label vector from *df*.

    Categorical columns are label-encoded to integers so LightGBM can
    use them as native categoricals.  Missing numeric values are filled
    with 0.

    Args:
        df: Labeled feature DataFrame (must contain ``FEATURE_COLUMNS``
            and a ``label`` column).
        label_encoder: Pre-fitted encoder to reuse (e.g. the one returned
            from the training call).  If ``None``, a new encoder is fitted
            on the canonical ``LABEL_SET_V1``.
        cat_encoders: Pre-fitted categorical encoders.  If ``None``, new
            ones are fitted from *df*.

    Returns:
        A ``(X, y, label_encoder, cat_encoders)`` tuple.
    """
    feat_df = df[FEATURE_COLUMNS].copy()
    feat_df, cat_encoders = encode_categoricals(feat_df, cat_encoders)
    x = feat_df.fillna(0).to_numpy(dtype=np.float64)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(sorted(LABEL_SET_V1))

    y = label_encoder.transform(df["label"].values)
    return x, y, label_encoder, cat_encoders


def _categorical_feature_indices() -> list[int]:
    """Return the positional indices of categorical columns in FEATURE_COLUMNS."""
    return [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    extra_params: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, dict, pd.DataFrame, dict[str, Any], dict[str, LabelEncoder]]:
    """Train a LightGBM multiclass model and evaluate on the val set.

    Args:
        train_df: Training DataFrame with feature columns and a ``label``
            column.
        val_df: Validation DataFrame (same schema as *train_df*).
        num_boost_round: Number of boosting iterations.
        extra_params: Additional LightGBM parameters merged on top of the
            built-in defaults.

    Returns:
        A ``(model, metrics, confusion_df, params, cat_encoders)`` tuple
        where *cat_encoders* maps each categorical column name to its
        fitted ``LabelEncoder``.
    """
    x_train, y_train, le, cat_encoders = prepare_xy(train_df)
    x_val, y_val, _, _ = prepare_xy(val_df, label_encoder=le, cat_encoders=cat_encoders)

    params = {**_DEFAULT_PARAMS, "num_class": len(le.classes_)}
    if extra_params:
        params.update(extra_params)

    cat_indices = _categorical_feature_indices()

    train_ds = lgb.Dataset(
        x_train, label=y_train, feature_name=FEATURE_COLUMNS,
        categorical_feature=cat_indices, free_raw_data=False,
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

    return model, metrics, cm_df, params, cat_encoders
