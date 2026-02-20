"""LightGBM multiclass baseline trainer with evaluation."""

from __future__ import annotations

from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from taskclf.core.metrics import compute_metrics, confusion_matrix_df
from taskclf.core.types import LABEL_SET_V1

FEATURE_COLUMNS: Final[list[str]] = [
    "is_browser",
    "is_editor",
    "is_terminal",
    "app_switch_count_last_5m",
    "keys_per_min",
    "backspace_ratio",
    "shortcut_rate",
    "clicks_per_min",
    "scroll_events_per_min",
    "mouse_distance",
    "hour_of_day",
    "day_of_week",
    "session_length_so_far",
]

_DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "verbose": -1,
}


def prepare_xy(
    df: pd.DataFrame,
    label_encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extract feature matrix and encoded label vector from *df*.

    Missing values are filled with 0.  If *label_encoder* is ``None`` a
    new one is fitted on the sorted ``LABEL_SET_V1`` vocabulary so that
    class indices are stable across train/val.

    Args:
        df: Labeled feature DataFrame (must contain ``FEATURE_COLUMNS``
            and a ``label`` column).
        label_encoder: Pre-fitted encoder to reuse (e.g. the one returned
            from the training call).  If ``None``, a new encoder is fitted
            on the canonical ``LABEL_SET_V1``.

    Returns:
        A ``(X, y, label_encoder)`` tuple where *X* is the feature matrix,
        *y* is the integer-encoded label vector, and *label_encoder* is the
        encoder used (either the one passed in or a newly fitted one).
    """
    x = df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=np.float64)

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(sorted(LABEL_SET_V1))

    y = label_encoder.transform(df["label"].values)
    return x, y, label_encoder


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    num_boost_round: int = 100,
    extra_params: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, dict, pd.DataFrame, dict[str, Any]]:
    """Train a LightGBM multiclass model and evaluate on the val set.

    Args:
        train_df: Training DataFrame with feature columns and a ``label``
            column.
        val_df: Validation DataFrame (same schema as *train_df*).
        num_boost_round: Number of boosting iterations.
        extra_params: Additional LightGBM parameters merged on top of the
            built-in defaults.

    Returns:
        A ``(model, metrics, confusion_df, params)`` tuple where *model*
        is the trained ``lgb.Booster``, *metrics* is the evaluation dict,
        *confusion_df* is the labelled confusion matrix, and *params* is
        the full LightGBM parameter dict that was used.
    """
    x_train, y_train, le = prepare_xy(train_df)
    x_val, y_val, _ = prepare_xy(val_df, label_encoder=le)

    params = {**_DEFAULT_PARAMS, "num_class": len(le.classes_)}
    if extra_params:
        params.update(extra_params)

    train_ds = lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_COLUMNS, free_raw_data=False)
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

    return model, metrics, cm_df, params
