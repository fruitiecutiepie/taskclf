"""Evaluation metrics: macro-F1, confusion matrices, and related helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> dict:
    """Return macro-F1 and a nested confusion matrix.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted label strings.
    label_names:
        Ordered label vocabulary (defines row/column order of the matrix).

    Returns
    -------
    dict with keys ``macro_f1`` (float) and ``confusion_matrix`` (list of lists).
    """
    macro_f1: float = float(
        f1_score(y_true, y_pred, labels=list(label_names), average="macro", zero_division=0)
    )
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=list(label_names))
    return {
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "label_names": list(label_names),
    }


def confusion_matrix_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Build a labelled confusion-matrix DataFrame suitable for CSV export."""
    cm = confusion_matrix(y_true, y_pred, labels=list(label_names))
    return pd.DataFrame(cm, index=list(label_names), columns=list(label_names))
