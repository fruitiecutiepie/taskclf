"""Evaluation metrics: macro-F1, confusion matrices, and related helpers."""

from __future__ import annotations

from collections import Counter
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

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Ordered label vocabulary (defines row/column order
            of the matrix).

    Returns:
        Dict with keys ``macro_f1`` (float), ``confusion_matrix``
        (list of lists), and ``label_names`` (list of str).
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


def class_distribution(
    y_true: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Per-class counts and fractions for imbalance reporting.

    Args:
        y_true: Ground-truth label strings.
        label_names: Full label vocabulary (defines which classes appear
            in the output, even if absent from *y_true*).

    Returns:
        Dict mapping each label to ``{"count": int, "fraction": float}``.
        Fractions sum to 1.0 (within rounding tolerance).  If *y_true* is
        empty, all fractions are 0.0.
    """
    counts = Counter(y_true)
    total = len(y_true)
    return {
        label: {
            "count": counts.get(label, 0),
            "fraction": round(counts.get(label, 0) / total, 4) if total > 0 else 0.0,
        }
        for label in label_names
    }


def confusion_matrix_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Build a labelled confusion-matrix DataFrame suitable for CSV export.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Ordered label vocabulary (used as both row and column
            index of the resulting DataFrame).

    Returns:
        Square DataFrame with *label_names* as row and column labels.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(label_names))
    return pd.DataFrame(cm, index=list(label_names), columns=list(label_names))
