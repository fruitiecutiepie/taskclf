"""Evaluation metrics: macro-F1, confusion matrices, and related helpers."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from taskclf.core.defaults import MIXED_UNKNOWN


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


def reject_rate(
    labels: Sequence[str],
    reject_label: str = MIXED_UNKNOWN,
) -> float:
    """Fraction of *labels* that equal *reject_label*.

    Args:
        labels: Predicted label strings.
        reject_label: The label treated as a reject / unknown.

    Returns:
        A float in ``[0, 1]``.  Returns 0.0 for an empty sequence.
    """
    if not labels:
        return 0.0
    return sum(1 for lbl in labels if lbl == reject_label) / len(labels)


def per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, and F1.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Ordered label vocabulary.

    Returns:
        Dict mapping each label to
        ``{"precision": float, "recall": float, "f1": float}``.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(label_names), zero_division=0,
    )
    return {
        name: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
        }
        for i, name in enumerate(label_names)
    }


def compare_baselines(
    y_true: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    label_names: Sequence[str],
    reject_label: str = MIXED_UNKNOWN,
) -> dict[str, dict]:
    """Compare multiple prediction methods against the same ground truth.

    Args:
        y_true: Ground-truth label strings.
        predictions: Mapping of ``{method_name: predicted_labels}``.
        label_names: Ordered core label vocabulary.
        reject_label: The label treated as a reject.

    Returns:
        Dict keyed by method name, each containing ``macro_f1``,
        ``weighted_f1``, ``reject_rate``, ``per_class``, and
        ``confusion_matrix``.
    """
    results: dict[str, dict] = {}
    all_labels = list(label_names) + (
        [reject_label] if reject_label not in label_names else []
    )

    for name, y_pred in predictions.items():
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)
        )
        results[name] = {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "reject_rate": round(reject_rate(list(y_pred), reject_label), 4),
            "per_class": per_class_metrics(y_true, y_pred, all_labels),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=all_labels,
            ).tolist(),
            "label_names": all_labels,
        }

    return results
