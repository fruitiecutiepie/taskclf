"""Evaluation metrics: macro-F1, confusion matrices, calibration, and per-user helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from taskclf.core.defaults import MIXED_UNKNOWN


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> dict:
    """Return macro-F1, weighted-F1, and a nested confusion matrix.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Ordered label vocabulary (defines row/column order
            of the matrix).

    Returns:
        Dict with keys ``macro_f1``, ``weighted_f1`` (floats),
        ``confusion_matrix`` (list of lists), and ``label_names``
        (list of str).
    """
    labels_list = list(label_names)
    macro_f1: float = float(
        f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )
    weighted_f1: float = float(
        f1_score(y_true, y_pred, labels=labels_list, average="weighted", zero_division=0)
    )
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=labels_list)
    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "label_names": labels_list,
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


# ---------------------------------------------------------------------------
# Per-user metrics
# ---------------------------------------------------------------------------


def per_user_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    user_ids: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute macro-F1 and per-class F1 grouped by user.

    Args:
        y_true: Ground-truth label strings (one per window).
        y_pred: Predicted label strings (same length as *y_true*).
        user_ids: User identifier per window (same length as *y_true*).
        label_names: Ordered label vocabulary.

    Returns:
        Dict keyed by user_id, each containing ``macro_f1`` and a nested
        ``per_class`` dict of precision / recall / F1 per label.
    """
    groups: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for uid, yt, yp in zip(user_ids, y_true, y_pred):
        groups[uid][0].append(yt)
        groups[uid][1].append(yp)

    results: dict[str, dict[str, float]] = {}
    labels_list = list(label_names)
    for uid, (true_list, pred_list) in sorted(groups.items()):
        mf1 = float(
            f1_score(true_list, pred_list, labels=labels_list, average="macro", zero_division=0)
        )
        results[uid] = {
            "macro_f1": round(mf1, 4),
            "count": len(true_list),
            **{
                f"{lbl}_f1": round(float(v), 4)
                for lbl, v in zip(
                    labels_list,
                    precision_recall_fscore_support(
                        true_list, pred_list, labels=labels_list, zero_division=0,
                    )[2],
                )
            },
        }
    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_curve_data(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
    label_names: Sequence[str],
    *,
    n_bins: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """Per-class calibration curve data for reliability diagrams.

    Uses one-vs-rest binarization so each class gets its own curve.

    Args:
        y_true_indices: Integer-encoded true labels (shape ``(n,)``).
        y_proba: Predicted probability matrix (shape ``(n, n_classes)``).
        label_names: Ordered label vocabulary matching columns of *y_proba*.
        n_bins: Number of probability bins.

    Returns:
        Dict keyed by label name, each containing ``fraction_of_positives``
        and ``mean_predicted_value`` lists suitable for plotting.
    """
    result: dict[str, dict[str, list[float]]] = {}
    for i, name in enumerate(label_names):
        binary_true = (y_true_indices == i).astype(int)
        proba_class = y_proba[:, i]
        if binary_true.sum() == 0:
            result[name] = {"fraction_of_positives": [], "mean_predicted_value": []}
            continue
        frac_pos, mean_pred = sk_calibration_curve(
            binary_true, proba_class, n_bins=n_bins, strategy="uniform",
        )
        result[name] = {
            "fraction_of_positives": [round(float(v), 6) for v in frac_pos],
            "mean_predicted_value": [round(float(v), 6) for v in mean_pred],
        }
    return result


# ---------------------------------------------------------------------------
# User stratification
# ---------------------------------------------------------------------------


def user_stratification_report(
    user_ids: Sequence[str],
    labels: Sequence[str],
    label_names: Sequence[str],
    *,
    dominance_threshold: float = 0.5,
) -> dict:
    """Analyse per-user contribution to the training set and flag imbalance.

    Args:
        user_ids: User identifier per row.
        labels: Label per row.
        label_names: Ordered label vocabulary.
        dominance_threshold: Fraction above which a single user is
            considered dominant and a warning is emitted.

    Returns:
        Dict with ``per_user`` (row count, fraction, label distribution),
        ``total_rows``, and ``warnings`` (list of human-readable strings
        for any user exceeding *dominance_threshold*).
    """
    total = len(user_ids)
    user_counts: Counter[str] = Counter(user_ids)
    user_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for uid, lbl in zip(user_ids, labels):
        user_labels[uid][lbl] += 1

    per_user: dict[str, dict] = {}
    warnings: list[str] = []

    for uid in sorted(user_counts):
        count = user_counts[uid]
        fraction = round(count / total, 4) if total > 0 else 0.0
        dist = {lbl: user_labels[uid].get(lbl, 0) for lbl in label_names}
        per_user[uid] = {"count": count, "fraction": fraction, "label_distribution": dist}
        if fraction > dominance_threshold:
            warnings.append(
                f"User {uid!r} contributes {fraction:.0%} of rows "
                f"({count}/{total}), exceeding threshold {dominance_threshold:.0%}"
            )

    return {
        "per_user": per_user,
        "total_rows": total,
        "user_count": len(user_counts),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Per-user/day reject rate (drift detection)
# ---------------------------------------------------------------------------


def reject_rate_by_group(
    labels: Sequence[str],
    user_ids: Sequence[str],
    timestamps: Sequence,
    *,
    reject_label: str = MIXED_UNKNOWN,
    spike_multiplier: float = 2.0,
) -> dict:
    """Compute reject rate grouped by ``(user_id, date)`` for drift detection.

    A group is flagged as a drift signal when its reject rate exceeds
    *spike_multiplier* times the global reject rate.

    Args:
        labels: Predicted label strings (may include *reject_label*).
        user_ids: User identifier per window.
        timestamps: Timestamp per window (anything parseable by
            ``pd.Timestamp``; only the date portion is used).
        reject_label: The label treated as a reject / unknown.
        spike_multiplier: A group's reject rate must exceed
            ``global_reject_rate * spike_multiplier`` to be flagged.

    Returns:
        Dict with ``global_reject_rate``, ``per_group`` (keyed by
        ``"user_id|YYYY-MM-DD"`` with ``reject_rate``, ``total``, and
        ``rejected``), and ``drift_flags`` (list of flagged group keys).
    """
    global_rr = reject_rate(labels, reject_label)

    groups: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for lbl, uid, ts in zip(labels, user_ids, timestamps):
        date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        key = f"{uid}|{date_str}"
        total, rejected = groups[key]
        groups[key] = (total + 1, rejected + (1 if lbl == reject_label else 0))

    per_group: dict[str, dict[str, float | int]] = {}
    drift_flags: list[str] = []
    spike_threshold = global_rr * spike_multiplier

    for key, (total, rejected) in sorted(groups.items()):
        grp_rr = rejected / total if total > 0 else 0.0
        per_group[key] = {
            "reject_rate": round(grp_rr, 4),
            "total": total,
            "rejected": rejected,
        }
        if grp_rr > spike_threshold and total > 0:
            drift_flags.append(key)

    return {
        "global_reject_rate": round(global_rr, 4),
        "per_group": per_group,
        "drift_flags": drift_flags,
    }
