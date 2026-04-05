"""Evaluation metrics: macro-F1, confusion matrices, calibration, and per-user helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

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
        f1_score(
            y_true, y_pred, labels=labels_list, average="weighted", zero_division=0
        )
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
    *,
    include_support: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Per-class precision, recall, F1, and optionally support (true-class counts).

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Ordered label vocabulary.
        include_support: When ``True``, each value includes ``support`` (int).

    Returns:
        Dict mapping each label to precision, recall, f1, and optionally support.
    """
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(label_names),
        zero_division=0,
    )
    out: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(label_names):
        row: dict[str, float | int] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
        }
        if include_support:
            row["support"] = int(sup[i])
        out[name] = row
    return out


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
            f1_score(
                y_true, y_pred, labels=all_labels, average="macro", zero_division=0
            )
        )
        weighted_f1 = float(
            f1_score(
                y_true, y_pred, labels=all_labels, average="weighted", zero_division=0
            )
        )
        results[name] = {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "reject_rate": round(reject_rate(list(y_pred), reject_label), 4),
            "per_class": per_class_metrics(y_true, y_pred, all_labels),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=all_labels,
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
            f1_score(
                true_list,
                pred_list,
                labels=labels_list,
                average="macro",
                zero_division=0,
            )
        )
        results[uid] = {
            "macro_f1": round(mf1, 4),
            "count": len(true_list),
            **{
                f"{lbl}_f1": round(float(v), 4)
                for lbl, v in zip(
                    labels_list,
                    precision_recall_fscore_support(
                        true_list,
                        pred_list,
                        labels=labels_list,
                        zero_division=0,
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
            binary_true,
            proba_class,
            n_bins=n_bins,
            strategy="uniform",
        )
        result[name] = {
            "fraction_of_positives": [round(float(v), 6) for v in frac_pos],
            "mean_predicted_value": [round(float(v), 6) for v in mean_pred],
        }
    return result


# ---------------------------------------------------------------------------
# High-value inspection metrics (calibration, confusion structure, slices)
# ---------------------------------------------------------------------------

_UNKNOWN_TOKEN = "__unknown__"


def top_confusion_pairs(
    cm: list[list[int]] | np.ndarray,
    label_names: Sequence[str],
    *,
    k: int = 20,
) -> list[dict[str, str | int]]:
    """Rank largest off-diagonal confusion counts (true_class -> pred_class).

    Args:
        cm: Square confusion matrix (rows true, columns predicted).
        label_names: Label order for rows/columns.
        k: Maximum number of pairs to return.

    Returns:
        List of dicts with ``true_label``, ``pred_label``, ``count``, sorted
        by count descending (off-diagonal only).
    """
    mat = np.asarray(cm, dtype=np.int64)
    n = len(label_names)
    if mat.shape != (n, n):
        raise ValueError(
            f"confusion matrix shape {mat.shape} does not match {n} labels"
        )
    pairs: list[tuple[int, str, str]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(mat[i, j])
            if c > 0:
                pairs.append((c, label_names[i], label_names[j]))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [{"true_label": t[1], "pred_label": t[2], "count": t[0]} for t in pairs[:k]]


def _binary_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for binary labels (uniform probability bins)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(y_prob)
    y_true_s = y_true[order]
    y_prob_s = y_prob[order]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (y_prob_s >= lo) & (y_prob_s <= hi)
        else:
            mask = (y_prob_s >= lo) & (y_prob_s < hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(y_prob_s[mask]))
        bin_acc = float(np.mean(y_true_s[mask]))
        ece += abs(bin_acc - bin_conf) * (float(np.sum(mask)) / n)
    return ece


def expected_calibration_error_multiclass(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
    label_names: Sequence[str],
    *,
    n_bins: int = 10,
) -> float:
    """Weighted mean of one-vs-rest binary ECE across classes with support."""
    y_true_indices = np.asarray(y_true_indices, dtype=np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    n_classes = len(label_names)
    if len(y_true_indices) == 0 or y_proba.size == 0:
        return 0.0
    eces: list[float] = []
    weights: list[float] = []
    for i in range(n_classes):
        binary = (y_true_indices == i).astype(np.float64)
        if binary.sum() == 0:
            continue
        proba_i = y_proba[:, i]
        eces.append(_binary_ece(binary, proba_i, n_bins=n_bins))
        weights.append(float(binary.sum()))
    if not eces:
        return 0.0
    return float(np.average(np.asarray(eces), weights=np.asarray(weights)))


def multiclass_brier_score(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """Mean squared error between one-hot true labels and predicted probabilities."""
    y_true_indices = np.asarray(y_true_indices, dtype=np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    n, k = y_proba.shape
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), y_true_indices] = 1.0
    return float(np.mean(np.sum((one_hot - y_proba) ** 2, axis=1)))


def multiclass_log_loss_score(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
    *,
    eps: float = 1e-15,
) -> float:
    """Multiclass log loss with clipped probabilities."""
    y_true_indices = np.asarray(y_true_indices, dtype=np.int64)
    y_proba = np.clip(np.asarray(y_proba, dtype=np.float64), eps, 1.0 - eps)
    n_classes = y_proba.shape[1]
    if len(y_true_indices) == 0:
        return 0.0
    return float(
        log_loss(
            y_true_indices,
            y_proba,
            labels=list(range(n_classes)),
        )
    )


DEFAULT_SLICE_COLUMNS: tuple[str, ...] = (
    "user_id",
    "app_id",
    "app_category",
    "domain_category",
    "hour_of_day",
)


def slice_metrics_by_columns(
    df: pd.DataFrame,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
    slice_columns: Sequence[str] | None = None,
    *,
    max_groups_per_column: int = 100,
    reject_label: str = MIXED_UNKNOWN,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-slice macro/weighted F1, reject rate, and row counts.

    For each column, groups are sorted by frequency and truncated to
    *max_groups_per_column* to keep output bounded when cardinality is high.

    Args:
        df: Feature rows aligned with *y_true* / *y_pred*.
        y_true: Ground-truth labels.
        y_pred: Predicted labels (after reject/smoothing if applicable).
        label_names: Core label vocabulary for sklearn metrics.
        slice_columns: Columns to slice by; defaults to
            :data:`DEFAULT_SLICE_COLUMNS` intersected with ``df.columns``.
        max_groups_per_column: Max distinct slice values per column.
        reject_label: Label counted as rejected for per-slice reject_rate.

    Returns:
        Nested dict ``{column: {slice_value_str: metrics_dict}}``.
    """
    if slice_columns is None:
        slice_columns = DEFAULT_SLICE_COLUMNS
    y_true_l = list(y_true)
    y_pred_l = list(y_pred)
    n = len(y_true_l)
    if n == 0 or len(y_pred_l) != n:
        return {}
    labels_list = list(label_names)
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for col in slice_columns:
        if col not in df.columns:
            continue
        series = df[col]
        # stringify for JSON keys (handles int hour_of_day, etc.)
        keys = series.astype(str).tolist()
        counts = Counter(keys)
        top_keys = [k for k, _ in counts.most_common(max_groups_per_column)]
        col_out: dict[str, dict[str, Any]] = {}
        for key in top_keys:
            idx = [i for i in range(n) if keys[i] == key]
            if not idx:
                continue
            yt = [y_true_l[i] for i in idx]
            yp = [y_pred_l[i] for i in idx]
            mf1 = float(
                f1_score(yt, yp, labels=labels_list, average="macro", zero_division=0)
            )
            wf1 = float(
                f1_score(
                    yt, yp, labels=labels_list, average="weighted", zero_division=0
                )
            )
            rr = reject_rate(yp, reject_label)
            col_out[key] = {
                "row_count": len(idx),
                "macro_f1": round(mf1, 4),
                "weighted_f1": round(wf1, 4),
                "reject_rate": round(rr, 4),
                "per_class": per_class_metrics(yt, yp, labels_list),
            }
        if col_out:
            out[col] = col_out
    return out


def unknown_category_rates(
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None,
    categorical_columns: Sequence[str],
) -> dict[str, Any]:
    """Fraction of rows where a categorical maps to unknown or legacy -1 encoding.

    Mirrors inference-time behavior in :func:`taskclf.train.lgbm.encode_categoricals`:
    values not in the fitted encoder vocabulary map to ``__unknown__`` when
    present in the vocabulary, else ``-1``.

    Args:
        df: Feature rows (same rows as evaluation).
        cat_encoders: Fitted encoders from the training bundle (may be empty).
        categorical_columns: Categorical column names for this schema (e.g. from
            ``get_categorical_columns``).

    Returns:
        Dict with ``per_column`` rates, ``overall_rate`` (mean of per-column
        rates over columns present), and ``columns_evaluated``.
    """
    if not cat_encoders or not categorical_columns:
        return {
            "per_column": {},
            "overall_rate": None,
            "columns_evaluated": [],
            "note": "no categorical encoders or columns",
        }
    per_column: dict[str, float] = {}
    evaluated: list[str] = []
    n = len(df)
    if n == 0:
        return {
            "per_column": {},
            "overall_rate": None,
            "columns_evaluated": [],
            "note": "empty dataframe",
        }

    for col in categorical_columns:
        if col not in df.columns or col not in cat_encoders:
            continue
        le = cat_encoders[col]
        known = set(le.classes_)
        n_unknown = 0
        for v in df[col].astype(str):
            if str(v) not in known:
                n_unknown += 1
        rate = n_unknown / n if n else 0.0
        per_column[col] = round(float(rate), 4)
        evaluated.append(col)

    overall = float(np.mean(list(per_column.values()))) if per_column else None
    return {
        "per_column": per_column,
        "overall_rate": overall,
        "columns_evaluated": evaluated,
    }


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
        per_user[uid] = {
            "count": count,
            "fraction": fraction,
            "label_distribution": dist,
        }
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
