"""Full model evaluation pipeline: metrics, calibration, acceptance checks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_SMOOTH_WINDOW,
    MIXED_UNKNOWN,
)
from taskclf.core.metrics import (
    calibration_curve_data,
    compute_metrics,
    confusion_matrix_df,
    expected_calibration_error_multiclass,
    multiclass_brier_score,
    multiclass_log_loss_score,
    per_class_metrics,
    per_user_metrics,
    reject_rate,
    slice_metrics_by_columns,
    top_confusion_pairs,
    unknown_category_rates,
    user_stratification_report,
)
from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.batch import predict_proba
from taskclf.train.lgbm import get_categorical_columns
from taskclf.infer.calibration import Calibrator
from taskclf.infer.smooth import flap_rate, rolling_majority, segmentize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceptance thresholds (from docs/guide/acceptance.md)
# ---------------------------------------------------------------------------

_ACCEPT_MACRO_F1: float = 0.65
_ACCEPT_WEIGHTED_F1: float = 0.70
_ACCEPT_BREAKIDLE_PRECISION: float = 0.95
_ACCEPT_BREAKIDLE_RECALL: float = 0.90
_ACCEPT_MIN_CLASS_PRECISION: float = 0.50
_ACCEPT_SEEN_MACRO_F1: float = 0.70
_ACCEPT_UNSEEN_MACRO_F1: float = 0.60
_ACCEPT_REJECT_RATE_MIN: float = 0.05
_ACCEPT_REJECT_RATE_MAX: float = 0.30


class EvaluationReport(BaseModel, frozen=True):
    """Comprehensive evaluation output for a trained model on a test set."""

    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    label_names: list[str]
    per_user: dict[str, dict[str, float]]
    calibration: dict[str, dict[str, list[float]]]
    stratification: dict[str, Any]
    seen_user_f1: float | None = None
    unseen_user_f1: float | None = None
    reject_rate: float
    acceptance_checks: dict[str, bool]
    acceptance_details: dict[str, str]
    flip_rate: float | None = None
    segment_duration_distribution: dict[str, int] | None = None
    eval_mode: str = "raw"
    top_confusion_pairs: list[dict[str, str | int]] = Field(default_factory=list)
    expected_calibration_error: float = 0.0
    multiclass_brier_score: float = 0.0
    multiclass_log_loss: float = 0.0
    slice_metrics: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    unknown_category_rates: dict[str, Any] = Field(default_factory=dict)


class RejectTuningResult(BaseModel, frozen=True):
    """Result of sweeping reject thresholds on a validation set.

    Attributes:
        best_threshold: Threshold that maximises accuracy on accepted
            windows while keeping reject rate within acceptance bounds.
        sweep: List of dicts, one per candidate threshold, each with
            ``threshold``, ``accuracy_on_accepted``, ``reject_rate``,
            ``coverage``, and ``macro_f1``.
    """

    best_threshold: float
    sweep: list[dict[str, float]]


def _check_acceptance(
    macro_f1: float,
    weighted_f1: float,
    per_class: dict[str, dict[str, float]],
    rr: float,
    seen_f1: float | None,
    unseen_f1: float | None,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Run acceptance gates and return pass/fail with human-readable details."""
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    checks["macro_f1"] = macro_f1 >= _ACCEPT_MACRO_F1
    details["macro_f1"] = (
        f"{macro_f1:.4f} >= {_ACCEPT_MACRO_F1} -> {'PASS' if checks['macro_f1'] else 'FAIL'}"
    )

    checks["weighted_f1"] = weighted_f1 >= _ACCEPT_WEIGHTED_F1
    details["weighted_f1"] = (
        f"{weighted_f1:.4f} >= {_ACCEPT_WEIGHTED_F1} -> {'PASS' if checks['weighted_f1'] else 'FAIL'}"
    )

    bi = per_class.get("BreakIdle", {})
    bi_prec = bi.get("precision", 0.0)
    bi_rec = bi.get("recall", 0.0)
    checks["breakidle_precision"] = bi_prec >= _ACCEPT_BREAKIDLE_PRECISION
    details["breakidle_precision"] = (
        f"{bi_prec:.4f} >= {_ACCEPT_BREAKIDLE_PRECISION} -> "
        f"{'PASS' if checks['breakidle_precision'] else 'FAIL'}"
    )
    checks["breakidle_recall"] = bi_rec >= _ACCEPT_BREAKIDLE_RECALL
    details["breakidle_recall"] = (
        f"{bi_rec:.4f} >= {_ACCEPT_BREAKIDLE_RECALL} -> "
        f"{'PASS' if checks['breakidle_recall'] else 'FAIL'}"
    )

    low_prec = [
        (lbl, m["precision"])
        for lbl, m in per_class.items()
        if m["precision"] < _ACCEPT_MIN_CLASS_PRECISION
    ]
    checks["no_class_below_50_precision"] = len(low_prec) == 0
    if low_prec:
        details["no_class_below_50_precision"] = "FAIL: " + ", ".join(
            f"{lbl}={p:.4f}" for lbl, p in low_prec
        )
    else:
        details["no_class_below_50_precision"] = "PASS (all classes >= 0.50)"

    checks["reject_rate_bounds"] = (
        _ACCEPT_REJECT_RATE_MIN <= rr <= _ACCEPT_REJECT_RATE_MAX
    )
    details["reject_rate_bounds"] = (
        f"{rr:.4f} in [{_ACCEPT_REJECT_RATE_MIN}, {_ACCEPT_REJECT_RATE_MAX}] -> "
        f"{'PASS' if checks['reject_rate_bounds'] else 'FAIL'}"
    )

    if seen_f1 is not None:
        checks["seen_user_f1"] = seen_f1 >= _ACCEPT_SEEN_MACRO_F1
        details["seen_user_f1"] = (
            f"{seen_f1:.4f} >= {_ACCEPT_SEEN_MACRO_F1} -> "
            f"{'PASS' if checks['seen_user_f1'] else 'FAIL'}"
        )

    if unseen_f1 is not None:
        checks["unseen_user_f1"] = unseen_f1 >= _ACCEPT_UNSEEN_MACRO_F1
        details["unseen_user_f1"] = (
            f"{unseen_f1:.4f} >= {_ACCEPT_UNSEEN_MACRO_F1} -> "
            f"{'PASS' if checks['unseen_user_f1'] else 'FAIL'}"
        )

    return checks, details


def _segment_duration_distribution(
    labels: Sequence[str],
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> dict[str, int]:
    """Build a histogram of segment durations from a label sequence.

    Duration buckets: ``"60s"``, ``"120s"``, ``"180s"``, ``"300s"``,
    ``"300s+"``.  Each segment's duration is
    ``bucket_count * bucket_seconds``.
    """
    if not labels:
        return {}

    timestamps = [
        datetime(2000, 1, 1) + timedelta(seconds=i * bucket_seconds)
        for i in range(len(labels))
    ]
    segments = segmentize(timestamps, list(labels), bucket_seconds)

    bins = [60, 120, 180, 300]
    hist: dict[str, int] = {}
    for seg in segments:
        dur = seg.bucket_count * bucket_seconds
        placed = False
        for b in bins:
            if dur <= b:
                key = f"{b}s"
                hist[key] = hist.get(key, 0) + 1
                placed = True
                break
        if not placed:
            hist["300s+"] = hist.get("300s+", 0) + 1
    return hist


def evaluate_model(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    holdout_users: Sequence[str] = (),
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    eval_mode: Literal[
        "raw", "calibrated", "calibrated_reject", "smoothed", "interval"
    ] = "raw",
    calibrator: Calibrator | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    schema_version: str = "v1",
) -> EvaluationReport:
    """Run comprehensive evaluation of a trained model on a test set.

    Computes overall metrics (macro-F1, weighted-F1), per-class precision /
    recall / F1, per-user macro-F1, calibration curves, user-stratification
    report, and acceptance-gate checks.

    Args:
        model: Trained LightGBM booster.
        test_df: Test DataFrame containing ``FEATURE_COLUMNS``, ``label``,
            and ``user_id`` columns.
        cat_encoders: Pre-fitted categorical encoders from the training run.
        holdout_users: User IDs that were held out from training, used to
            split seen-vs-unseen evaluation.
        reject_threshold: Max-probability below which a prediction is
            treated as rejected (``Mixed/Unknown``).
        eval_mode: Evaluation pipeline to use.  ``"raw"`` uses model
            probabilities directly.  ``"calibrated"`` applies a calibrator
            before metrics (no reject).  ``"calibrated_reject"`` applies
            calibrator + reject.  ``"smoothed"`` adds rolling-majority
            smoothing after reject.  ``"interval"`` aggregates smoothed
            predictions into segments and evaluates per-interval accuracy.
        calibrator: Probability calibrator to apply in non-raw modes.
            Required when *eval_mode* is not ``"raw"``.
        smooth_window: Window size for rolling-majority smoothing.
        schema_version: ``"v1"`` or ``"v2"`` — selects categorical columns for
            unknown-category-rate (see :func:`~taskclf.train.lgbm.get_categorical_columns`).

    Returns:
        A frozen :class:`EvaluationReport` with all evaluation artifacts.
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))
    label_names = list(le.classes_)

    y_proba = predict_proba(model, test_df, cat_encoders)

    if eval_mode != "raw" and calibrator is not None:
        y_proba = calibrator.calibrate(y_proba)

    y_pred_indices = y_proba.argmax(axis=1)
    y_pred_labels = list(le.inverse_transform(y_pred_indices))

    apply_reject = eval_mode in ("raw", "calibrated_reject", "smoothed", "interval")
    if apply_reject:
        confidences = y_proba.max(axis=1)
        rejected = confidences < reject_threshold
        labels_for_metrics = [
            MIXED_UNKNOWN if rej else lbl for lbl, rej in zip(y_pred_labels, rejected)
        ]
    else:
        labels_for_metrics = list(y_pred_labels)

    if eval_mode in ("smoothed", "interval"):
        labels_for_metrics = rolling_majority(labels_for_metrics, smooth_window)

    y_true = list(test_df["label"].values)
    user_ids = list(test_df["user_id"].values)
    y_true_indices = le.transform(y_true)

    if eval_mode == "interval":
        bucket_starts = [
            datetime(2000, 1, 1) + timedelta(seconds=i * DEFAULT_BUCKET_SECONDS)
            for i in range(len(labels_for_metrics))
        ]
        pred_segments = segmentize(
            bucket_starts, labels_for_metrics, DEFAULT_BUCKET_SECONDS
        )
        true_segments = segmentize(bucket_starts, y_true, DEFAULT_BUCKET_SECONDS)

        interval_correct = 0
        interval_total = len(pred_segments)
        true_map = {s.start_ts: s.label for s in true_segments}
        for seg in pred_segments:
            gold = true_map.get(seg.start_ts)
            if gold is not None and gold == seg.label:
                interval_correct += 1
        interval_accuracy = (
            interval_correct / interval_total if interval_total > 0 else 0.0
        )

        metrics = {
            "macro_f1": round(interval_accuracy, 4),
            "weighted_f1": round(interval_accuracy, 4),
        }
        pc = per_class_metrics(y_true, labels_for_metrics, label_names)
    else:
        metrics = compute_metrics(y_true, labels_for_metrics, label_names)
        pc = per_class_metrics(y_true, labels_for_metrics, label_names)

    cm_df = confusion_matrix_df(y_true, labels_for_metrics, label_names)
    cm_list = cm_df.values.tolist()
    top_pairs = top_confusion_pairs(cm_list, label_names)
    ece = round(
        expected_calibration_error_multiclass(y_true_indices, y_proba, label_names),
        4,
    )
    brier = round(multiclass_brier_score(y_true_indices, y_proba), 4)
    ll = round(multiclass_log_loss_score(y_true_indices, y_proba), 4)
    slices = slice_metrics_by_columns(
        test_df,
        y_true,
        labels_for_metrics,
        label_names,
    )
    cat_cols = get_categorical_columns(schema_version)
    unknown_rates = unknown_category_rates(test_df, cat_encoders, cat_cols)

    pu = per_user_metrics(y_true, labels_for_metrics, user_ids, label_names)
    cal = calibration_curve_data(y_true_indices, y_proba, label_names)
    strat = user_stratification_report(user_ids, y_true, label_names)
    rr = reject_rate(labels_for_metrics, MIXED_UNKNOWN)

    fr = round(flap_rate(labels_for_metrics), 4)
    seg_dist = _segment_duration_distribution(labels_for_metrics)

    seen_f1: float | None = None
    unseen_f1: float | None = None
    holdout_set = set(holdout_users)

    if holdout_set:
        seen_mask = [uid not in holdout_set for uid in user_ids]
        unseen_mask = [uid in holdout_set for uid in user_ids]

        if any(seen_mask):
            seen_true = [y for y, m in zip(y_true, seen_mask) if m]
            seen_pred = [y for y, m in zip(labels_for_metrics, seen_mask) if m]
            seen_f1 = round(
                float(
                    f1_score(
                        seen_true,
                        seen_pred,
                        labels=label_names,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            )

        if any(unseen_mask):
            unseen_true = [y for y, m in zip(y_true, unseen_mask) if m]
            unseen_pred = [y for y, m in zip(labels_for_metrics, unseen_mask) if m]
            unseen_f1 = round(
                float(
                    f1_score(
                        unseen_true,
                        unseen_pred,
                        labels=label_names,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            )

    checks, check_details = _check_acceptance(
        metrics["macro_f1"],
        metrics["weighted_f1"],
        pc,
        rr,
        seen_f1,
        unseen_f1,
    )

    return EvaluationReport(
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class=pc,
        confusion_matrix=cm_list,
        label_names=label_names,
        per_user=pu,
        calibration=cal,
        stratification=strat,
        seen_user_f1=seen_f1,
        unseen_user_f1=unseen_f1,
        reject_rate=round(rr, 4),
        acceptance_checks=checks,
        acceptance_details=check_details,
        flip_rate=fr,
        segment_duration_distribution=seg_dist,
        eval_mode=eval_mode,
        top_confusion_pairs=top_pairs,
        expected_calibration_error=ece,
        multiclass_brier_score=brier,
        multiclass_log_loss=ll,
        slice_metrics=slices,
        unknown_category_rates=unknown_rates,
    )


def tune_reject_threshold(
    model: lgb.Booster,
    val_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    thresholds: Sequence[float] | None = None,
    reject_rate_min: float = _ACCEPT_REJECT_RATE_MIN,
    reject_rate_max: float = _ACCEPT_REJECT_RATE_MAX,
    calibrator: Calibrator | None = None,
) -> RejectTuningResult:
    """Sweep reject thresholds and pick the best one.

    For each candidate threshold the function computes accuracy on
    accepted (non-rejected) windows, the reject rate, coverage (fraction
    of windows kept), and macro-F1.  The best threshold is the one that
    maximises accuracy on accepted windows while keeping reject rate
    within *[reject_rate_min, reject_rate_max]*.

    Args:
        model: Trained LightGBM booster.
        val_df: Validation DataFrame with ``FEATURE_COLUMNS``, ``label``,
            and ``user_id`` columns.
        cat_encoders: Pre-fitted categorical encoders.
        thresholds: Candidate thresholds to evaluate.  Defaults to
            ``np.arange(0.10, 1.00, 0.05)``.
        reject_rate_min: Lower bound for acceptable reject rate.
        reject_rate_max: Upper bound for acceptable reject rate.
        calibrator: When provided, raw probabilities are calibrated
            before extracting confidences for the threshold sweep.
            This ensures the threshold is tuned on the same probability
            space used at inference time.

    Returns:
        A :class:`RejectTuningResult` with the optimal threshold and
        the full sweep table.
    """
    if thresholds is None:
        thresholds = list(np.round(np.arange(0.10, 1.00, 0.05), 2))

    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))
    label_names = list(le.classes_)

    y_proba = predict_proba(model, val_df, cat_encoders)
    if calibrator is not None:
        y_proba = calibrator.calibrate(y_proba)
    y_pred_indices = y_proba.argmax(axis=1)
    y_pred_labels = np.array(le.inverse_transform(y_pred_indices))
    y_true = np.array(val_df["label"].values)
    confidences = y_proba.max(axis=1)

    sweep: list[dict[str, float]] = []
    best_threshold = DEFAULT_REJECT_THRESHOLD
    best_acc = -1.0

    for t in thresholds:
        rejected = confidences < t
        rr = float(rejected.mean())
        coverage = 1.0 - rr

        accepted_mask = ~rejected
        if accepted_mask.any():
            acc = float(
                accuracy_score(y_true[accepted_mask], y_pred_labels[accepted_mask])
            )
            mf1 = float(
                f1_score(
                    y_true[accepted_mask],
                    y_pred_labels[accepted_mask],
                    labels=label_names,
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            acc = 0.0
            mf1 = 0.0

        sweep.append(
            {
                "threshold": round(float(t), 4),
                "accuracy_on_accepted": round(acc, 4),
                "reject_rate": round(rr, 4),
                "coverage": round(coverage, 4),
                "macro_f1": round(mf1, 4),
            }
        )

        if reject_rate_min <= rr <= reject_rate_max and acc > best_acc:
            best_acc = acc
            best_threshold = float(t)

    return RejectTuningResult(
        best_threshold=round(best_threshold, 4),
        sweep=sweep,
    )


def write_evaluation_artifacts(
    report: EvaluationReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Write evaluation report artifacts to disk.

    Writes ``evaluation.json`` (full report) and ``calibration.json``
    (per-class calibration curve data) into *output_dir*.

    Args:
        report: A completed evaluation report.
        output_dir: Target directory (created if needed).

    Returns:
        Dict mapping artifact name to its written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    eval_path = output_dir / "evaluation.json"
    eval_path.write_text(json.dumps(report.model_dump(), indent=2, default=str))
    paths["evaluation"] = eval_path

    cal_path = output_dir / "calibration.json"
    cal_path.write_text(json.dumps(report.calibration, indent=2))
    paths["calibration"] = cal_path

    cm_df = pd.DataFrame(
        report.confusion_matrix,
        index=report.label_names,
        columns=report.label_names,
    )
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    paths["confusion_matrix"] = cm_path

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes_flat = axes.flatten()
        for idx, name in enumerate(report.label_names):
            ax = axes_flat[idx]
            cal = report.calibration.get(name, {})
            frac = cal.get("fraction_of_positives", [])
            mean_pred = cal.get("mean_predicted_value", [])
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            if frac and mean_pred:
                ax.plot(mean_pred, frac, "s-")
            ax.set_title(name, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted", fontsize=7)
            ax.set_ylabel("Fraction positive", fontsize=7)

        for idx in range(len(report.label_names), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Per-Class Calibration Curves")
        fig.tight_layout()
        plot_path = output_dir / "calibration.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        paths["calibration_plot"] = plot_path
    except Exception:
        logger.debug("Calibration plot generation failed", exc_info=True)

    return paths
