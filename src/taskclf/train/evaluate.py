"""Full model evaluation pipeline: metrics, calibration, acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.metrics import (
    calibration_curve_data,
    compute_metrics,
    confusion_matrix_df,
    per_class_metrics,
    per_user_metrics,
    reject_rate,
    user_stratification_report,
)
from taskclf.core.types import LABEL_SET_V1
from taskclf.train.lgbm import FEATURE_COLUMNS, encode_categoricals

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
    per_class: dict[str, dict[str, float]]
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


def _predict_proba(
    model: lgb.Booster,
    df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None,
) -> np.ndarray:
    """Return raw probability matrix for *df*."""
    feat_df = df[FEATURE_COLUMNS].copy()
    feat_df, _ = encode_categoricals(feat_df, cat_encoders)
    x = feat_df.fillna(0).to_numpy(dtype=np.float64)
    return model.predict(x)


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
    details["macro_f1"] = f"{macro_f1:.4f} >= {_ACCEPT_MACRO_F1} -> {'PASS' if checks['macro_f1'] else 'FAIL'}"

    checks["weighted_f1"] = weighted_f1 >= _ACCEPT_WEIGHTED_F1
    details["weighted_f1"] = f"{weighted_f1:.4f} >= {_ACCEPT_WEIGHTED_F1} -> {'PASS' if checks['weighted_f1'] else 'FAIL'}"

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
        (lbl, m["precision"]) for lbl, m in per_class.items()
        if m["precision"] < _ACCEPT_MIN_CLASS_PRECISION
    ]
    checks["no_class_below_50_precision"] = len(low_prec) == 0
    if low_prec:
        details["no_class_below_50_precision"] = (
            "FAIL: " + ", ".join(f"{l}={p:.4f}" for l, p in low_prec)
        )
    else:
        details["no_class_below_50_precision"] = "PASS (all classes >= 0.50)"

    checks["reject_rate_bounds"] = _ACCEPT_REJECT_RATE_MIN <= rr <= _ACCEPT_REJECT_RATE_MAX
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


def evaluate_model(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    holdout_users: Sequence[str] = (),
    reject_threshold: float = 0.55,
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

    Returns:
        A frozen :class:`EvaluationReport` with all evaluation artifacts.
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))
    label_names = list(le.classes_)

    y_proba = _predict_proba(model, test_df, cat_encoders)
    y_pred_indices = y_proba.argmax(axis=1)
    y_pred_labels = list(le.inverse_transform(y_pred_indices))

    confidences = y_proba.max(axis=1)
    rejected = confidences < reject_threshold
    labels_with_reject = [
        MIXED_UNKNOWN if rej else lbl
        for lbl, rej in zip(y_pred_labels, rejected)
    ]

    y_true = list(test_df["label"].values)
    user_ids = list(test_df["user_id"].values)
    y_true_indices = le.transform(y_true)

    metrics = compute_metrics(y_true, y_pred_labels, label_names)
    pc = per_class_metrics(y_true, y_pred_labels, label_names)
    cm_df = confusion_matrix_df(y_true, y_pred_labels, label_names)
    pu = per_user_metrics(y_true, y_pred_labels, user_ids, label_names)
    cal = calibration_curve_data(y_true_indices, y_proba, label_names)
    strat = user_stratification_report(user_ids, y_true, label_names)
    rr = reject_rate(labels_with_reject, MIXED_UNKNOWN)

    seen_f1: float | None = None
    unseen_f1: float | None = None
    holdout_set = set(holdout_users)

    if holdout_set:
        seen_mask = [uid not in holdout_set for uid in user_ids]
        unseen_mask = [uid in holdout_set for uid in user_ids]

        if any(seen_mask):
            seen_true = [y for y, m in zip(y_true, seen_mask) if m]
            seen_pred = [y for y, m in zip(y_pred_labels, seen_mask) if m]
            from sklearn.metrics import f1_score

            seen_f1 = round(float(
                f1_score(seen_true, seen_pred, labels=label_names, average="macro", zero_division=0)
            ), 4)

        if any(unseen_mask):
            unseen_true = [y for y, m in zip(y_true, unseen_mask) if m]
            unseen_pred = [y for y, m in zip(y_pred_labels, unseen_mask) if m]
            from sklearn.metrics import f1_score

            unseen_f1 = round(float(
                f1_score(unseen_true, unseen_pred, labels=label_names, average="macro", zero_division=0)
            ), 4)

    checks, check_details = _check_acceptance(
        metrics["macro_f1"], metrics["weighted_f1"], pc, rr, seen_f1, unseen_f1,
    )

    return EvaluationReport(
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class=pc,
        confusion_matrix=cm_df.values.tolist(),
        label_names=label_names,
        per_user=pu,
        calibration=cal,
        stratification=strat,
        seen_user_f1=seen_f1,
        unseen_user_f1=unseen_f1,
        reject_rate=round(rr, 4),
        acceptance_checks=checks,
        acceptance_details=check_details,
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
        pass

    return paths
