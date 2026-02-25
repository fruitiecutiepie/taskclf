"""Drift monitoring orchestrator: run checks and auto-create labeling tasks.

Ties together the pure drift statistics from :mod:`taskclf.core.drift` with
the telemetry store and the active labeling queue.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.defaults import (
    DEFAULT_CLASS_SHIFT_THRESHOLD,
    DEFAULT_DRIFT_AUTO_LABEL_LIMIT,
    DEFAULT_ENTROPY_SPIKE_MULTIPLIER,
    DEFAULT_KS_ALPHA,
    DEFAULT_PSI_THRESHOLD,
    DEFAULT_REJECT_RATE_INCREASE_THRESHOLD,
    MIXED_UNKNOWN,
)
from taskclf.core.drift import (
    ClassShiftResult,
    EntropyDrift,
    FeatureDriftReport,
    RejectRateDrift,
    detect_class_shift,
    detect_entropy_spike,
    detect_reject_rate_increase,
    feature_drift_report,
)
from taskclf.core.telemetry import (
    NUMERICAL_FEATURES,
    TelemetrySnapshot,
    compute_telemetry,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DriftTrigger(StrEnum):
    """Enumeration of drift trigger types."""

    feature_psi = "feature_psi"
    feature_ks = "feature_ks"
    reject_rate_increase = "reject_rate_increase"
    entropy_spike = "entropy_spike"
    class_shift = "class_shift"


class DriftAlert(BaseModel):
    """A single drift alert raised during a check."""

    trigger: DriftTrigger
    details: dict[str, object] = Field(default_factory=dict)
    severity: Literal["warning", "critical"]
    affected_user_ids: list[str] = Field(default_factory=list)
    affected_features: list[str] = Field(default_factory=list)
    timestamp: datetime


class DriftReport(BaseModel):
    """Aggregated output of a full drift check run."""

    alerts: list[DriftAlert] = Field(default_factory=list)
    feature_report: FeatureDriftReport | None = None
    reject_rate_drift: RejectRateDrift | None = None
    entropy_drift: EntropyDrift | None = None
    class_shift: ClassShiftResult | None = None
    telemetry_snapshot: TelemetrySnapshot | None = None
    summary: str = ""
    any_critical: bool = False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_drift_check(
    ref_features_df: pd.DataFrame,
    cur_features_df: pd.DataFrame,
    ref_labels: Sequence[str],
    cur_labels: Sequence[str],
    *,
    ref_probs: np.ndarray | None = None,
    cur_probs: np.ndarray | None = None,
    cur_confidences: np.ndarray | None = None,
    user_ids: Sequence[str] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    reject_increase_threshold: float = DEFAULT_REJECT_RATE_INCREASE_THRESHOLD,
    entropy_multiplier: float = DEFAULT_ENTROPY_SPIKE_MULTIPLIER,
    class_shift_threshold: float = DEFAULT_CLASS_SHIFT_THRESHOLD,
    reject_label: str = MIXED_UNKNOWN,
) -> DriftReport:
    """Run all drift checks and return a consolidated report.

    Args:
        ref_features_df: Reference-period feature DataFrame.
        cur_features_df: Current-period feature DataFrame.
        ref_labels: Reference predicted labels.
        cur_labels: Current predicted labels.
        ref_probs: Reference probability matrix ``(n, k)``.
        cur_probs: Current probability matrix ``(n, k)``.
        cur_confidences: ``max(proba)`` per current row (for telemetry).
        user_ids: User IDs corresponding to current rows.
        psi_threshold: PSI threshold for feature drift.
        ks_alpha: KS significance level.
        reject_increase_threshold: Reject-rate increase threshold.
        entropy_multiplier: Entropy spike multiplier.
        class_shift_threshold: Class-distribution shift threshold.
        reject_label: Label used for rejected predictions.

    Returns:
        A :class:`DriftReport` containing all alerts and sub-reports.
    """
    now = datetime.now(tz=timezone.utc)
    alerts: list[DriftAlert] = []

    feat_report = feature_drift_report(
        ref_features_df,
        cur_features_df,
        NUMERICAL_FEATURES,
        psi_threshold=psi_threshold,
        ks_alpha=ks_alpha,
    )
    if feat_report.flagged_features:
        for result in feat_report.results:
            if not result.is_drifted:
                continue
            trigger = (
                DriftTrigger.feature_psi
                if result.psi > psi_threshold
                else DriftTrigger.feature_ks
            )
            severity: Literal["warning", "critical"] = (
                "critical" if result.psi > psi_threshold * 2 else "warning"
            )
            alerts.append(DriftAlert(
                trigger=trigger,
                details={
                    "feature": result.feature,
                    "psi": result.psi,
                    "ks_statistic": result.ks_statistic,
                    "ks_p_value": result.ks_p_value,
                },
                severity=severity,
                affected_features=[result.feature],
                timestamp=now,
            ))

    rr_drift = detect_reject_rate_increase(
        ref_labels, cur_labels,
        threshold=reject_increase_threshold,
        reject_label=reject_label,
    )
    if rr_drift.is_flagged:
        alerts.append(DriftAlert(
            trigger=DriftTrigger.reject_rate_increase,
            details={
                "ref_rate": rr_drift.ref_rate,
                "cur_rate": rr_drift.cur_rate,
                "increase": rr_drift.increase,
            },
            severity="critical",
            timestamp=now,
        ))

    ent_drift: EntropyDrift | None = None
    if ref_probs is not None and cur_probs is not None:
        ent_drift = detect_entropy_spike(
            ref_probs, cur_probs,
            spike_multiplier=entropy_multiplier,
        )
        if ent_drift.is_flagged:
            alerts.append(DriftAlert(
                trigger=DriftTrigger.entropy_spike,
                details={
                    "ref_mean_entropy": ent_drift.ref_mean_entropy,
                    "cur_mean_entropy": ent_drift.cur_mean_entropy,
                    "ratio": ent_drift.ratio,
                },
                severity="warning",
                timestamp=now,
            ))

    cls_shift = detect_class_shift(
        ref_labels, cur_labels,
        threshold=class_shift_threshold,
    )
    if cls_shift.is_flagged:
        alerts.append(DriftAlert(
            trigger=DriftTrigger.class_shift,
            details={
                "max_shift": cls_shift.max_shift,
                "shifted_classes": cls_shift.shifted_classes,
            },
            severity="warning",
            affected_features=cls_shift.shifted_classes,
            timestamp=now,
        ))

    telemetry = compute_telemetry(
        cur_features_df,
        labels=cur_labels,
        confidences=cur_confidences,
        core_probs=cur_probs,
    )

    any_critical = any(a.severity == "critical" for a in alerts)
    parts: list[str] = []
    if not alerts:
        parts.append("No drift detected.")
    else:
        parts.append(f"{len(alerts)} alert(s) raised.")
        if feat_report.flagged_features:
            parts.append(f"Drifted features: {', '.join(feat_report.flagged_features)}.")
        if rr_drift.is_flagged:
            parts.append(
                f"Reject rate increased from {rr_drift.ref_rate:.2%} "
                f"to {rr_drift.cur_rate:.2%}."
            )
        if ent_drift is not None and ent_drift.is_flagged:
            parts.append(f"Entropy spike: {ent_drift.ratio:.1f}x reference.")
        if cls_shift.is_flagged:
            parts.append(
                f"Class shift in: {', '.join(cls_shift.shifted_classes)}."
            )

    return DriftReport(
        alerts=alerts,
        feature_report=feat_report,
        reject_rate_drift=rr_drift,
        entropy_drift=ent_drift,
        class_shift=cls_shift,
        telemetry_snapshot=telemetry,
        summary=" ".join(parts),
        any_critical=any_critical,
    )


# ---------------------------------------------------------------------------
# Auto-enqueue labeling tasks
# ---------------------------------------------------------------------------


def auto_enqueue_drift_labels(
    drift_report: DriftReport,
    cur_features_df: pd.DataFrame,
    queue_path: Path,
    *,
    cur_confidences: np.ndarray | None = None,
    limit: int = DEFAULT_DRIFT_AUTO_LABEL_LIMIT,
) -> int:
    """Create labeling tasks for drifted buckets.

    Selects buckets with the lowest confidence from the current window
    and enqueues them via :class:`ActiveLabelingQueue`.

    Args:
        drift_report: Output of :func:`run_drift_check`.
        cur_features_df: Current-period feature DataFrame.
        queue_path: Path to the labeling queue JSON file.
        cur_confidences: ``max(proba)`` per current row.
        limit: Maximum number of buckets to enqueue.

    Returns:
        Number of newly enqueued items.
    """
    if not drift_report.alerts:
        return 0

    from taskclf.labels.queue import ActiveLabelingQueue

    queue = ActiveLabelingQueue(queue_path)

    df = cur_features_df.copy()

    if cur_confidences is not None and len(cur_confidences) == len(df):
        df = df.assign(_confidence=cur_confidences)
        df = df.sort_values("_confidence", ascending=True)
    df = df.head(limit)

    buckets: list[dict[str, object]] = []
    for _, row in df.iterrows():
        bucket: dict[str, object] = {
            "user_id": str(row.get("user_id", "unknown")),
            "bucket_start_ts": row["bucket_start_ts"],
            "bucket_end_ts": row.get(
                "bucket_end_ts",
                pd.Timestamp(row["bucket_start_ts"]) + pd.Timedelta(seconds=60),
            ),
        }
        if "_confidence" in row.index:
            bucket["confidence"] = float(row["_confidence"])
        buckets.append(bucket)

    return queue.enqueue_drift(buckets)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_drift_report(report: DriftReport, path: Path) -> Path:
    """Persist a drift report as JSON.

    Args:
        report: The report to write.
        path: Destination file path.

    Returns:
        The *path* that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))
    return path
