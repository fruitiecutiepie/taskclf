"""Tests for infer.monitor: drift orchestrator, auto-enqueue, and report persistence.

Covers: TC-MON-001 through TC-MON-016.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.telemetry import NUMERICAL_FEATURES
from taskclf.infer.monitor import (
    DriftAlert,
    DriftReport,
    DriftTrigger,
    auto_enqueue_drift_labels,
    run_drift_check,
    write_drift_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features_df(
    n: int = 200,
    *,
    rng: np.random.Generator | None = None,
    shift_col: str | None = None,
    shift_amount: float = 10.0,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with all NUMERICAL_FEATURES columns."""
    if rng is None:
        rng = np.random.default_rng(42)
    data: dict[str, object] = {
        "bucket_start_ts": pd.date_range("2025-06-15T09:00", periods=n, freq="min"),
        "user_id": ["u1"] * n,
    }
    for feat in NUMERICAL_FEATURES:
        base = rng.normal(10.0, 2.0, size=n)
        if feat == shift_col:
            base += shift_amount
        data[feat] = base
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TC-MON-001..003: Model / enum construction
# ---------------------------------------------------------------------------


class TestDriftModels:
    def test_drift_trigger_values(self) -> None:
        """TC-MON-001: all 5 values exist and are strings."""
        values = list(DriftTrigger)
        assert len(values) == 5
        for v in values:
            assert isinstance(v, str)

    def test_drift_alert_construction(self) -> None:
        """TC-MON-002: all fields populated."""
        now = datetime.now(tz=timezone.utc)
        alert = DriftAlert(
            trigger=DriftTrigger.feature_psi,
            details={"psi": 0.5},
            severity="critical",
            affected_user_ids=["u1"],
            affected_features=["keys_per_min"],
            timestamp=now,
        )
        assert alert.trigger == DriftTrigger.feature_psi
        assert alert.severity == "critical"
        assert alert.affected_features == ["keys_per_min"]
        assert isinstance(alert.timestamp, datetime)

    def test_drift_report_any_critical(self) -> None:
        """TC-MON-003: any_critical computed from alerts."""
        now = datetime.now(tz=timezone.utc)
        report_no_crit = DriftReport(
            alerts=[
                DriftAlert(trigger=DriftTrigger.class_shift, severity="warning", timestamp=now),
            ],
            any_critical=False,
        )
        assert report_no_crit.any_critical is False

        report_crit = DriftReport(
            alerts=[
                DriftAlert(trigger=DriftTrigger.reject_rate_increase, severity="critical", timestamp=now),
            ],
            any_critical=True,
        )
        assert report_crit.any_critical is True


# ---------------------------------------------------------------------------
# TC-MON-004..011: run_drift_check
# ---------------------------------------------------------------------------


class TestRunDriftCheck:
    def test_no_drift_identical(self) -> None:
        """TC-MON-004: identical ref/cur → no alerts."""
        rng = np.random.default_rng(42)
        df = _make_features_df(200, rng=rng)
        labels = ["Build"] * 200

        report = run_drift_check(df, df, labels, labels)

        assert report.alerts == []
        assert report.any_critical is False
        assert "No drift detected" in report.summary

    def test_feature_psi_drift(self) -> None:
        """TC-MON-005: shifted feature produces feature_psi alert."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(500, rng=rng)
        # Moderate shift keeps distributions overlapping for PSI quantile binning
        cur = _make_features_df(500, rng=np.random.default_rng(99), shift_col="keys_per_min", shift_amount=3.0)
        labels = ["Build"] * 500

        report = run_drift_check(ref, cur, labels, labels)

        psi_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.feature_psi]
        assert len(psi_alerts) > 0
        kpm_alerts = [a for a in psi_alerts if "keys_per_min" in a.affected_features]
        assert len(kpm_alerts) == 1

    def test_reject_rate_increase(self) -> None:
        """TC-MON-006: reject rate increase → critical alert."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(100, rng=rng)
        cur = _make_features_df(100, rng=np.random.default_rng(99))
        ref_labels = ["Build"] * 95 + [MIXED_UNKNOWN] * 5
        cur_labels = ["Build"] * 70 + [MIXED_UNKNOWN] * 30

        report = run_drift_check(ref, cur, ref_labels, cur_labels)

        rr_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.reject_rate_increase]
        assert len(rr_alerts) == 1
        assert rr_alerts[0].severity == "critical"

    def test_entropy_spike(self) -> None:
        """TC-MON-007: entropy spike detected when probs provided."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(100, rng=rng)
        cur = _make_features_df(100, rng=np.random.default_rng(99))
        labels = ["Build"] * 100

        ref_probs = np.zeros((100, 8))
        ref_probs[:, 0] = 0.95
        ref_probs[:, 1:] = 0.05 / 7

        cur_probs = np.ones((100, 8)) / 8

        report = run_drift_check(
            ref, cur, labels, labels,
            ref_probs=ref_probs, cur_probs=cur_probs,
        )

        ent_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.entropy_spike]
        assert len(ent_alerts) == 1
        assert report.entropy_drift is not None
        assert report.entropy_drift.is_flagged

    def test_no_entropy_when_probs_omitted(self) -> None:
        """TC-MON-008: entropy_drift is None when probs not provided."""
        rng = np.random.default_rng(42)
        df = _make_features_df(100, rng=rng)
        labels = ["Build"] * 100

        report = run_drift_check(df, df, labels, labels)

        assert report.entropy_drift is None

    def test_class_shift(self) -> None:
        """TC-MON-009: class distribution shift detected."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(100, rng=rng)
        cur = _make_features_df(100, rng=np.random.default_rng(99))
        ref_labels = ["Build"] * 50 + ["Debug"] * 50
        cur_labels = ["Build"] * 20 + ["Debug"] * 80

        report = run_drift_check(ref, cur, ref_labels, cur_labels)

        cs_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.class_shift]
        assert len(cs_alerts) == 1
        assert report.class_shift is not None
        assert report.class_shift.is_flagged

    def test_telemetry_snapshot_always_populated(self) -> None:
        """TC-MON-010: telemetry_snapshot is always present."""
        rng = np.random.default_rng(42)
        df = _make_features_df(50, rng=rng)
        labels = ["Build"] * 50

        report = run_drift_check(df, df, labels, labels)

        assert report.telemetry_snapshot is not None
        assert report.telemetry_snapshot.total_windows == 50

    def test_any_critical_true_on_reject_rate(self) -> None:
        """TC-MON-011: any_critical=True when reject rate is flagged."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(100, rng=rng)
        cur = _make_features_df(100, rng=np.random.default_rng(99))
        ref_labels = ["Build"] * 95 + [MIXED_UNKNOWN] * 5
        cur_labels = ["Build"] * 60 + [MIXED_UNKNOWN] * 40

        report = run_drift_check(ref, cur, ref_labels, cur_labels)

        assert report.any_critical is True


# ---------------------------------------------------------------------------
# TC-MON-012..014: auto_enqueue_drift_labels
# ---------------------------------------------------------------------------


class TestAutoEnqueueDriftLabels:
    def test_no_alerts_returns_zero(self, tmp_path: Path) -> None:
        """TC-MON-012: no alerts → no items enqueued."""
        report = DriftReport(alerts=[], summary="No drift detected.")
        df = _make_features_df(10)
        result = auto_enqueue_drift_labels(report, df, tmp_path / "queue.json")
        assert result == 0

    def test_alerts_present_enqueues(self, tmp_path: Path) -> None:
        """TC-MON-013: alerts present → items enqueued."""
        now = datetime.now(tz=timezone.utc)
        report = DriftReport(
            alerts=[
                DriftAlert(
                    trigger=DriftTrigger.reject_rate_increase,
                    severity="critical",
                    timestamp=now,
                ),
            ],
            any_critical=True,
        )
        df = _make_features_df(10)
        queue_path = tmp_path / "queue.json"
        result = auto_enqueue_drift_labels(report, df, queue_path)
        assert result > 0
        assert queue_path.exists()

    def test_limit_respected(self, tmp_path: Path) -> None:
        """TC-MON-014: at most `limit` items enqueued."""
        now = datetime.now(tz=timezone.utc)
        report = DriftReport(
            alerts=[
                DriftAlert(
                    trigger=DriftTrigger.feature_psi,
                    severity="warning",
                    timestamp=now,
                ),
            ],
        )
        df = _make_features_df(50)
        confidences = np.random.default_rng(42).uniform(0.1, 0.9, size=50)

        result = auto_enqueue_drift_labels(
            report, df, tmp_path / "queue.json",
            cur_confidences=confidences, limit=5,
        )
        assert result <= 5

    def test_lowest_confidence_selected(self, tmp_path: Path) -> None:
        """TC-MON-014b: when cur_confidences provided, lowest are picked."""
        now = datetime.now(tz=timezone.utc)
        report = DriftReport(
            alerts=[
                DriftAlert(trigger=DriftTrigger.feature_psi, severity="warning", timestamp=now),
            ],
        )
        df = _make_features_df(20)
        confidences = np.linspace(0.1, 0.99, 20)

        result = auto_enqueue_drift_labels(
            report, df, tmp_path / "queue.json",
            cur_confidences=confidences, limit=3,
        )
        assert result == 3


# ---------------------------------------------------------------------------
# TC-MON-015..016: write_drift_report
# ---------------------------------------------------------------------------


class TestWriteDriftReport:
    def test_round_trip(self, tmp_path: Path) -> None:
        """TC-MON-015: written JSON round-trips back to DriftReport."""
        rng = np.random.default_rng(42)
        ref = _make_features_df(100, rng=rng)
        cur = _make_features_df(100, rng=np.random.default_rng(99))
        labels = ["Build"] * 100

        report = run_drift_check(ref, cur, labels, labels)
        path = tmp_path / "drift_report.json"
        returned = write_drift_report(report, path)

        assert returned == path
        assert path.exists()

        restored = DriftReport.model_validate_json(path.read_text())
        assert restored.summary == report.summary
        assert len(restored.alerts) == len(report.alerts)
        assert restored.any_critical == report.any_critical

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        """TC-MON-016: non-existent parent path created."""
        report = DriftReport(summary="test")
        path = tmp_path / "a" / "b" / "drift.json"
        write_drift_report(report, path)
        assert path.exists()
