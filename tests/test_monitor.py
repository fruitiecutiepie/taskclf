"""Tests for taskclf.infer.monitor — drift monitoring orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.infer.monitor import (
    DriftReport,
    DriftTrigger,
    auto_enqueue_drift_labels,
    run_drift_check,
    write_drift_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(rng: np.random.Generator, n: int, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "bucket_start_ts": pd.date_range("2026-02-20T08:00", periods=n, freq="1min"),
        "user_id": ["user-1"] * n,
        "keys_per_min": rng.normal(30 + shift, 5, n),
        "clicks_per_min": rng.normal(10 + shift, 2, n),
        "scroll_events_per_min": rng.normal(5, 1, n),
        "backspace_ratio": rng.uniform(0, 0.3, n),
        "shortcut_rate": rng.uniform(0, 2, n),
        "mouse_distance": rng.normal(500, 100, n),
        "app_switch_count_last_5m": rng.integers(0, 20, n),
        "app_foreground_time_ratio": rng.uniform(0.5, 1.0, n),
        "app_change_count": rng.integers(0, 10, n),
        "hour_of_day": rng.integers(8, 18, n),
        "session_length_so_far": rng.uniform(0, 120, n),
        "active_seconds_keyboard": rng.uniform(0, 60, n),
        "active_seconds_mouse": rng.uniform(0, 60, n),
        "active_seconds_any": rng.uniform(0, 60, n),
        "max_idle_run_seconds": rng.uniform(0, 60, n),
        "event_density": rng.uniform(0, 10, n),
    })


# ---------------------------------------------------------------------------
# run_drift_check
# ---------------------------------------------------------------------------


class TestRunDriftCheck:
    def test_no_drift_clean_data(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)
        ref_labels = ["Build"] * 180 + [MIXED_UNKNOWN] * 20
        cur_labels = ["Build"] * 175 + [MIXED_UNKNOWN] * 25

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        assert isinstance(report, DriftReport)
        assert report.telemetry_snapshot is not None

    def test_feature_drift_detected(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200, shift=20.0)

        ref_labels = ["Build"] * 200
        cur_labels = ["Build"] * 200

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        psi_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.feature_psi]
        ks_alerts = [a for a in report.alerts if a.trigger == DriftTrigger.feature_ks]
        assert len(psi_alerts) + len(ks_alerts) > 0

    def test_reject_rate_increase_detected(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)

        ref_labels = ["Build"] * 180 + [MIXED_UNKNOWN] * 20
        cur_labels = ["Build"] * 120 + [MIXED_UNKNOWN] * 80

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        assert report.reject_rate_drift is not None
        assert report.reject_rate_drift.is_flagged

        rr_alerts = [
            a for a in report.alerts
            if a.trigger == DriftTrigger.reject_rate_increase
        ]
        assert len(rr_alerts) == 1

    def test_entropy_spike_detected(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 100)
        cur_df = _make_features(rng, 100)

        confident = np.zeros((100, 8))
        confident[:, 0] = 0.90
        confident[:, 1:] = 0.10 / 7

        uniform = np.ones((100, 8)) / 8

        report = run_drift_check(
            ref_df, cur_df,
            ref_labels=["Build"] * 100,
            cur_labels=["Build"] * 100,
            ref_probs=confident,
            cur_probs=uniform,
        )
        assert report.entropy_drift is not None
        assert report.entropy_drift.is_flagged

    def test_class_shift_detected(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)

        ref_labels = ["Build"] * 100 + ["Debug"] * 100
        cur_labels = ["Build"] * 30 + ["Debug"] * 170

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        assert report.class_shift is not None
        assert report.class_shift.is_flagged

    def test_summary_not_empty(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 100)
        cur_df = _make_features(rng, 100)
        report = run_drift_check(ref_df, cur_df, ["Build"] * 100, ["Build"] * 100)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_any_critical_flag(self) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)
        ref_labels = ["Build"] * 180 + [MIXED_UNKNOWN] * 20
        cur_labels = ["Build"] * 100 + [MIXED_UNKNOWN] * 100

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        if report.alerts:
            assert report.any_critical == any(
                a.severity == "critical" for a in report.alerts
            )


# ---------------------------------------------------------------------------
# auto_enqueue_drift_labels
# ---------------------------------------------------------------------------


class TestAutoEnqueueDriftLabels:
    def test_no_alerts_no_enqueue(self, tmp_path: Path) -> None:
        report = DriftReport(summary="No drift detected.")
        df = pd.DataFrame({
            "bucket_start_ts": pd.date_range("2026-02-20", periods=10, freq="1min"),
            "user_id": ["u1"] * 10,
        })
        queue_path = tmp_path / "queue.json"
        count = auto_enqueue_drift_labels(report, df, queue_path)
        assert count == 0

    def test_enqueues_on_alerts(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)

        ref_labels = ["Build"] * 180 + [MIXED_UNKNOWN] * 20
        cur_labels = ["Build"] * 100 + [MIXED_UNKNOWN] * 100

        report = run_drift_check(ref_df, cur_df, ref_labels, cur_labels)
        assert len(report.alerts) > 0

        confidences = rng.uniform(0.3, 0.9, size=200)
        queue_path = tmp_path / "queue.json"
        count = auto_enqueue_drift_labels(
            report, cur_df, queue_path,
            cur_confidences=confidences,
            limit=10,
        )
        assert count > 0
        assert count <= 10
        assert queue_path.exists()

    def test_respects_limit(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 200)
        cur_df = _make_features(rng, 200)

        report = run_drift_check(
            ref_df, cur_df,
            ref_labels=["Build"] * 180 + [MIXED_UNKNOWN] * 20,
            cur_labels=["Build"] * 100 + [MIXED_UNKNOWN] * 100,
        )

        queue_path = tmp_path / "queue.json"
        count = auto_enqueue_drift_labels(report, cur_df, queue_path, limit=3)
        assert count <= 3


# ---------------------------------------------------------------------------
# write_drift_report
# ---------------------------------------------------------------------------


class TestWriteDriftReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        report = DriftReport(summary="Test report")
        path = write_drift_report(report, tmp_path / "drift.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["summary"] == "Test report"

    def test_full_report_serializable(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        ref_df = _make_features(rng, 100)
        cur_df = _make_features(rng, 100, shift=15.0)

        report = run_drift_check(
            ref_df, cur_df,
            ref_labels=["Build"] * 100,
            cur_labels=["Build"] * 100,
        )
        path = write_drift_report(report, tmp_path / "full.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "alerts" in data
        assert "summary" in data
