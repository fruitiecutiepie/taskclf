"""Tests for taskclf.core.drift — pure statistical drift detection functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.drift import (
    ClassShiftResult,
    EntropyDrift,
    FeatureDriftReport,
    KsResult,
    RejectRateDrift,
    compute_ks,
    compute_psi,
    detect_class_shift,
    detect_entropy_spike,
    detect_reject_rate_increase,
    feature_drift_report,
)


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


class TestComputePsi:
    def test_identical_distributions_zero(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=500)
        psi = compute_psi(data, data)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_shifted_distribution_high_psi(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=500)
        cur = rng.normal(3, 1, size=500)
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_empty_arrays_return_zero(self) -> None:
        assert compute_psi(np.array([]), np.array([1.0, 2.0])) == 0.0
        assert compute_psi(np.array([1.0]), np.array([])) == 0.0

    def test_constant_arrays_return_zero(self) -> None:
        assert compute_psi(np.array([5.0] * 100), np.array([5.0] * 100)) == 0.0

    def test_nans_handled(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=200)
        cur = np.concatenate([rng.normal(0, 1, size=180), np.full(20, np.nan)])
        psi = compute_psi(ref, cur)
        assert np.isfinite(psi)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            ref = rng.normal(rng.uniform(-5, 5), 1, size=200)
            cur = rng.normal(rng.uniform(-5, 5), 1, size=200)
            assert compute_psi(ref, cur) >= 0.0


# ---------------------------------------------------------------------------
# compute_ks
# ---------------------------------------------------------------------------


class TestComputeKs:
    def test_identical_distributions_high_p_value(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=500)
        result = compute_ks(data, data)
        assert isinstance(result, KsResult)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert not result.is_significant()

    def test_different_distributions_significant(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=500)
        cur = rng.normal(5, 1, size=500)
        result = compute_ks(ref, cur)
        assert result.statistic > 0.5
        assert result.p_value < 0.001
        assert result.is_significant()

    def test_empty_arrays_not_significant(self) -> None:
        result = compute_ks(np.array([]), np.array([1.0]))
        assert not result.is_significant()

    def test_custom_alpha(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=100)
        cur = rng.normal(0.3, 1, size=100)
        result = compute_ks(ref, cur)
        sig_at_05 = result.is_significant(0.05)
        sig_at_001 = result.is_significant(0.001)
        if sig_at_001:
            assert sig_at_05


# ---------------------------------------------------------------------------
# feature_drift_report
# ---------------------------------------------------------------------------


class TestFeatureDriftReport:
    @pytest.fixture
    def make_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42)
        n = 200
        ref = pd.DataFrame({
            "keys_per_min": rng.normal(30, 5, n),
            "clicks_per_min": rng.normal(10, 2, n),
            "scroll_events_per_min": rng.normal(5, 1, n),
        })
        cur = pd.DataFrame({
            "keys_per_min": rng.normal(30, 5, n),
            "clicks_per_min": rng.normal(25, 2, n),  # shifted
            "scroll_events_per_min": rng.normal(5, 1, n),
        })
        return ref, cur

    def test_flags_drifted_features(self, make_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ref, cur = make_dfs
        features = ["keys_per_min", "clicks_per_min", "scroll_events_per_min"]
        report = feature_drift_report(ref, cur, features)
        assert isinstance(report, FeatureDriftReport)
        assert "clicks_per_min" in report.flagged_features
        assert len(report.results) == 3

    def test_no_drift_when_same(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.uniform(0, 1, n)})
        report = feature_drift_report(df, df, ["f1", "f2"])
        assert report.flagged_features == []

    def test_missing_columns_skipped(self) -> None:
        ref = pd.DataFrame({"f1": [1.0, 2.0]})
        cur = pd.DataFrame({"f1": [1.0, 2.0]})
        report = feature_drift_report(ref, cur, ["f1", "missing_col"])
        assert len(report.results) == 1


# ---------------------------------------------------------------------------
# detect_reject_rate_increase
# ---------------------------------------------------------------------------


class TestDetectRejectRateIncrease:
    def test_no_increase(self) -> None:
        ref = ["Build"] * 90 + [MIXED_UNKNOWN] * 10
        cur = ["Build"] * 85 + [MIXED_UNKNOWN] * 15
        result = detect_reject_rate_increase(ref, cur, threshold=0.10)
        assert isinstance(result, RejectRateDrift)
        assert not result.is_flagged

    def test_flagged_on_large_increase(self) -> None:
        ref = ["Build"] * 90 + [MIXED_UNKNOWN] * 10
        cur = ["Build"] * 70 + [MIXED_UNKNOWN] * 30
        result = detect_reject_rate_increase(ref, cur, threshold=0.10)
        assert result.is_flagged
        assert result.increase >= 0.10

    def test_empty_labels(self) -> None:
        result = detect_reject_rate_increase([], [], threshold=0.10)
        assert not result.is_flagged

    def test_exact_threshold_boundary(self) -> None:
        ref = ["Build"] * 90 + [MIXED_UNKNOWN] * 10
        cur = ["Build"] * 80 + [MIXED_UNKNOWN] * 20
        result = detect_reject_rate_increase(ref, cur, threshold=0.10)
        assert result.is_flagged


# ---------------------------------------------------------------------------
# detect_entropy_spike
# ---------------------------------------------------------------------------


class TestDetectEntropySpike:
    def test_no_spike(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(8) * 5, size=100)
        result = detect_entropy_spike(probs, probs, spike_multiplier=2.0)
        assert isinstance(result, EntropyDrift)
        assert not result.is_flagged

    def test_spike_detected(self) -> None:
        confident = np.zeros((100, 8))
        confident[:, 0] = 0.95
        confident[:, 1:] = 0.05 / 7

        uniform = np.ones((100, 8)) / 8

        result = detect_entropy_spike(confident, uniform, spike_multiplier=1.5)
        assert result.is_flagged
        assert result.ratio > 1.5

    def test_empty_arrays(self) -> None:
        result = detect_entropy_spike(np.empty((0, 8)), np.empty((0, 8)))
        assert not result.is_flagged


# ---------------------------------------------------------------------------
# detect_class_shift
# ---------------------------------------------------------------------------


class TestDetectClassShift:
    def test_no_shift(self) -> None:
        labels = ["Build"] * 30 + ["Debug"] * 30 + ["Write"] * 40
        result = detect_class_shift(labels, labels, threshold=0.15)
        assert isinstance(result, ClassShiftResult)
        assert not result.is_flagged

    def test_shift_detected(self) -> None:
        ref = ["Build"] * 50 + ["Debug"] * 50
        cur = ["Build"] * 20 + ["Debug"] * 80
        result = detect_class_shift(ref, cur, threshold=0.15)
        assert result.is_flagged
        assert "Build" in result.shifted_classes or "Debug" in result.shifted_classes

    def test_new_class_counts_as_shift(self) -> None:
        ref = ["Build"] * 100
        cur = ["Build"] * 70 + ["Meet"] * 30
        result = detect_class_shift(ref, cur, threshold=0.15)
        assert result.is_flagged
        assert "Meet" in result.shifted_classes

    def test_empty_labels(self) -> None:
        result = detect_class_shift([], [], threshold=0.15)
        assert not result.is_flagged
