"""Tests for evaluation metrics helpers.

Covers: TC-EVAL-002 (confusion matrix shape), TC-EVAL-003 (missing classes),
TC-EVAL-004 (class imbalance reporting), TC-EVAL-005 (weighted_f1),
TC-EVAL-006 (per_user_metrics), TC-EVAL-007 (calibration curves),
TC-EVAL-008 (user stratification report).
"""

from __future__ import annotations

import numpy as np

from taskclf.core.metrics import (
    calibration_curve_data,
    class_distribution,
    compute_metrics,
    confusion_matrix_df,
    per_user_metrics,
    user_stratification_report,
)


class TestConfusionMatrixShape:
    """TC-EVAL-002: confusion matrix shape matches label set."""

    def test_square_shape_matches_label_names(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Write", "Build", "BreakIdle"]
        y_pred = ["Build", "Build", "Build", "BreakIdle"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert df.shape == (len(labels), len(labels))

    def test_row_col_labels_match_label_names(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Write"]
        y_pred = ["Build", "Build"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert list(df.index) == labels
        assert list(df.columns) == labels

    def test_all_eight_labels(self) -> None:
        labels = sorted([
            "Build", "Debug", "Review", "Write",
            "ReadResearch", "Communicate", "Meet", "BreakIdle",
        ])
        y_true = ["Build", "BreakIdle"]
        y_pred = ["Build", "BreakIdle"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert df.shape == (8, 8)


class TestMacroF1MissingClasses:
    """TC-EVAL-003: macro-F1 does not crash when a class is absent from predictions."""

    def test_missing_class_in_predictions(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Build", "Write"]
        y_pred = ["Build", "Build", "Build"]

        result = compute_metrics(y_true, y_pred, labels)
        assert "macro_f1" in result
        assert isinstance(result["macro_f1"], float)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_single_class_present(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Build", "Build"]
        y_pred = ["Build", "Build", "Build"]

        result = compute_metrics(y_true, y_pred, labels)
        assert result["macro_f1"] >= 0.0

    def test_confusion_matrix_in_result(self) -> None:
        labels = ["Build", "BreakIdle"]
        y_true = ["Build", "BreakIdle"]
        y_pred = ["Build", "Build"]

        result = compute_metrics(y_true, y_pred, labels)
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        assert len(cm) == len(labels)
        assert len(cm[0]) == len(labels)


class TestClassDistribution:
    """TC-EVAL-004: class imbalance is reported via per-class counts and fractions."""

    def test_per_class_counts_are_correct(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Build", "Build", "Write", "BreakIdle"]

        dist = class_distribution(y_true, labels)
        assert dist["Build"]["count"] == 3
        assert dist["Write"]["count"] == 1
        assert dist["BreakIdle"]["count"] == 1

    def test_fractions_sum_to_one(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Build", "Write"]

        dist = class_distribution(y_true, labels)
        total_frac = sum(d["fraction"] for d in dist.values())
        assert abs(total_frac - 1.0) < 0.01

    def test_missing_class_has_zero_count_and_fraction(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Build"]

        dist = class_distribution(y_true, labels)
        assert dist["BreakIdle"]["count"] == 0
        assert dist["BreakIdle"]["fraction"] == 0.0

    def test_empty_y_true_returns_all_zeros(self) -> None:
        labels = ["Build", "BreakIdle"]
        dist = class_distribution([], labels)
        for label in labels:
            assert dist[label]["count"] == 0
            assert dist[label]["fraction"] == 0.0


class TestWeightedF1InComputeMetrics:
    """TC-EVAL-005: compute_metrics includes weighted_f1."""

    def test_weighted_f1_present(self) -> None:
        labels = ["Build", "Write", "BreakIdle"]
        y_true = ["Build", "Write", "Build", "BreakIdle"]
        y_pred = ["Build", "Build", "Build", "BreakIdle"]

        result = compute_metrics(y_true, y_pred, labels)
        assert "weighted_f1" in result
        assert isinstance(result["weighted_f1"], float)
        assert 0.0 <= result["weighted_f1"] <= 1.0

    def test_weighted_f1_differs_from_macro_on_imbalanced(self) -> None:
        labels = ["Build", "Write"]
        y_true = ["Build"] * 10 + ["Write"] * 2
        y_pred = ["Build"] * 10 + ["Build"] * 2

        result = compute_metrics(y_true, y_pred, labels)
        assert result["weighted_f1"] != result["macro_f1"]


class TestPerUserMetrics:
    """TC-EVAL-006: per_user_metrics returns per-user breakdown."""

    def test_returns_entry_per_user(self) -> None:
        labels = ["Build", "Write"]
        y_true = ["Build", "Write", "Build", "Write"]
        y_pred = ["Build", "Build", "Build", "Write"]
        user_ids = ["u1", "u1", "u2", "u2"]

        result = per_user_metrics(y_true, y_pred, user_ids, labels)
        assert "u1" in result
        assert "u2" in result
        assert "macro_f1" in result["u1"]
        assert "count" in result["u1"]

    def test_single_user_metrics(self) -> None:
        labels = ["Build", "Write"]
        y_true = ["Build", "Build"]
        y_pred = ["Build", "Build"]
        user_ids = ["u1", "u1"]

        result = per_user_metrics(y_true, y_pred, user_ids, labels)
        assert result["u1"]["macro_f1"] >= 0.0

    def test_per_class_f1_keys_present(self) -> None:
        labels = ["Build", "Write"]
        y_true = ["Build", "Write"]
        y_pred = ["Build", "Write"]
        user_ids = ["u1", "u1"]

        result = per_user_metrics(y_true, y_pred, user_ids, labels)
        assert "Build_f1" in result["u1"]
        assert "Write_f1" in result["u1"]


class TestCalibrationCurveData:
    """TC-EVAL-007: calibration_curve_data returns valid bin data."""

    def test_returns_entry_per_class(self) -> None:
        labels = ["A", "B"]
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.3, 0.7],
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9],
        ])

        result = calibration_curve_data(y_true, y_proba, labels, n_bins=3)
        assert "A" in result
        assert "B" in result
        assert "fraction_of_positives" in result["A"]
        assert "mean_predicted_value" in result["A"]

    def test_empty_class_returns_empty_lists(self) -> None:
        labels = ["A", "B"]
        y_true = np.array([0, 0, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])

        result = calibration_curve_data(y_true, y_proba, labels, n_bins=2)
        assert result["B"]["fraction_of_positives"] == []

    def test_values_are_bounded(self) -> None:
        labels = ["A", "B"]
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.integers(0, 2, size=n)
        raw = rng.random((n, 2))
        y_proba = raw / raw.sum(axis=1, keepdims=True)

        result = calibration_curve_data(y_true, y_proba, labels, n_bins=5)
        for cls in labels:
            for v in result[cls]["fraction_of_positives"]:
                assert 0.0 <= v <= 1.0
            for v in result[cls]["mean_predicted_value"]:
                assert 0.0 <= v <= 1.0


class TestUserStratificationReport:
    """TC-EVAL-008: user stratification report flags dominant users."""

    def test_no_warnings_when_balanced(self) -> None:
        user_ids = ["u1"] * 50 + ["u2"] * 50
        labels = ["Build"] * 50 + ["Write"] * 50

        result = user_stratification_report(user_ids, labels, ["Build", "Write"])
        assert result["warnings"] == []
        assert result["user_count"] == 2

    def test_flags_dominant_user(self) -> None:
        user_ids = ["u1"] * 90 + ["u2"] * 10
        labels = ["Build"] * 90 + ["Write"] * 10

        result = user_stratification_report(
            user_ids, labels, ["Build", "Write"], dominance_threshold=0.5,
        )
        assert len(result["warnings"]) == 1
        assert "u1" in result["warnings"][0]

    def test_per_user_label_distribution(self) -> None:
        user_ids = ["u1", "u1", "u2", "u2"]
        labels = ["Build", "Build", "Write", "Write"]

        result = user_stratification_report(user_ids, labels, ["Build", "Write"])
        assert result["per_user"]["u1"]["label_distribution"]["Build"] == 2
        assert result["per_user"]["u2"]["label_distribution"]["Write"] == 2
        assert result["total_rows"] == 4
