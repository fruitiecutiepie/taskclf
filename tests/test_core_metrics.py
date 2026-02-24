"""Tests for evaluation metrics helpers.

Covers: TC-EVAL-002 (confusion matrix shape), TC-EVAL-003 (missing classes),
TC-EVAL-004 (class imbalance reporting).
"""

from __future__ import annotations

from taskclf.core.metrics import class_distribution, compute_metrics, confusion_matrix_df


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
