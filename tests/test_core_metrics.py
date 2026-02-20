"""Tests for evaluation metrics helpers.

Covers: TC-EVAL-002 (confusion matrix shape), TC-EVAL-003 (missing classes),
TC-EVAL-004 (class imbalance reporting).
"""

from __future__ import annotations

from taskclf.core.metrics import class_distribution, compute_metrics, confusion_matrix_df


class TestConfusionMatrixShape:
    """TC-EVAL-002: confusion matrix shape matches label set."""

    def test_square_shape_matches_label_names(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "writing_docs", "coding", "break_idle"]
        y_pred = ["coding", "coding", "coding", "break_idle"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert df.shape == (len(labels), len(labels))

    def test_row_col_labels_match_label_names(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "writing_docs"]
        y_pred = ["coding", "coding"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert list(df.index) == labels
        assert list(df.columns) == labels

    def test_all_six_labels(self) -> None:
        labels = sorted([
            "coding", "writing_docs", "messaging_email",
            "browsing_research", "meetings_calls", "break_idle",
        ])
        y_true = ["coding", "break_idle"]
        y_pred = ["coding", "break_idle"]

        df = confusion_matrix_df(y_true, y_pred, labels)
        assert df.shape == (6, 6)


class TestMacroF1MissingClasses:
    """TC-EVAL-003: macro-F1 does not crash when a class is absent from predictions."""

    def test_missing_class_in_predictions(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "coding", "writing_docs"]
        y_pred = ["coding", "coding", "coding"]

        result = compute_metrics(y_true, y_pred, labels)
        assert "macro_f1" in result
        assert isinstance(result["macro_f1"], float)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_single_class_present(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "coding", "coding"]
        y_pred = ["coding", "coding", "coding"]

        result = compute_metrics(y_true, y_pred, labels)
        assert result["macro_f1"] >= 0.0

    def test_confusion_matrix_in_result(self) -> None:
        labels = ["coding", "break_idle"]
        y_true = ["coding", "break_idle"]
        y_pred = ["coding", "coding"]

        result = compute_metrics(y_true, y_pred, labels)
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        assert len(cm) == len(labels)
        assert len(cm[0]) == len(labels)


class TestClassDistribution:
    """TC-EVAL-004: class imbalance is reported via per-class counts and fractions."""

    def test_per_class_counts_are_correct(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "coding", "coding", "writing_docs", "break_idle"]

        dist = class_distribution(y_true, labels)
        assert dist["coding"]["count"] == 3
        assert dist["writing_docs"]["count"] == 1
        assert dist["break_idle"]["count"] == 1

    def test_fractions_sum_to_one(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "coding", "writing_docs"]

        dist = class_distribution(y_true, labels)
        total_frac = sum(d["fraction"] for d in dist.values())
        assert abs(total_frac - 1.0) < 0.01

    def test_missing_class_has_zero_count_and_fraction(self) -> None:
        labels = ["coding", "writing_docs", "break_idle"]
        y_true = ["coding", "coding"]

        dist = class_distribution(y_true, labels)
        assert dist["break_idle"]["count"] == 0
        assert dist["break_idle"]["fraction"] == 0.0

    def test_empty_y_true_returns_all_zeros(self) -> None:
        labels = ["coding", "break_idle"]
        dist = class_distribution([], labels)
        for label in labels:
            assert dist[label]["count"] == 0
            assert dist[label]["fraction"] == 0.0
