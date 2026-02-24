"""Tests for the full evaluation pipeline and class-weight computation.

Covers: TC-EVAL-010 (class weights), TC-EVAL-011 (evaluate_model report),
TC-EVAL-012 (acceptance checks), TC-EVAL-013 (write artifacts).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import compute_sample_weights, train_lgbm


def _build_labeled_df() -> pd.DataFrame:
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=20))

    features_df = pd.DataFrame([r.model_dump() for r in all_rows])

    spans: list[LabelSpan] = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day)
        spans.extend([
            LabelSpan(start_ts=base.replace(hour=9), end_ts=base.replace(hour=12),
                       label="Build", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=12), end_ts=base.replace(hour=14),
                       label="Write", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=14), end_ts=base.replace(hour=16),
                       label="Communicate", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=16), end_ts=base.replace(hour=17),
                       label="BreakIdle", provenance="test"),
        ])

    return project_blocks_to_windows(features_df, spans)


# ---------------------------------------------------------------------------
# TC-EVAL-010: class weight computation
# ---------------------------------------------------------------------------


class TestComputeSampleWeights:
    def test_balanced_produces_correct_shape(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 2])
        weights = compute_sample_weights(y, method="balanced")
        assert weights is not None
        assert weights.shape == y.shape

    def test_balanced_minority_gets_higher_weight(self) -> None:
        y = np.array([0, 0, 0, 0, 1])
        weights = compute_sample_weights(y, method="balanced")
        assert weights is not None
        assert weights[4] > weights[0]

    def test_balanced_equal_classes_give_equal_weights(self) -> None:
        y = np.array([0, 0, 1, 1])
        weights = compute_sample_weights(y, method="balanced")
        assert weights is not None
        np.testing.assert_allclose(weights, 1.0)

    def test_none_method_returns_none(self) -> None:
        y = np.array([0, 1, 2])
        assert compute_sample_weights(y, method="none") is None


# ---------------------------------------------------------------------------
# TC-EVAL-011: evaluate_model returns a complete EvaluationReport
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    labeled = _build_labeled_df()
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)
    test_df = labeled.iloc[splits["test"]].reset_index(drop=True)

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=5, class_weight="balanced",
    )

    return {
        "model": model,
        "cat_encoders": cat_encoders,
        "test_df": test_df,
        "labeled_df": labeled,
        "splits": splits,
    }


class TestEvaluateModel:
    def test_report_has_required_fields(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )

        assert 0.0 <= report.macro_f1 <= 1.0
        assert 0.0 <= report.weighted_f1 <= 1.0
        assert isinstance(report.per_class, dict)
        assert isinstance(report.per_user, dict)
        assert isinstance(report.calibration, dict)
        assert isinstance(report.acceptance_checks, dict)
        assert isinstance(report.confusion_matrix, list)
        assert 0.0 <= report.reject_rate <= 1.0

    def test_per_class_covers_all_labels(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )

        for lbl in sorted(LABEL_SET_V1):
            assert lbl in report.per_class
            assert "precision" in report.per_class[lbl]
            assert "recall" in report.per_class[lbl]
            assert "f1" in report.per_class[lbl]

    def test_calibration_covers_all_labels(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )

        for lbl in sorted(LABEL_SET_V1):
            assert lbl in report.calibration
            assert "fraction_of_positives" in report.calibration[lbl]
            assert "mean_predicted_value" in report.calibration[lbl]

    def test_seen_unseen_f1_with_holdout(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        test_df = trained_artifacts["test_df"]
        users = test_df["user_id"].unique().tolist()
        if len(users) < 2:
            pytest.skip("Need at least 2 users for holdout test")

        holdout = users[:1]
        report = evaluate_model(
            trained_artifacts["model"],
            test_df,
            cat_encoders=trained_artifacts["cat_encoders"],
            holdout_users=holdout,
        )

        assert report.seen_user_f1 is not None or report.unseen_user_f1 is not None


# ---------------------------------------------------------------------------
# TC-EVAL-012: acceptance checks
# ---------------------------------------------------------------------------


class TestAcceptanceChecks:
    def test_acceptance_checks_are_booleans(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )

        for check_name, value in report.acceptance_checks.items():
            assert isinstance(value, bool), f"{check_name} is not bool"

    def test_acceptance_details_have_matching_keys(self, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )

        for key in report.acceptance_checks:
            assert key in report.acceptance_details


# ---------------------------------------------------------------------------
# TC-EVAL-013: write evaluation artifacts
# ---------------------------------------------------------------------------


class TestWriteEvaluationArtifacts:
    def test_writes_evaluation_json(self, tmp_path, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model, write_evaluation_artifacts

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )
        paths = write_evaluation_artifacts(report, tmp_path)
        assert paths["evaluation"].exists()
        data = json.loads(paths["evaluation"].read_text())
        assert "macro_f1" in data
        assert "acceptance_checks" in data

    def test_writes_calibration_json(self, tmp_path, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model, write_evaluation_artifacts

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )
        paths = write_evaluation_artifacts(report, tmp_path)
        assert paths["calibration"].exists()
        data = json.loads(paths["calibration"].read_text())
        assert isinstance(data, dict)

    def test_writes_confusion_matrix_csv(self, tmp_path, trained_artifacts) -> None:
        from taskclf.train.evaluate import evaluate_model, write_evaluation_artifacts

        report = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["test_df"],
            cat_encoders=trained_artifacts["cat_encoders"],
        )
        paths = write_evaluation_artifacts(report, tmp_path)
        assert paths["confusion_matrix"].exists()
        cm = pd.read_csv(paths["confusion_matrix"], index_col=0)
        assert cm.shape[0] == cm.shape[1]


# ---------------------------------------------------------------------------
# TC-EVAL-014: train_lgbm with class_weight="balanced" vs "none"
# ---------------------------------------------------------------------------


class TestTrainLgbmClassWeight:
    def test_balanced_records_method_in_params(self) -> None:
        labeled = _build_labeled_df()
        splits = split_by_time(labeled)
        train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
        val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

        _, _, _, params, _ = train_lgbm(
            train_df, val_df, num_boost_round=3, class_weight="balanced",
        )
        assert params["class_weight_method"] == "balanced"

    def test_none_records_method_in_params(self) -> None:
        labeled = _build_labeled_df()
        splits = split_by_time(labeled)
        train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
        val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

        _, _, _, params, _ = train_lgbm(
            train_df, val_df, num_boost_round=3, class_weight="none",
        )
        assert params["class_weight_method"] == "none"
