"""Tests for reject-threshold support in batch inference.

Covers: TC-REJECT-001 (predict_labels with reject), TC-REJECT-002
(backward compat), TC-REJECT-003 (run_batch_inference reject outputs),
TC-REJECT-004 (write_predictions_csv with reject columns).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import train_lgbm


def _build_model_and_data():
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

    labeled = project_blocks_to_windows(features_df, spans)
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

    model, _, _, _, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=5, class_weight="balanced",
    )
    return model, cat_encoders, labeled


@pytest.fixture(scope="module")
def artifacts():
    return _build_model_and_data()


# ---------------------------------------------------------------------------
# TC-REJECT-001: predict_labels with reject threshold
# ---------------------------------------------------------------------------

class TestPredictLabelsReject:
    def test_high_threshold_produces_rejections(self, artifacts) -> None:
        from sklearn.preprocessing import LabelEncoder
        from taskclf.infer.batch import predict_labels

        model, cat_encoders, labeled_df = artifacts
        le = LabelEncoder()
        le.fit(sorted(LABEL_SET_V1))

        labels = predict_labels(
            model, labeled_df, le, cat_encoders=cat_encoders,
            reject_threshold=0.99,
        )

        assert MIXED_UNKNOWN in labels

    def test_low_threshold_produces_no_rejections(self, artifacts) -> None:
        from sklearn.preprocessing import LabelEncoder
        from taskclf.infer.batch import predict_labels

        model, cat_encoders, labeled_df = artifacts
        le = LabelEncoder()
        le.fit(sorted(LABEL_SET_V1))

        labels = predict_labels(
            model, labeled_df, le, cat_encoders=cat_encoders,
            reject_threshold=0.01,
        )

        assert MIXED_UNKNOWN not in labels


# ---------------------------------------------------------------------------
# TC-REJECT-002: backward compatibility (no reject_threshold)
# ---------------------------------------------------------------------------

class TestPredictLabelsBackwardCompat:
    def test_none_threshold_never_rejects(self, artifacts) -> None:
        from sklearn.preprocessing import LabelEncoder
        from taskclf.infer.batch import predict_labels

        model, cat_encoders, labeled_df = artifacts
        le = LabelEncoder()
        le.fit(sorted(LABEL_SET_V1))

        labels = predict_labels(
            model, labeled_df, le, cat_encoders=cat_encoders,
            reject_threshold=None,
        )

        assert MIXED_UNKNOWN not in labels


# ---------------------------------------------------------------------------
# TC-REJECT-003: run_batch_inference returns reject metadata
# ---------------------------------------------------------------------------

class TestRunBatchInferenceReject:
    def test_returns_result_object(self, artifacts) -> None:
        from taskclf.infer.batch import BatchInferenceResult, run_batch_inference

        model, cat_encoders, labeled_df = artifacts

        result = run_batch_inference(
            model, labeled_df,
            cat_encoders=cat_encoders,
            reject_threshold=0.55,
        )
        assert isinstance(result, BatchInferenceResult)

        assert len(result.smoothed_labels) == len(labeled_df)
        assert result.confidences.shape == (len(labeled_df),)
        assert result.is_rejected.shape == (len(labeled_df),)
        assert result.is_rejected.dtype == bool

    def test_high_threshold_flags_rejections(self, artifacts) -> None:
        from taskclf.infer.batch import run_batch_inference

        model, cat_encoders, labeled_df = artifacts
        result = run_batch_inference(
            model, labeled_df,
            cat_encoders=cat_encoders,
            reject_threshold=0.99,
        )
        assert result.is_rejected.any()

    def test_no_threshold_means_no_rejections(self, artifacts) -> None:
        from taskclf.infer.batch import run_batch_inference

        model, cat_encoders, labeled_df = artifacts
        result = run_batch_inference(
            model, labeled_df,
            cat_encoders=cat_encoders,
            reject_threshold=None,
        )
        assert not result.is_rejected.any()


# ---------------------------------------------------------------------------
# TC-REJECT-004: write_predictions_csv includes reject columns
# ---------------------------------------------------------------------------

class TestWritePredictionsCsvReject:
    def test_csv_includes_confidence_and_rejected(self, tmp_path, artifacts) -> None:
        from taskclf.infer.batch import run_batch_inference, write_predictions_csv

        model, cat_encoders, labeled_df = artifacts
        result = run_batch_inference(
            model, labeled_df,
            cat_encoders=cat_encoders,
            reject_threshold=0.55,
        )

        csv_path = write_predictions_csv(
            labeled_df, result.smoothed_labels, tmp_path / "preds.csv",
            confidences=result.confidences, is_rejected=result.is_rejected,
        )
        out = pd.read_csv(csv_path)
        assert "confidence" in out.columns
        assert "is_rejected" in out.columns
        assert len(out) == len(labeled_df)

    def test_csv_without_reject_columns(self, tmp_path, artifacts) -> None:
        from taskclf.infer.batch import run_batch_inference, write_predictions_csv

        model, cat_encoders, labeled_df = artifacts
        result = run_batch_inference(
            model, labeled_df, cat_encoders=cat_encoders,
        )

        csv_path = write_predictions_csv(
            labeled_df, result.smoothed_labels, tmp_path / "preds_basic.csv",
        )
        out = pd.read_csv(csv_path)
        assert "confidence" not in out.columns
        assert "is_rejected" not in out.columns
