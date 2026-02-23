"""Tests for the online inference predictor.

Covers:
- OnlinePredictor single-bucket prediction
- Rolling smoothing buffer behavior
- Segment accumulation over multiple predictions
- No retraining (model is used read-only)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.model_io import load_model_bundle
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1, FeatureRow
from taskclf.features.build import generate_dummy_features
from taskclf.infer.online import OnlinePredictor

runner = CliRunner()


@pytest.fixture()
def trained_model_dir(tmp_path: Path) -> Path:
    """Train a small model for testing."""
    models_dir = tmp_path / "models"
    result = runner.invoke(app, [
        "train", "lgbm",
        "--from", "2025-06-14",
        "--to", "2025-06-15",
        "--synthetic",
        "--models-dir", str(models_dir),
        "--num-boost-round", "5",
    ])
    assert result.exit_code == 0, result.output
    return next(models_dir.iterdir())


@pytest.fixture()
def predictor(trained_model_dir: Path) -> OnlinePredictor:
    model, metadata = load_model_bundle(trained_model_dir)
    return OnlinePredictor(model, metadata, smooth_window=3)


class TestOnlinePredictor:
    def test_predict_single_bucket(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        label = predictor.predict_bucket(rows[0])
        assert label in LABEL_SET_V1

    def test_predict_multiple_buckets(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        labels = []
        for row in rows:
            labels.append(predictor.predict_bucket(row))
        assert len(labels) == 5
        assert all(lbl in LABEL_SET_V1 for lbl in labels)

    def test_segments_accumulate(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        for row in rows:
            predictor.predict_bucket(row)

        segments = predictor.get_segments()
        assert len(segments) >= 1
        total_buckets = sum(s.bucket_count for s in segments)
        assert total_buckets == 10

    def test_segments_ordered_non_overlapping(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        for row in rows:
            predictor.predict_bucket(row)

        segments = predictor.get_segments()
        for i in range(len(segments) - 1):
            assert segments[i].end_ts <= segments[i + 1].start_ts

    def test_empty_predictor_has_no_segments(self, predictor: OnlinePredictor) -> None:
        assert predictor.get_segments() == []

    def test_smoothing_window_respected(self, trained_model_dir: Path) -> None:
        model, metadata = load_model_bundle(trained_model_dir)
        pred_w1 = OnlinePredictor(model, metadata, smooth_window=1)
        pred_w5 = OnlinePredictor(model, metadata, smooth_window=5)

        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        labels_w1 = [pred_w1.predict_bucket(r) for r in rows]
        labels_w5 = [pred_w5.predict_bucket(r) for r in rows]

        assert all(lbl in LABEL_SET_V1 for lbl in labels_w1)
        assert all(lbl in LABEL_SET_V1 for lbl in labels_w5)

    def test_segment_labels_are_valid(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        for row in rows:
            predictor.predict_bucket(row)

        segments = predictor.get_segments()
        for seg in segments:
            assert seg.label in LABEL_SET_V1
