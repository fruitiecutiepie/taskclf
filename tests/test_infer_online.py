"""Tests for the online inference predictor.

Covers:
- OnlinePredictor single-bucket prediction (returns WindowPrediction)
- Rolling smoothing buffer behavior
- Segment accumulation over multiple predictions
- No retraining (model is used read-only)
- Session tracking across poll cycles (via build_features_from_aw_events)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.cli.main import app
from taskclf.core.model_io import load_model_bundle
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.types import LABEL_SET_V1, FeatureRow

_VALID_LABELS = LABEL_SET_V1 | {MIXED_UNKNOWN}
from taskclf.features.build import build_features_from_aw_events, generate_dummy_features
from taskclf.infer.online import OnlinePredictor
from taskclf.infer.prediction import WindowPrediction

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
    model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
    return OnlinePredictor(model, metadata, cat_encoders=cat_encoders, smooth_window=3)


class TestOnlinePredictor:
    def test_predict_single_bucket(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        pred = predictor.predict_bucket(rows[0])
        assert isinstance(pred, WindowPrediction)
        assert pred.mapped_label_name in _VALID_LABELS

    def test_predict_multiple_buckets(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        preds = [predictor.predict_bucket(row) for row in rows]
        assert len(preds) == 5
        assert all(isinstance(p, WindowPrediction) for p in preds)
        assert all(p.mapped_label_name in _VALID_LABELS for p in preds)

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
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred_w1 = OnlinePredictor(model, metadata, cat_encoders=cat_encoders, smooth_window=1)
        pred_w5 = OnlinePredictor(model, metadata, cat_encoders=cat_encoders, smooth_window=5)

        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        preds_w1 = [pred_w1.predict_bucket(r) for r in rows]
        preds_w5 = [pred_w5.predict_bucket(r) for r in rows]

        assert all(p.mapped_label_name in _VALID_LABELS for p in preds_w1)
        assert all(p.mapped_label_name in _VALID_LABELS for p in preds_w5)

    def test_segment_labels_are_valid(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        for row in rows:
            predictor.predict_bucket(row)

        segments = predictor.get_segments()
        for seg in segments:
            assert seg.label in _VALID_LABELS


class TestOnlineSessionTracking:
    """Verify that the online loop's session_start mechanism works.

    The online loop passes session_start to build_features_from_aw_events,
    resetting it when an idle gap is detected between poll cycles.
    """

    @staticmethod
    def _ev(ts: dt.datetime, duration: float = 30.0) -> AWEvent:
        return AWEvent(
            timestamp=ts,
            duration_seconds=duration,
            app_id="org.mozilla.firefox",
            window_title_hash="hash",
            is_browser=True,
            is_editor=False,
            is_terminal=False,
            app_category="browser",
        )

    def test_session_start_persists_across_polls(self) -> None:
        """Consecutive poll windows with no idle gap share session_start."""
        session_start = dt.datetime(2026, 2, 23, 10, 0, 0)

        poll_1_events = [self._ev(dt.datetime(2026, 2, 23, 10, 0, 0))]
        poll_2_events = [self._ev(dt.datetime(2026, 2, 23, 10, 1, 0))]

        rows_1 = build_features_from_aw_events(poll_1_events, session_start=session_start)
        rows_2 = build_features_from_aw_events(poll_2_events, session_start=session_start)

        assert rows_1[0].session_length_so_far == 0.0
        assert rows_2[0].session_length_so_far == 1.0

    def test_session_start_resets_after_gap(self) -> None:
        """After an idle gap the online loop would set a new session_start."""
        new_session_start = dt.datetime(2026, 2, 23, 11, 0, 0)
        events = [self._ev(dt.datetime(2026, 2, 23, 11, 2, 0))]

        rows = build_features_from_aw_events(events, session_start=new_session_start)
        assert rows[0].session_length_so_far == 2.0
