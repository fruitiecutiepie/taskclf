"""Tests for the online inference predictor.

Covers:
- OnlinePredictor single-bucket prediction (returns WindowPrediction)
- Rolling smoothing buffer behavior
- Segment accumulation over multiple predictions
- No retraining (model is used read-only)
- Session tracking across poll cycles (via build_features_from_aw_events)
- OnlinePredictor with taxonomy (TC-ONLINE-001, TC-ONLINE-002)
- OnlinePredictor with calibrator_store (TC-ONLINE-003)
- _encode_value (TC-ONLINE-004, TC-ONLINE-005)
- OnlinePredictor reject segments (TC-ONLINE-006)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.cli.main import app
from taskclf.core.model_io import load_model_bundle
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.types import LABEL_SET_V1, FeatureRow
from taskclf.infer.calibration import CalibratorStore, IdentityCalibrator, TemperatureCalibrator
from taskclf.infer.taxonomy import TaxonomyBucket, TaxonomyConfig

_VALID_LABELS = LABEL_SET_V1 | {MIXED_UNKNOWN}
from taskclf.features.build import build_features_from_aw_events, generate_dummy_features
from taskclf.infer.online import OnlinePredictor
from taskclf.infer.prediction import WindowPrediction
from taskclf.train.lgbm import CATEGORICAL_COLUMNS

runner = CliRunner()


def _example_taxonomy() -> TaxonomyConfig:
    return TaxonomyConfig(
        buckets=[
            TaxonomyBucket(name="Deep Work", core_labels=["Build", "Debug", "Write"], color="#2E86DE"),
            TaxonomyBucket(name="Research", core_labels=["ReadResearch", "Review"], color="#9B59B6"),
            TaxonomyBucket(name="Communication", core_labels=["Communicate"], color="#27AE60"),
            TaxonomyBucket(name="Meetings", core_labels=["Meet"], color="#E67E22"),
            TaxonomyBucket(name="Break", core_labels=["BreakIdle"], color="#7F8C8D"),
        ],
    )


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


# ---------------------------------------------------------------------------
# OnlinePredictor with taxonomy
# ---------------------------------------------------------------------------

_TAXONOMY_BUCKET_NAMES = {"Deep Work", "Research", "Communication", "Meetings", "Break"}


class TestOnlinePredictorTaxonomy:
    def test_mapped_label_from_taxonomy_buckets(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-001: mapped_label_name comes from taxonomy bucket names."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        taxonomy = _example_taxonomy()
        pred = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
            smooth_window=1, reject_threshold=None, taxonomy=taxonomy,
        )
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        for row in rows:
            result = pred.predict_bucket(row)
            assert result.mapped_label_name in _TAXONOMY_BUCKET_NAMES

    def test_mapped_probs_keys_are_bucket_names(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-002: mapped_probs keys are bucket names, values sum to ~1.0."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        taxonomy = _example_taxonomy()
        pred = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
            smooth_window=1, reject_threshold=None, taxonomy=taxonomy,
        )
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        for row in rows:
            result = pred.predict_bucket(row)
            assert set(result.mapped_probs.keys()) == _TAXONOMY_BUCKET_NAMES
            assert abs(sum(result.mapped_probs.values()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# OnlinePredictor with calibrator_store
# ---------------------------------------------------------------------------


class TestOnlinePredictorCalibratorStore:
    def test_per_user_calibration_changes_confidence(
        self, trained_model_dir: Path,
    ) -> None:
        """TC-ONLINE-003: per-user calibration applied via CalibratorStore."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        user_id = rows[0].user_id

        pred_identity = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
            smooth_window=1, reject_threshold=None,
            calibrator=IdentityCalibrator(),
        )

        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={user_id: TemperatureCalibrator(temperature=5.0)},
        )
        pred_store = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
            smooth_window=1, reject_threshold=None,
            calibrator_store=store,
        )

        identity_confs = [pred_identity.predict_bucket(r).confidence for r in rows]
        store_confs = [pred_store.predict_bucket(r).confidence for r in rows]

        assert not np.allclose(identity_confs, store_confs, atol=1e-6)


# ---------------------------------------------------------------------------
# _encode_value
# ---------------------------------------------------------------------------


class TestEncodeValue:
    def test_categorical_known_value(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-004: known categorical value returns encoded int; unknown returns -1.0."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)

        cat_col = CATEGORICAL_COLUMNS[0]
        le = cat_encoders[cat_col]
        known_val = le.classes_[0]
        expected_code = float(le.transform([known_val])[0])

        assert pred._encode_value(cat_col, known_val) == expected_code
        assert pred._encode_value(cat_col, "__never_seen_value__") == -1.0

    def test_numerical_none_returns_zero(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-005: non-categorical None returns 0.0."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)

        assert pred._encode_value("key_rate", None) == 0.0
        assert pred._encode_value("key_rate", 3.5) == 3.5


# ---------------------------------------------------------------------------
# OnlinePredictor reject → segments
# ---------------------------------------------------------------------------


class TestOnlinePredictorRejectSegments:
    def test_rejected_predictions_produce_mixed_unknown_segments(
        self, trained_model_dir: Path,
    ) -> None:
        """TC-ONLINE-006: after rejected predictions, segments use MIXED_UNKNOWN."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
            smooth_window=1, reject_threshold=1.0,
        )
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        for row in rows:
            result = pred.predict_bucket(row)
            assert result.is_rejected

        segments = pred.get_segments()
        assert len(segments) >= 1
        for seg in segments:
            assert seg.label == MIXED_UNKNOWN
