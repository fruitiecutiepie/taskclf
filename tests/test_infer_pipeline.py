"""Tests for the online inference pipeline (TODO 8).

Covers:
- WindowPrediction contract validation
- Calibrator protocol and implementations
- Enriched CSV output columns
- Hysteresis-based segment merging
- OnlinePredictor returns WindowPrediction with full fields
- Batch inference includes core_probs and hysteresis
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.defaults import MIXED_UNKNOWN, MIN_BLOCK_DURATION_SECONDS
from taskclf.core.model_io import load_model_bundle
from taskclf.core.types import LABEL_SET_V1
from taskclf.features.build import generate_dummy_features
from taskclf.infer.calibration import (
    Calibrator,
    IdentityCalibrator,
    TemperatureCalibrator,
    load_calibrator,
    save_calibrator,
)
from taskclf.infer.online import OnlinePredictor, _append_prediction_csv
from taskclf.infer.prediction import WindowPrediction
from taskclf.infer.smooth import Segment, merge_short_segments, segmentize

runner = CliRunner()

_SORTED_LABELS = sorted(LABEL_SET_V1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    models_dir = tmp_path_factory.mktemp("models")
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
    return OnlinePredictor(
        model, metadata,
        cat_encoders=cat_encoders,
        smooth_window=3,
        reject_threshold=0.55,
    )


# ---------------------------------------------------------------------------
# WindowPrediction contract
# ---------------------------------------------------------------------------


class TestWindowPrediction:
    def test_valid_prediction(self) -> None:
        pred = WindowPrediction(
            user_id="u1",
            bucket_start_ts=dt.datetime(2026, 1, 1, 10, 0),
            core_label_id=0,
            core_label_name="Build",
            core_probs=[0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            confidence=0.6,
            is_rejected=False,
            mapped_label_name="Build",
            mapped_probs={"Build": 0.6, "Other": 0.4},
            model_version="abc123",
        )
        assert pred.schema_version == "features_v1"
        assert pred.label_version == "labels_v1"

    def test_core_probs_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="core_probs must sum to 1.0"):
            WindowPrediction(
                user_id="u1",
                bucket_start_ts=dt.datetime(2026, 1, 1, 10, 0),
                core_label_id=0,
                core_label_name="Build",
                core_probs=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                confidence=0.5,
                is_rejected=False,
                mapped_label_name="Build",
                mapped_probs={"Build": 1.0},
                model_version="abc123",
            )

    def test_mapped_probs_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="mapped_probs must sum to 1.0"):
            WindowPrediction(
                user_id="u1",
                bucket_start_ts=dt.datetime(2026, 1, 1, 10, 0),
                core_label_id=0,
                core_label_name="Build",
                core_probs=[0.125] * 8,
                confidence=0.125,
                is_rejected=False,
                mapped_label_name="Build",
                mapped_probs={"Build": 0.3},
                model_version="abc123",
            )

    def test_core_probs_must_have_8_elements(self) -> None:
        with pytest.raises(ValueError):
            WindowPrediction(
                user_id="u1",
                bucket_start_ts=dt.datetime(2026, 1, 1, 10, 0),
                core_label_id=0,
                core_label_name="Build",
                core_probs=[0.5, 0.5],
                confidence=0.5,
                is_rejected=False,
                mapped_label_name="Build",
                mapped_probs={"Build": 1.0},
                model_version="abc123",
            )


# ---------------------------------------------------------------------------
# Calibrator tests
# ---------------------------------------------------------------------------


class TestIdentityCalibrator:
    def test_identity_preserves_probabilities(self) -> None:
        cal = IdentityCalibrator()
        probs = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.1, 0.1])
        result = cal.calibrate(probs)
        np.testing.assert_array_equal(result, probs)

    def test_identity_satisfies_protocol(self) -> None:
        assert isinstance(IdentityCalibrator(), Calibrator)

    def test_identity_2d(self) -> None:
        cal = IdentityCalibrator()
        probs = np.array([[0.125] * 8, [0.125] * 8])
        result = cal.calibrate(probs)
        np.testing.assert_array_equal(result, probs)


class TestTemperatureCalibrator:
    def test_temperature_one_is_identity(self) -> None:
        cal = TemperatureCalibrator(temperature=1.0)
        probs = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.1, 0.1])
        result = cal.calibrate(probs)
        np.testing.assert_allclose(result, probs, atol=1e-6)

    def test_high_temperature_softens(self) -> None:
        cal = TemperatureCalibrator(temperature=5.0)
        probs = np.array([0.9, 0.05, 0.01, 0.01, 0.01, 0.005, 0.005, 0.01])
        result = cal.calibrate(probs)
        assert result.max() < probs.max()
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_low_temperature_sharpens(self) -> None:
        cal = TemperatureCalibrator(temperature=0.1)
        probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        result = cal.calibrate(probs)
        assert result.max() > probs.max()
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_preserves_sum_2d(self) -> None:
        cal = TemperatureCalibrator(temperature=2.0)
        probs = np.array([[0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025],
                          [0.125] * 8])
        result = cal.calibrate(probs)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TemperatureCalibrator(temperature=0.0)
        with pytest.raises(ValueError, match="positive"):
            TemperatureCalibrator(temperature=-1.0)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(TemperatureCalibrator(1.5), Calibrator)


class TestCalibratorPersistence:
    def test_save_load_identity(self, tmp_path: Path) -> None:
        cal = IdentityCalibrator()
        path = save_calibrator(cal, tmp_path / "cal.json")
        loaded = load_calibrator(path)
        assert isinstance(loaded, IdentityCalibrator)

    def test_save_load_temperature(self, tmp_path: Path) -> None:
        cal = TemperatureCalibrator(temperature=2.5)
        path = save_calibrator(cal, tmp_path / "cal.json")
        loaded = load_calibrator(path)
        assert isinstance(loaded, TemperatureCalibrator)
        assert loaded.temperature == 2.5

    def test_load_unknown_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"type": "fancy"}))
        with pytest.raises(ValueError, match="Unknown calibrator type"):
            load_calibrator(path)


# ---------------------------------------------------------------------------
# OnlinePredictor with WindowPrediction output
# ---------------------------------------------------------------------------


class TestOnlinePredictorWindowPrediction:
    def test_returns_window_prediction(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        pred = predictor.predict_bucket(rows[0])
        assert isinstance(pred, WindowPrediction)

    def test_core_probs_sum_to_one(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=3)
        for row in rows:
            pred = predictor.predict_bucket(row)
            assert abs(sum(pred.core_probs) - 1.0) < 1e-4

    def test_mapped_probs_sum_to_one(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=3)
        for row in rows:
            pred = predictor.predict_bucket(row)
            assert abs(sum(pred.mapped_probs.values()) - 1.0) < 1e-4

    def test_all_fields_populated(self, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        pred = predictor.predict_bucket(rows[0])
        assert pred.user_id == rows[0].user_id
        assert pred.bucket_start_ts == rows[0].bucket_start_ts
        assert 0 <= pred.core_label_id <= 7
        assert pred.core_label_name in _SORTED_LABELS
        assert len(pred.core_probs) == 8
        assert 0.0 <= pred.confidence <= 1.0
        assert isinstance(pred.is_rejected, bool)
        assert pred.mapped_label_name
        assert pred.model_version
        assert pred.schema_version == "features_v1"
        assert pred.label_version == "labels_v1"

    def test_rejected_prediction(self, trained_model_dir: Path) -> None:
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(
            model, metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=0.9999,
        )
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        result = pred.predict_bucket(rows[0])
        assert result.is_rejected is True
        assert result.mapped_label_name == MIXED_UNKNOWN

    def test_calibrator_integration(self, trained_model_dir: Path) -> None:
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)

        pred_identity = OnlinePredictor(
            model, metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            calibrator=IdentityCalibrator(),
        )
        pred_temp = OnlinePredictor(
            model, metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            calibrator=TemperatureCalibrator(temperature=5.0),
        )

        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        r1 = pred_identity.predict_bucket(rows[0])
        r2 = pred_temp.predict_bucket(rows[0])

        assert abs(sum(r1.core_probs) - 1.0) < 1e-4
        assert abs(sum(r2.core_probs) - 1.0) < 1e-4
        assert r2.confidence <= r1.confidence


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestPredictionCsvOutput:
    def test_append_prediction_csv_columns(self, tmp_path: Path, predictor: OnlinePredictor) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=2)
        csv_path = tmp_path / "preds.csv"

        for row in rows:
            pred = predictor.predict_bucket(row)
            _append_prediction_csv(csv_path, pred)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            field_names = reader.fieldnames
            assert "bucket_start_ts" in field_names
            assert "core_label" in field_names
            assert "confidence" in field_names
            assert "is_rejected" in field_names
            assert "mapped_label" in field_names
            assert "core_probs" in field_names
            assert "mapped_probs" in field_names
            assert "model_version" in field_names

            data_rows = list(reader)
            assert len(data_rows) == 2

            for dr in data_rows:
                probs = json.loads(dr["core_probs"])
                assert len(probs) == 8
                assert abs(sum(probs) - 1.0) < 0.01

    def test_batch_csv_includes_core_probs(self, tmp_path: Path, trained_model_dir: Path) -> None:
        import pandas as pd

        from taskclf.infer.batch import run_batch_inference, write_predictions_csv

        model, _, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        features_df = pd.DataFrame([r.model_dump() for r in rows])

        result = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
            reject_threshold=0.55,
        )

        csv_path = write_predictions_csv(
            features_df, result.smoothed_labels, tmp_path / "preds.csv",
            confidences=result.confidences,
            is_rejected=result.is_rejected,
            core_probs=result.core_probs,
        )

        df = pd.read_csv(csv_path)
        assert "core_probs" in df.columns
        probs = json.loads(df["core_probs"].iloc[0])
        assert len(probs) == 8


# ---------------------------------------------------------------------------
# Hysteresis (merge_short_segments)
# ---------------------------------------------------------------------------


class TestMergeShortSegments:
    def _make_segments(self, specs: list[tuple[str, int]]) -> list[Segment]:
        """Build segments from (label, bucket_count) pairs."""
        ts = dt.datetime(2026, 1, 1, 10, 0)
        segs = []
        for label, count in specs:
            end = ts + dt.timedelta(seconds=60 * count)
            segs.append(Segment(start_ts=ts, end_ts=end, label=label, bucket_count=count))
            ts = end
        return segs

    def test_no_change_when_all_long(self) -> None:
        segs = self._make_segments([("Build", 5), ("Write", 5)])
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        assert len(result) == 2
        assert result[0].label == "Build"
        assert result[1].label == "Write"

    def test_short_segment_absorbed_by_same_label_neighbor(self) -> None:
        segs = self._make_segments([("Build", 5), ("Write", 1), ("Build", 5)])
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        assert len(result) <= 2

    def test_short_segment_absorbed_by_longer_neighbor(self) -> None:
        segs = self._make_segments([("Build", 10), ("Debug", 1), ("Write", 3)])
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        total = sum(s.bucket_count for s in result)
        assert total == 14

    def test_single_segment_unchanged(self) -> None:
        segs = self._make_segments([("Build", 1)])
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert merge_short_segments([]) == []

    def test_total_buckets_preserved(self) -> None:
        segs = self._make_segments([
            ("Build", 5), ("Debug", 1), ("Write", 2), ("Build", 4), ("Meet", 1),
        ])
        original_total = sum(s.bucket_count for s in segs)
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        merged_total = sum(s.bucket_count for s in result)
        assert merged_total == original_total

    def test_segments_remain_non_overlapping(self) -> None:
        segs = self._make_segments([
            ("Build", 5), ("Debug", 2), ("Write", 1), ("Build", 5),
        ])
        result = merge_short_segments(segs, min_duration_seconds=180, bucket_seconds=60)
        for i in range(len(result) - 1):
            assert result[i].end_ts <= result[i + 1].start_ts

    def test_batch_inference_applies_hysteresis(self, trained_model_dir: Path) -> None:
        import pandas as pd

        from taskclf.infer.batch import run_batch_inference

        model, _, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=20)
        features_df = pd.DataFrame([r.model_dump() for r in rows])

        result = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
        )
        total = sum(s.bucket_count for s in result.segments)
        assert total == 20


# ---------------------------------------------------------------------------
# Batch inference with calibrator
# ---------------------------------------------------------------------------


class TestBatchInferenceCalibrator:
    def test_batch_with_identity_calibrator(self, trained_model_dir: Path) -> None:
        import pandas as pd

        from taskclf.infer.batch import run_batch_inference

        model, _, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        features_df = pd.DataFrame([r.model_dump() for r in rows])

        result = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
            calibrator=IdentityCalibrator(),
        )
        assert result.core_probs.shape == (5, 8)
        np.testing.assert_allclose(result.core_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_batch_with_temperature_calibrator(self, trained_model_dir: Path) -> None:
        import pandas as pd

        from taskclf.infer.batch import run_batch_inference

        model, _, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        features_df = pd.DataFrame([r.model_dump() for r in rows])

        result_id = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
            calibrator=IdentityCalibrator(),
        )
        result_temp = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
            calibrator=TemperatureCalibrator(temperature=5.0),
        )

        np.testing.assert_allclose(result_temp.core_probs.sum(axis=1), 1.0, atol=1e-6)
        assert result_temp.confidences.max() <= result_id.confidences.max()

    def test_core_probs_on_result(self, trained_model_dir: Path) -> None:
        import pandas as pd

        from taskclf.infer.batch import run_batch_inference

        model, _, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        features_df = pd.DataFrame([r.model_dump() for r in rows])

        result = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
        )
        assert hasattr(result, "core_probs")
        assert result.core_probs.shape[0] == 5
        assert result.core_probs.shape[1] == 8
