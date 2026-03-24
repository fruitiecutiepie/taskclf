"""Tests for segment JSON I/O and batch inference in infer.batch.

Covers:
- TC-BATCH-SEG-001 through TC-BATCH-SEG-003 (segment JSON round-trip)
- TC-BATCH-001 through TC-BATCH-003 (predict_proba)
- TC-BATCH-004 through TC-BATCH-006 (run_batch_inference with taxonomy)
- TC-BATCH-007 through TC-BATCH-008 (run_batch_inference with calibrator_store)
- TC-BATCH-UTC-001 through TC-BATCH-UTC-004 (aware-UTC normalization)
"""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.model_io import load_model_bundle
from taskclf.core.types import LABEL_SET_V1
from taskclf.features.build import generate_dummy_features
from taskclf.infer.batch import (
    predict_proba,
    read_segments_json,
    run_batch_inference,
    write_segments_json,
)
from taskclf.infer.calibration import (
    CalibratorStore,
    IdentityCalibrator,
    TemperatureCalibrator,
)
from taskclf.infer.smooth import Segment
from taskclf.infer.taxonomy import TaxonomyBucket, TaxonomyConfig

runner = CliRunner()

_N_CLASSES = len(LABEL_SET_V1)  # 8


def _example_taxonomy() -> TaxonomyConfig:
    return TaxonomyConfig(
        buckets=[
            TaxonomyBucket(
                name="Deep Work",
                core_labels=["Build", "Debug", "Write"],
                color="#2E86DE",
            ),
            TaxonomyBucket(
                name="Research", core_labels=["ReadResearch", "Review"], color="#9B59B6"
            ),
            TaxonomyBucket(
                name="Communication", core_labels=["Communicate"], color="#27AE60"
            ),
            TaxonomyBucket(name="Meetings", core_labels=["Meet"], color="#E67E22"),
            TaxonomyBucket(name="Break", core_labels=["BreakIdle"], color="#7F8C8D"),
        ],
    )


@pytest.fixture(scope="module")
def _trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """Train a small model once for the whole module."""
    models_dir = tmp_path_factory.mktemp("models")
    result = runner.invoke(
        app,
        [
            "train",
            "lgbm",
            "--from",
            "2025-06-14",
            "--to",
            "2025-06-15",
            "--synthetic",
            "--models-dir",
            str(models_dir),
            "--num-boost-round",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    model_dir = next(models_dir.iterdir())
    model, metadata, cat_encoders = load_model_bundle(model_dir)
    return model, metadata, cat_encoders


@pytest.fixture(scope="module")
def _features_df() -> pd.DataFrame:
    rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=20)
    return pd.DataFrame([r.model_dump() for r in rows])


class TestSegmentJsonRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        """TC-BATCH-SEG-001: write then read preserves all fields."""
        _utc = timezone.utc
        segments = [
            Segment(
                start_ts=datetime(2025, 6, 15, 10, 0, tzinfo=_utc),
                end_ts=datetime(2025, 6, 15, 10, 5, tzinfo=_utc),
                label="Build",
                bucket_count=5,
            ),
            Segment(
                start_ts=datetime(2025, 6, 15, 10, 5, tzinfo=_utc),
                end_ts=datetime(2025, 6, 15, 10, 12, tzinfo=_utc),
                label="Write",
                bucket_count=7,
            ),
        ]
        path = tmp_path / "segments.json"
        returned = write_segments_json(segments, path)
        assert returned == path

        loaded = read_segments_json(path)
        assert len(loaded) == 2
        for original, restored in zip(segments, loaded):
            assert restored.start_ts == original.start_ts
            assert restored.end_ts == original.end_ts
            assert restored.label == original.label
            assert restored.bucket_count == original.bucket_count

    def test_empty_segments(self, tmp_path: Path) -> None:
        """TC-BATCH-SEG-002: empty list writes [] and reads back []."""
        path = tmp_path / "empty.json"
        write_segments_json([], path)
        loaded = read_segments_json(path)
        assert loaded == []

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        """TC-BATCH-SEG-003: non-existent parent dirs created automatically."""
        _utc = timezone.utc
        path = tmp_path / "deep" / "nested" / "segments.json"
        segments = [
            Segment(
                start_ts=datetime(2025, 6, 15, 10, 0, tzinfo=_utc),
                end_ts=datetime(2025, 6, 15, 10, 5, tzinfo=_utc),
                label="Build",
                bucket_count=5,
            ),
        ]
        write_segments_json(segments, path)
        assert path.exists()
        loaded = read_segments_json(path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# predict_proba direct tests
# ---------------------------------------------------------------------------


class TestPredictProba:
    def test_shape_matches_n_rows_n_classes(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-001: output shape is (N, 8)."""
        model, _, cat_encoders = _trained_artifacts
        proba = predict_proba(model, _features_df, cat_encoders)
        assert proba.shape == (len(_features_df), _N_CLASSES)

    def test_rows_sum_to_one(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-002: each row sums to ~1.0."""
        model, _, cat_encoders = _trained_artifacts
        proba = predict_proba(model, _features_df, cat_encoders)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_cat_encoders_none_fallback(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-003: cat_encoders=None falls back to integer encoding."""
        model, _, _ = _trained_artifacts
        proba = predict_proba(model, _features_df, cat_encoders=None)
        assert proba.shape == (len(_features_df), _N_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# run_batch_inference with taxonomy
# ---------------------------------------------------------------------------


class TestRunBatchInferenceTaxonomy:
    def test_taxonomy_populates_mapped_labels(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-004: mapped_labels is not None, length matches input."""
        model, _, cat_encoders = _trained_artifacts
        taxonomy = _example_taxonomy()
        result = run_batch_inference(
            model,
            _features_df,
            cat_encoders=cat_encoders,
            taxonomy=taxonomy,
        )
        assert result.mapped_labels is not None
        assert len(result.mapped_labels) == len(_features_df)

    def test_taxonomy_populates_mapped_probs(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-005: mapped_probs is not None, each dict sums to ~1.0."""
        model, _, cat_encoders = _trained_artifacts
        taxonomy = _example_taxonomy()
        result = run_batch_inference(
            model,
            _features_df,
            cat_encoders=cat_encoders,
            taxonomy=taxonomy,
        )
        assert result.mapped_probs is not None
        assert len(result.mapped_probs) == len(_features_df)
        for mp in result.mapped_probs:
            assert abs(sum(mp.values()) - 1.0) < 1e-4

    def test_without_taxonomy_mapped_fields_are_none(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-006: without taxonomy, mapped_labels and mapped_probs are None."""
        model, _, cat_encoders = _trained_artifacts
        result = run_batch_inference(
            model,
            _features_df,
            cat_encoders=cat_encoders,
        )
        assert result.mapped_labels is None
        assert result.mapped_probs is None


# ---------------------------------------------------------------------------
# run_batch_inference with calibrator_store
# ---------------------------------------------------------------------------


class TestRunBatchInferenceCalibratorStore:
    def test_per_user_calibration_applied(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-007: per-user calibration changes probabilities vs identity."""
        model, _, cat_encoders = _trained_artifacts
        df = _features_df.copy()

        identity_result = run_batch_inference(
            model,
            df,
            cat_encoders=cat_encoders,
            calibrator=IdentityCalibrator(),
        )

        store = CalibratorStore(
            global_calibrator=TemperatureCalibrator(temperature=5.0),
        )
        store_result = run_batch_inference(
            model,
            df,
            cat_encoders=cat_encoders,
            calibrator_store=store,
        )

        assert not np.allclose(
            identity_result.core_probs,
            store_result.core_probs,
            atol=1e-6,
        )

    def test_user_id_column_dispatches_per_user(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-008: user_id column used for per-user calibrator dispatch."""
        model, _, cat_encoders = _trained_artifacts
        df = _features_df.copy()

        user_ids = df["user_id"].unique()
        assert len(user_ids) >= 1
        first_user = str(user_ids[0])

        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={first_user: TemperatureCalibrator(temperature=5.0)},
        )
        result = run_batch_inference(
            model,
            df,
            cat_encoders=cat_encoders,
            calibrator_store=store,
        )

        assert result.core_probs.shape == (len(df), _N_CLASSES)
        assert np.allclose(result.core_probs.sum(axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# TC-BATCH-UTC-*: aware-UTC normalization tests (Phase 4 migration)
# ---------------------------------------------------------------------------

_UTC = timezone.utc


class TestSegmentJsonAwareUtc:
    """Verify read_segments_json normalizes timestamps to aware UTC."""

    def test_legacy_naive_json_normalized_to_aware(self, tmp_path: Path) -> None:
        """TC-BATCH-UTC-001: legacy naive ISO strings are read as aware UTC."""
        import json as _json

        records = [
            {
                "start_ts": "2025-06-15T10:00:00",
                "end_ts": "2025-06-15T10:05:00",
                "label": "Build",
                "bucket_count": 5,
            },
        ]
        path = tmp_path / "legacy.json"
        path.write_text(_json.dumps(records))

        loaded = read_segments_json(path)
        assert len(loaded) == 1
        seg = loaded[0]
        assert seg.start_ts.tzinfo is not None
        assert seg.start_ts == datetime(2025, 6, 15, 10, 0, tzinfo=_UTC)
        assert seg.end_ts.tzinfo is not None
        assert seg.end_ts == datetime(2025, 6, 15, 10, 5, tzinfo=_UTC)

    def test_aware_utc_json_preserved(self, tmp_path: Path) -> None:
        """TC-BATCH-UTC-002: aware UTC ISO strings round-trip exactly."""
        _utc = timezone.utc
        segments = [
            Segment(
                start_ts=datetime(2025, 6, 15, 10, 0, tzinfo=_utc),
                end_ts=datetime(2025, 6, 15, 10, 5, tzinfo=_utc),
                label="Build",
                bucket_count=5,
            ),
        ]
        path = tmp_path / "aware.json"
        write_segments_json(segments, path)
        loaded = read_segments_json(path)

        assert loaded[0].start_ts == segments[0].start_ts
        assert loaded[0].start_ts.tzinfo is not None

    def test_non_utc_offset_json_converted_to_utc(self, tmp_path: Path) -> None:
        """TC-BATCH-UTC-003: non-UTC offset ISO strings are converted to UTC."""
        import json as _json

        records = [
            {
                "start_ts": "2025-06-15T15:00:00+05:00",
                "end_ts": "2025-06-15T15:05:00+05:00",
                "label": "Build",
                "bucket_count": 5,
            },
        ]
        path = tmp_path / "offset.json"
        path.write_text(_json.dumps(records))

        loaded = read_segments_json(path)
        seg = loaded[0]
        assert seg.start_ts == datetime(2025, 6, 15, 10, 0, tzinfo=_UTC)
        assert seg.end_ts == datetime(2025, 6, 15, 10, 5, tzinfo=_UTC)


class TestBatchInferenceAwareUtc:
    """Verify run_batch_inference produces aware-UTC segment timestamps."""

    def test_segments_have_aware_utc_timestamps(
        self,
        _trained_artifacts,
        _features_df: pd.DataFrame,
    ) -> None:
        """TC-BATCH-UTC-004: segments from batch inference have aware-UTC timestamps."""
        model, _, cat_encoders = _trained_artifacts
        result = run_batch_inference(
            model,
            _features_df,
            cat_encoders=cat_encoders,
        )
        for seg in result.segments:
            assert seg.start_ts.tzinfo is not None, "start_ts must be aware"
            assert seg.end_ts.tzinfo is not None, "end_ts must be aware"
