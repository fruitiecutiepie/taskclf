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
- run_online_loop single-poll integration (TC-ONLINE-007)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.cli.main import app
from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.model_io import load_model_bundle
from taskclf.core.types import LABEL_SET_V1, FeatureRow
from taskclf.features.build import (
    build_features_from_aw_events,
    generate_dummy_features,
)
from taskclf.infer.calibration import (
    CalibratorStore,
    IdentityCalibrator,
    TemperatureCalibrator,
)
from taskclf.infer.online import OnlinePredictor, run_online_loop
from taskclf.infer.prediction import WindowPrediction
from taskclf.infer.taxonomy import TaxonomyBucket, TaxonomyConfig
from taskclf.train.lgbm import CATEGORICAL_COLUMNS

_VALID_LABELS = LABEL_SET_V1 | {MIXED_UNKNOWN}

runner = CliRunner()


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


@pytest.fixture()
def trained_model_dir(tmp_path: Path) -> Path:
    """Train a small model for testing."""
    models_dir = tmp_path / "models"
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
        pred_w1 = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders, smooth_window=1
        )
        pred_w5 = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders, smooth_window=5
        )

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

        rows_1 = build_features_from_aw_events(
            poll_1_events, session_start=session_start
        )
        rows_2 = build_features_from_aw_events(
            poll_2_events, session_start=session_start
        )

        assert rows_1[0].session_length_so_far == 0.0
        assert rows_2[0].session_length_so_far == 1.0

    def test_session_start_resets_after_gap(self) -> None:
        """After an idle gap the online loop would set a new session_start."""
        new_session_start = dt.datetime(2026, 2, 23, 11, 0, 0)
        events = [self._ev(dt.datetime(2026, 2, 23, 11, 2, 0))]

        rows = build_features_from_aw_events(events, session_start=new_session_start)
        assert rows[0].session_length_so_far == 2.0


# ---------------------------------------------------------------------------
# OnlineFeatureState — persistent rolling context (CTX-001, CTX-002, CTX-003)
# ---------------------------------------------------------------------------


class TestOnlineFeatureState:
    """Verify OnlineFeatureState provides correct rolling aggregates."""

    @staticmethod
    def _row(
        ts: dt.datetime,
        *,
        app_id: str = "org.mozilla.firefox",
        keys_per_min: float | None = 40.0,
        clicks_per_min: float | None = 5.0,
        mouse_distance: float | None = 300.0,
    ) -> FeatureRow:
        from taskclf.core.schema import FeatureSchemaV1
        from taskclf.core.hashing import stable_hash
        from taskclf.features.text import title_hash_bucket

        title_hash = stable_hash(f"title-{app_id}")
        return FeatureRow(
            user_id="test-user",
            session_id=stable_hash("test-session"),
            bucket_start_ts=ts,
            bucket_end_ts=ts + dt.timedelta(seconds=60),
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            source_ids=["test"],
            app_id=app_id,
            app_category="browser",
            window_title_hash=title_hash,
            is_browser=True,
            is_editor=False,
            is_terminal=False,
            app_switch_count_last_5m=0,
            app_foreground_time_ratio=1.0,
            app_change_count=0,
            app_dwell_time_seconds=60.0,
            keys_per_min=keys_per_min,
            clicks_per_min=clicks_per_min,
            mouse_distance=mouse_distance,
            domain_category="unknown",
            window_title_bucket=title_hash_bucket(title_hash, 256),
            title_repeat_count_session=1,
            app_switch_count_last_15m=0,
            hour_of_day=ts.hour,
            day_of_week=ts.weekday(),
            session_length_so_far=0.0,
        )

    def test_ctx001_15m_history_preserved(self) -> None:
        """CTX-001: 15-minute app switch count reflects full window."""
        from taskclf.infer.feature_state import OnlineFeatureState

        state = OnlineFeatureState(bucket_seconds=60)
        base = dt.datetime(2026, 3, 28, 10, 0, tzinfo=dt.timezone.utc)

        apps = [
            "com.app.a",
            "com.app.b",
            "com.app.c",
            "com.app.d",
            "com.app.e",
            "com.app.f",
            "com.app.g",
            "com.app.h",
            "com.app.a",
            "com.app.b",
            "com.app.c",
            "com.app.d",
            "com.app.e",
            "com.app.f",
            "com.app.g",
            "com.app.h",
        ]
        for i in range(16):
            ts = base + dt.timedelta(minutes=i)
            state.push(self._row(ts, app_id=apps[i]))

        ctx = state.get_context()
        assert ctx["app_switch_count_last_15m"] == 7

    def test_ctx002_delta_nonzero_on_second_bucket(self) -> None:
        """CTX-002: delta features are non-zero when history is available."""
        from taskclf.infer.feature_state import OnlineFeatureState

        state = OnlineFeatureState(bucket_seconds=60)
        base = dt.datetime(2026, 3, 28, 10, 0, tzinfo=dt.timezone.utc)

        state.push(
            self._row(base, keys_per_min=30.0, clicks_per_min=2.0, mouse_distance=100.0)
        )
        state.push(
            self._row(
                base + dt.timedelta(minutes=1),
                keys_per_min=60.0,
                clicks_per_min=8.0,
                mouse_distance=400.0,
            )
        )

        ctx = state.get_context()
        assert ctx["keys_per_min_delta"] is not None
        assert ctx["keys_per_min_delta"] != 0.0
        assert ctx["clicks_per_min_delta"] is not None
        assert ctx["clicks_per_min_delta"] != 0.0
        assert ctx["mouse_distance_delta"] is not None
        assert ctx["mouse_distance_delta"] != 0.0

    def test_ctx003_session_resets_after_idle_gap(self) -> None:
        """CTX-003: session_length_so_far resets after idle gap."""
        from taskclf.infer.feature_state import OnlineFeatureState

        state = OnlineFeatureState(bucket_seconds=60, idle_gap_seconds=300.0)
        base = dt.datetime(2026, 3, 28, 10, 0, tzinfo=dt.timezone.utc)

        for i in range(5):
            state.push(self._row(base + dt.timedelta(minutes=i)))

        ctx_before = state.get_context()
        assert ctx_before["session_length_so_far"] == 4.0

        after_gap = base + dt.timedelta(minutes=10)
        state.push(self._row(after_gap))

        ctx_after = state.get_context()
        assert ctx_after["session_length_so_far"] == 0.0


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
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            taxonomy=taxonomy,
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
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            taxonomy=taxonomy,
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
        self,
        trained_model_dir: Path,
    ) -> None:
        """TC-ONLINE-003: per-user calibration applied via CalibratorStore."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        user_id = rows[0].user_id

        pred_identity = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            calibrator=IdentityCalibrator(),
        )

        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={user_id: TemperatureCalibrator(temperature=5.0)},
        )
        pred_store = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            calibrator_store=store,
        )

        identity_confs = [pred_identity.predict_bucket(r).confidence for r in rows]
        store_confs = [pred_store.predict_bucket(r).confidence for r in rows]

        assert not np.allclose(identity_confs, store_confs, atol=1e-6)

    def test_calibrator_store_receives_row_user_id(
        self,
        trained_model_dir: Path,
    ) -> None:
        """UID-002: get_calibrator is called with the row's user_id, not 'default-user'."""
        from unittest.mock import MagicMock

        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        row = rows[0]

        mock_store = MagicMock()
        mock_store.get_calibrator.return_value = IdentityCalibrator()

        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
            calibrator_store=mock_store,
        )
        pred.predict_bucket(row)

        mock_store.get_calibrator.assert_called_once_with(row.user_id)
        assert row.user_id != "default-user"


# ---------------------------------------------------------------------------
# _encode_value
# ---------------------------------------------------------------------------


class TestEncodeValue:
    def test_categorical_known_value(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-004: known categorical value returns encoded int; unknown uses __unknown__ code."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)

        assert cat_encoders is not None
        cat_col = CATEGORICAL_COLUMNS[0]
        le = cat_encoders[cat_col]
        known_val = le.classes_[0]
        expected_code = float(le.transform([known_val])[0])

        assert pred._encode_value(cat_col, known_val) == expected_code
        if "__unknown__" in set(le.classes_):
            expected_unknown = float(le.transform(["__unknown__"])[0])
            assert (
                pred._encode_value(cat_col, "__never_seen_value__") == expected_unknown
            )
        else:
            assert pred._encode_value(cat_col, "__never_seen_value__") == -1.0

    def test_numerical_none_returns_zero(self, trained_model_dir: Path) -> None:
        """TC-ONLINE-005: non-categorical None returns 0.0 (matches train fillna(0))."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)

        assert pred._encode_value("key_rate", None) == 0.0
        assert pred._encode_value("key_rate", 3.5) == 3.5

    def test_unk004_unseen_categorical_uses_unknown_code(
        self, trained_model_dir: Path
    ) -> None:
        """UNK-004: unseen categorical returns __unknown__ code, not -1."""
        from sklearn.preprocessing import LabelEncoder as LE

        model, metadata, _ = load_model_bundle(trained_model_dir)

        custom_encoders: dict[str, LE] = {}
        for col in CATEGORICAL_COLUMNS:
            le = LE()
            le.fit(["val_a", "val_b", "__unknown__"])
            custom_encoders[col] = le

        pred = OnlinePredictor(model, metadata, cat_encoders=custom_encoders)
        unknown_code = float(custom_encoders["app_id"].transform(["__unknown__"])[0])

        result = pred._encode_value("app_id", "never-seen-app")
        assert result == unknown_code
        assert result != -1.0


# ---------------------------------------------------------------------------
# OnlinePredictor reject → segments
# ---------------------------------------------------------------------------


class TestOnlinePredictorRejectSegments:
    def test_rejected_predictions_produce_mixed_unknown_segments(
        self,
        trained_model_dir: Path,
    ) -> None:
        """TC-ONLINE-006: after rejected predictions, segments use MIXED_UNKNOWN."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=1.0,
        )
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=5)
        for row in rows:
            result = pred.predict_bucket(row)
            assert result.is_rejected

        segments = pred.get_segments()
        assert len(segments) >= 1
        for seg in segments:
            assert seg.label == MIXED_UNKNOWN


# ---------------------------------------------------------------------------
# run_online_loop integration
# ---------------------------------------------------------------------------


class TestRunOnlineLoop:
    def test_run_online_loop_single_poll_writes_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        trained_model_dir: Path,
    ) -> None:
        """TC-ONLINE-007: single poll produces predictions/segments and exits cleanly."""
        rows = generate_dummy_features(dt.date(2026, 3, 4), n_rows=1)
        event_ts = dt.datetime(2026, 3, 4, 12, 0, tzinfo=dt.timezone.utc)
        aw_event = AWEvent(
            timestamp=event_ts,
            duration_seconds=60.0,
            app_id="org.mozilla.firefox",
            window_title_hash="hash-123",
            is_browser=True,
            is_editor=False,
            is_terminal=False,
            app_category="browser",
        )

        fetch_calls: dict[str, dt.datetime | str] = {}
        build_calls: dict[str, dt.datetime | int | None] = {}

        monkeypatch.setattr(
            "taskclf.adapters.activitywatch.client.find_window_bucket_id",
            lambda aw_host: "aw-test-window",
        )
        monkeypatch.setattr(
            "taskclf.adapters.activitywatch.client.find_input_bucket_id",
            lambda aw_host: None,
        )

        def fake_fetch_aw_events(
            aw_host, bucket_id, window_start, window_end, title_salt
        ):
            fetch_calls["aw_host"] = aw_host
            fetch_calls["bucket_id"] = bucket_id
            fetch_calls["window_start"] = window_start
            fetch_calls["window_end"] = window_end
            return [aw_event]

        monkeypatch.setattr(
            "taskclf.adapters.activitywatch.client.fetch_aw_events",
            fake_fetch_aw_events,
        )

        def fake_build_features_from_aw_events(events, **kwargs):
            build_calls["session_start"] = kwargs.get("session_start")
            build_calls["bucket_seconds"] = kwargs.get("bucket_seconds")
            return rows

        monkeypatch.setattr(
            "taskclf.features.build.build_features_from_aw_events",
            fake_build_features_from_aw_events,
        )

        sleep_calls = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleep_calls["count"] += 1
            raise KeyboardInterrupt()

        monkeypatch.setattr("taskclf.infer.online.time.sleep", fake_sleep)

        run_online_loop(
            model_dir=trained_model_dir,
            aw_host="http://aw.local",
            poll_seconds=1,
            smooth_window=1,
            out_dir=tmp_path,
            bucket_seconds=60,
        )

        pred_path = tmp_path / "predictions.csv"
        seg_path = tmp_path / "segments.json"

        assert pred_path.exists()
        assert seg_path.exists()
        assert fetch_calls["bucket_id"] == "aw-test-window"
        assert build_calls["session_start"] == aw_event.timestamp
        assert sleep_calls["count"] == 1


# ---------------------------------------------------------------------------
# PER-003b: OnlinePredictor per-user reject thresholds in predict_bucket
# ---------------------------------------------------------------------------


class TestPerUserRejectInPredict:
    def test_per_user_threshold_overrides_global(self, trained_model_dir: Path) -> None:
        """Per-user threshold causes rejection while global would not."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)
        user_id = rows[0].user_id

        pred_global = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=0.0,
        )
        result_global = pred_global.predict_bucket(rows[0])
        assert not result_global.is_rejected

        pred_per_user = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=0.0,
            per_user_reject_thresholds={user_id: 1.0},
        )
        result_per_user = pred_per_user.predict_bucket(rows[0])
        assert result_per_user.is_rejected

    def test_per_user_threshold_missing_user_falls_back_to_global(
        self, trained_model_dir: Path
    ) -> None:
        """Users not in per-user dict use the global threshold."""
        model, metadata, cat_encoders = load_model_bundle(trained_model_dir)
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=1)

        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=0.0,
            per_user_reject_thresholds={"some-other-user": 1.0},
        )
        result = pred.predict_bucket(rows[0])
        assert not result.is_rejected
