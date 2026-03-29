"""Tests for taskclf.core.telemetry — snapshot computation and persistence.

Also covers Phase 5 Step 5.3: CNF-002, CNF-003, P5-003
(suggestions-per-active-day tracking).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.telemetry import (
    SuggestionTracker,
    TelemetrySnapshot,
    TelemetryStore,
    compute_telemetry,
)


# ---------------------------------------------------------------------------
# compute_telemetry
# ---------------------------------------------------------------------------


class TestComputeTelemetry:
    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 50
        feat_df = pd.DataFrame(
            {
                "bucket_start_ts": pd.date_range(
                    "2026-02-20T08:00", periods=n, freq="1min"
                ),
                "keys_per_min": rng.normal(30, 5, n),
                "clicks_per_min": rng.normal(10, 2, n),
                "mouse_distance": np.concatenate(
                    [rng.normal(500, 100, 45), np.full(5, np.nan)]
                ),
                "scroll_events_per_min": rng.uniform(0, 10, n),
            }
        )
        labels = ["Build"] * 30 + ["Debug"] * 10 + [MIXED_UNKNOWN] * 10
        probs = rng.dirichlet(np.ones(8) * 3, size=n)
        confidences = probs.max(axis=1)
        return feat_df, labels, confidences, probs

    def test_basic_snapshot(
        self,
        sample_data: tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray],
    ) -> None:
        feat_df, labels, confidences, probs = sample_data
        snap = compute_telemetry(
            feat_df,
            labels=labels,
            confidences=confidences,
            core_probs=probs,
        )
        assert isinstance(snap, TelemetrySnapshot)
        assert snap.total_windows == 50
        assert snap.reject_rate == pytest.approx(0.2, abs=0.01)
        assert snap.confidence_stats is not None
        assert 0 < snap.confidence_stats.mean < 1
        assert snap.mean_entropy > 0

    def test_feature_missingness(
        self,
        sample_data: tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray],
    ) -> None:
        feat_df, labels, confidences, probs = sample_data
        snap = compute_telemetry(feat_df, labels=labels)
        assert snap.feature_missingness.get("mouse_distance", 0) > 0
        assert snap.feature_missingness.get("keys_per_min", 0) == 0

    def test_class_distribution(
        self,
        sample_data: tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray],
    ) -> None:
        feat_df, labels, confidences, probs = sample_data
        snap = compute_telemetry(feat_df, labels=labels)
        assert "Build" in snap.class_distribution
        assert snap.class_distribution["Build"] == pytest.approx(0.6, abs=0.01)

    def test_window_range(
        self,
        sample_data: tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray],
    ) -> None:
        feat_df, labels, confidences, probs = sample_data
        snap = compute_telemetry(feat_df, labels=labels)
        assert snap.window_start is not None
        assert snap.window_end is not None
        assert snap.window_start < snap.window_end

    def test_user_id_propagated(self) -> None:
        df = pd.DataFrame(
            {"bucket_start_ts": pd.date_range("2026-01-01", periods=5, freq="1min")}
        )
        snap = compute_telemetry(df, user_id="user-42")
        assert snap.user_id == "user-42"

    def test_empty_df(self) -> None:
        df = pd.DataFrame()
        snap = compute_telemetry(df)
        assert snap.total_windows == 0
        assert snap.reject_rate == 0.0

    def test_no_labels(self) -> None:
        df = pd.DataFrame(
            {"bucket_start_ts": pd.date_range("2026-01-01", periods=5, freq="1min")}
        )
        snap = compute_telemetry(df)
        assert snap.class_distribution == {}
        assert snap.reject_rate == 0.0


# ---------------------------------------------------------------------------
# TelemetryStore
# ---------------------------------------------------------------------------


class TestTelemetryStore:
    def test_append_and_read(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telemetry")
        snap = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            total_windows=100,
            reject_rate=0.15,
        )
        path = store.append(snap)
        assert path.exists()

        recent = store.read_recent(5)
        assert len(recent) == 1
        assert recent[0].total_windows == 100

    def test_multiple_appends(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telemetry")
        for i in range(5):
            snap = TelemetrySnapshot(
                timestamp=datetime(2026, 2, 20, 10, i, tzinfo=timezone.utc),
                total_windows=i * 10,
            )
            store.append(snap)

        recent = store.read_recent(3)
        assert len(recent) == 3
        assert recent[-1].total_windows == 40

    def test_per_user_files(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telemetry")

        snap_global = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            total_windows=50,
        )
        snap_user = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            user_id="user-1",
            total_windows=30,
        )
        store.append(snap_global)
        store.append(snap_user)

        assert len(store.read_recent(10, user_id=None)) == 1
        assert len(store.read_recent(10, user_id="user-1")) == 1
        assert store.read_recent(10, user_id="user-1")[0].total_windows == 30

    def test_read_range(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "telemetry")
        base = datetime(2026, 2, 20, tzinfo=timezone.utc)
        for i in range(10):
            snap = TelemetrySnapshot(
                timestamp=base + timedelta(hours=i),
                total_windows=i,
            )
            store.append(snap)

        start = base + timedelta(hours=3)
        end = base + timedelta(hours=6)
        result = store.read_range(start, end)
        assert len(result) == 4
        assert all(start <= s.timestamp <= end for s in result)

    def test_empty_store(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "empty")
        assert store.read_recent(5) == []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        assert store.read_range(start, end) == []

    def test_append_no_temp_files_left(self, tmp_path: Path) -> None:
        """Regression: atomic write must not leave .tmp files behind."""
        store = TelemetryStore(tmp_path / "telemetry")
        snap = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            total_windows=42,
        )
        store.append(snap)
        tmp_files = list((tmp_path / "telemetry").glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# CNF-002, CNF-003, P5-003: Suggestion tracking (Phase 5, Step 5.3)
# ---------------------------------------------------------------------------


class TestSuggestionTracking:
    """Tests for SuggestionTracker and suggestions_per_day metric."""

    def test_cnf002_records_accepted_suggestions(self) -> None:
        """CNF-002: After N predictions with K accepted, telemetry shows
        K suggestions."""
        tracker = SuggestionTracker()
        date_str = "2026-03-28"
        base = datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc)

        n_accepted = 7
        for i in range(n_accepted):
            tracker.record(base + timedelta(minutes=i))

        assert tracker.count_for_date(date_str) == n_accepted

    def test_cnf003_zero_suggestions_triggers_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CNF-003: Loaded model + full active day + zero suggestions
        triggers a warning."""
        tracker = SuggestionTracker()
        date_str = "2026-03-28"

        with caplog.at_level(logging.WARNING, logger="taskclf.core.telemetry"):
            tracker.check_zero_suggestions(date_str, model_loaded=True)

        assert any("Zero suggestions" in r.message for r in caplog.records)
        assert any("2026-03-28" in r.message for r in caplog.records)

    def test_cnf003_no_warning_when_model_not_loaded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when model is not loaded."""
        tracker = SuggestionTracker()
        with caplog.at_level(logging.WARNING, logger="taskclf.core.telemetry"):
            tracker.check_zero_suggestions("2026-03-28", model_loaded=False)

        assert not any("Zero suggestions" in r.message for r in caplog.records)

    def test_cnf003_no_warning_when_suggestions_exist(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when suggestions were recorded."""
        tracker = SuggestionTracker()
        tracker.record(datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc))

        with caplog.at_level(logging.WARNING, logger="taskclf.core.telemetry"):
            tracker.check_zero_suggestions("2026-03-28", model_loaded=True)

        assert not any("Zero suggestions" in r.message for r in caplog.records)

    def test_p5003_simulated_active_day_produces_suggestions(self) -> None:
        """P5-003: Simulated active day produces > 0 suggestions."""
        tracker = SuggestionTracker()
        date_str = "2026-03-28"
        base = datetime(2026, 3, 28, 8, 0, tzinfo=timezone.utc)

        for i in range(480):
            tracker.record(base + timedelta(minutes=i))

        assert tracker.count_for_date(date_str) > 0

    def test_suggestions_per_day_in_snapshot(self) -> None:
        """TelemetrySnapshot includes suggestions_per_day field."""
        snap = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            total_windows=10,
            suggestions_per_day=5,
        )
        assert snap.suggestions_per_day == 5

    def test_suggestions_per_day_default_zero(self) -> None:
        """Default value for suggestions_per_day is 0."""
        snap = TelemetrySnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            total_windows=10,
        )
        assert snap.suggestions_per_day == 0
