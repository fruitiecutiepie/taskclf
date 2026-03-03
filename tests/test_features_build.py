"""Tests for features.build: generate_dummy_features, build_features_for_date,
and _aggregate_input_for_bucket.

Covers: TC-FEAT-BUILD-001 through TC-FEAT-BUILD-013,
        TC-FEAT-AGG-001 through TC-FEAT-AGG-006.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from taskclf.adapters.activitywatch.types import AWInputEvent
from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS, DEFAULT_DUMMY_ROWS
from taskclf.core.schema import FeatureSchemaV1
from taskclf.features.build import (
    _aggregate_input_for_bucket,
    build_features_for_date,
    generate_dummy_features,
)

_DATE = dt.date(2025, 6, 15)


class TestGenerateDummyFeatures:
    """TC-FEAT-BUILD-001 through TC-FEAT-BUILD-009."""

    def test_default_row_count(self) -> None:
        """TC-FEAT-BUILD-001: default call returns DEFAULT_DUMMY_ROWS rows."""
        rows = generate_dummy_features(_DATE)
        assert len(rows) == DEFAULT_DUMMY_ROWS

    def test_custom_n_rows(self) -> None:
        """TC-FEAT-BUILD-002: custom n_rows=5 returns exactly 5."""
        rows = generate_dummy_features(_DATE, n_rows=5)
        assert len(rows) == 5

    def test_schema_validation(self) -> None:
        """TC-FEAT-BUILD-003: all rows pass FeatureSchemaV1 validation."""
        rows = generate_dummy_features(_DATE)
        df = pd.DataFrame([r.model_dump() for r in rows])
        FeatureSchemaV1.validate_dataframe(df)

    def test_timestamps_on_correct_date(self) -> None:
        """TC-FEAT-BUILD-004: bucket_start_ts spans hours 9-17 of the given date."""
        rows = generate_dummy_features(_DATE)
        for row in rows:
            assert row.bucket_start_ts.date() == _DATE
            assert 9 <= row.bucket_start_ts.hour <= 17

    def test_custom_user_and_device(self) -> None:
        """TC-FEAT-BUILD-005: custom user_id and device_id propagated."""
        rows = generate_dummy_features(
            _DATE, n_rows=3, user_id="custom-user", device_id="dev-99"
        )
        for row in rows:
            assert row.user_id == "custom-user"
            assert row.device_id == "dev-99"

    def test_schema_version_and_hash(self) -> None:
        """TC-FEAT-BUILD-006: schema_version and schema_hash match FeatureSchemaV1."""
        rows = generate_dummy_features(_DATE)
        for row in rows:
            assert row.schema_version == FeatureSchemaV1.VERSION
            assert row.schema_hash == FeatureSchemaV1.SCHEMA_HASH

    def test_session_id_consistent(self) -> None:
        """TC-FEAT-BUILD-007: session_id is a non-empty hash, same within a call."""
        rows = generate_dummy_features(_DATE, n_rows=5)
        sids = {row.session_id for row in rows}
        assert len(sids) == 1
        sid = sids.pop()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_dynamics_fields_populated(self) -> None:
        """TC-FEAT-BUILD-008: rolling means are populated after first few rows."""
        rows = generate_dummy_features(_DATE, n_rows=10)
        later_rows = rows[5:]
        has_rolling = any(
            r.keys_per_min_rolling_5 is not None for r in later_rows
        )
        assert has_rolling

    def test_zero_rows(self) -> None:
        """TC-FEAT-BUILD-009: n_rows=0 returns empty list."""
        rows = generate_dummy_features(_DATE, n_rows=0)
        assert rows == []


class TestBuildFeaturesForDate:
    """TC-FEAT-BUILD-010 through TC-FEAT-BUILD-013."""

    def test_returns_existing_parquet_path(self, tmp_path: Path) -> None:
        """TC-FEAT-BUILD-010: returns a Path that exists and is .parquet."""
        result = build_features_for_date(_DATE, tmp_path)
        assert result.exists()
        assert result.suffix == ".parquet"

    def test_output_path_structure(self, tmp_path: Path) -> None:
        """TC-FEAT-BUILD-011: output path matches expected directory layout."""
        result = build_features_for_date(_DATE, tmp_path)
        expected = tmp_path / f"features_v1/date={_DATE.isoformat()}" / "features.parquet"
        assert result == expected

    def test_parquet_readable_with_correct_columns(self, tmp_path: Path) -> None:
        """TC-FEAT-BUILD-012: parquet readable with all FeatureRow columns."""
        result = build_features_for_date(_DATE, tmp_path)
        df = pd.read_parquet(result)
        for col in FeatureSchemaV1.COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_default(self, tmp_path: Path) -> None:
        """TC-FEAT-BUILD-013: row count matches DEFAULT_DUMMY_ROWS."""
        result = build_features_for_date(_DATE, tmp_path)
        df = pd.read_parquet(result)
        assert len(df) == DEFAULT_DUMMY_ROWS


# ---------------------------------------------------------------------------
# Helpers for _aggregate_input_for_bucket tests
# ---------------------------------------------------------------------------

_BUCKET_TS = dt.datetime(2025, 6, 15, 10, 0, 0)


def _input_ev(
    ts: dt.datetime,
    duration: float = 5.0,
    presses: int = 0,
    clicks: int = 0,
    delta_x: int = 0,
    delta_y: int = 0,
    scroll_x: int = 0,
    scroll_y: int = 0,
) -> AWInputEvent:
    return AWInputEvent(
        timestamp=ts,
        duration_seconds=duration,
        presses=presses,
        clicks=clicks,
        delta_x=delta_x,
        delta_y=delta_y,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
    )


class TestAggregateInputForBucket:
    """TC-FEAT-AGG-001 through TC-FEAT-AGG-006."""

    def test_empty_bucket_input(self) -> None:
        """TC-FEAT-AGG-001: empty input returns all None values."""
        result = _aggregate_input_for_bucket(_BUCKET_TS, [], DEFAULT_BUCKET_SECONDS)
        for key, val in result.items():
            assert val is None, f"Expected None for '{key}', got {val}"

    def test_single_active_event(self) -> None:
        """TC-FEAT-AGG-002: single active event produces positive occupancy."""
        ev = _input_ev(_BUCKET_TS, duration=5.0, presses=10, clicks=2)
        result = _aggregate_input_for_bucket(_BUCKET_TS, [ev], DEFAULT_BUCKET_SECONDS)

        assert result["active_seconds_any"] > 0
        assert result["active_seconds_keyboard"] > 0
        assert result["active_seconds_mouse"] > 0
        assert result["max_idle_run_seconds"] == 0.0

    def test_mixed_active_idle_events(self) -> None:
        """TC-FEAT-AGG-003: max_idle_run_seconds equals longest idle run."""
        events = [
            _input_ev(_BUCKET_TS, duration=5.0, presses=5),
            _input_ev(
                _BUCKET_TS + dt.timedelta(seconds=5),
                duration=10.0,
                presses=0, clicks=0,
            ),
            _input_ev(
                _BUCKET_TS + dt.timedelta(seconds=15),
                duration=3.0,
                presses=0, clicks=0,
            ),
            _input_ev(
                _BUCKET_TS + dt.timedelta(seconds=18),
                duration=5.0, presses=8,
            ),
        ]
        result = _aggregate_input_for_bucket(_BUCKET_TS, events, DEFAULT_BUCKET_SECONDS)

        assert result["max_idle_run_seconds"] == 13.0

    def test_keyboard_only(self) -> None:
        """TC-FEAT-AGG-004: only keyboard activity means mouse seconds is 0."""
        ev = _input_ev(_BUCKET_TS, duration=5.0, presses=20)
        result = _aggregate_input_for_bucket(_BUCKET_TS, [ev], DEFAULT_BUCKET_SECONDS)

        assert result["active_seconds_keyboard"] > 0
        assert result["active_seconds_mouse"] == 0.0

    def test_mouse_only(self) -> None:
        """TC-FEAT-AGG-005: only mouse activity means keyboard seconds is 0."""
        ev = _input_ev(_BUCKET_TS, duration=5.0, clicks=3, delta_x=100, delta_y=50)
        result = _aggregate_input_for_bucket(_BUCKET_TS, [ev], DEFAULT_BUCKET_SECONDS)

        assert result["active_seconds_mouse"] > 0
        assert result["active_seconds_keyboard"] == 0.0

    def test_event_density_formula(self) -> None:
        """TC-FEAT-AGG-006: event_density = active_event_count / active_seconds_any."""
        events = [
            _input_ev(_BUCKET_TS, duration=10.0, presses=5),
            _input_ev(
                _BUCKET_TS + dt.timedelta(seconds=10),
                duration=10.0, clicks=2,
            ),
        ]
        result = _aggregate_input_for_bucket(_BUCKET_TS, events, DEFAULT_BUCKET_SECONDS)

        expected_density = round(2 / result["active_seconds_any"], 4)
        assert result["event_density"] == expected_density
