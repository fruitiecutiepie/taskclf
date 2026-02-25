"""Shared fixtures for the taskclf test suite."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from taskclf.core.schema import FeatureSchemaV1


@pytest.fixture()
def sample_date() -> dt.date:
    return dt.date(2025, 6, 15)


@pytest.fixture()
def valid_feature_row_data() -> dict[str, Any]:
    """Minimal valid data dict for constructing a FeatureRow."""
    return {
        "user_id": "test-user-001",
        "device_id": None,
        "session_id": "test-session-001",
        "bucket_start_ts": dt.datetime(2025, 6, 15, 10, 0),
        "bucket_end_ts": dt.datetime(2025, 6, 15, 10, 1),
        "schema_version": FeatureSchemaV1.VERSION,
        "schema_hash": FeatureSchemaV1.SCHEMA_HASH,
        "source_ids": ["test-collector"],
        "app_id": "com.apple.Terminal",
        "app_category": "terminal",
        "window_title_hash": "abc123def456",
        "is_browser": False,
        "is_editor": False,
        "is_terminal": True,
        "app_switch_count_last_5m": 2,
        "app_foreground_time_ratio": 0.85,
        "app_change_count": 1,
        "keys_per_min": 60.0,
        "backspace_ratio": 0.05,
        "shortcut_rate": 0.1,
        "clicks_per_min": 5.0,
        "scroll_events_per_min": 2.0,
        "mouse_distance": 300.0,
        "active_seconds_keyboard": 40.0,
        "active_seconds_mouse": 30.0,
        "active_seconds_any": 45.0,
        "max_idle_run_seconds": 10.0,
        "event_density": 2.0,
        "domain_category": "non_browser",
        "window_title_bucket": 42,
        "title_repeat_count_session": 1,
        "keys_per_min_rolling_5": 55.0,
        "keys_per_min_rolling_15": 50.0,
        "mouse_distance_rolling_5": 280.0,
        "mouse_distance_rolling_15": 260.0,
        "keys_per_min_delta": 5.0,
        "clicks_per_min_delta": 1.0,
        "mouse_distance_delta": 20.0,
        "app_switch_count_last_15m": 4,
        "hour_of_day": 10,
        "day_of_week": 6,
        "session_length_so_far": 30.0,
    }
