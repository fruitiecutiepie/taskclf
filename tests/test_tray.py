"""Tests for the system tray labeling app.

Covers:
- ActivityMonitor transition detection logic
- Label span creation via tray callbacks
- TrayLabeler event publishing (_handle_transition, _handle_poll)
- ActivityMonitor._publish_status event shape
- _send_desktop_notification platform dispatch
- _make_icon_image output
- _LabelSuggester.suggest label prediction
- No tests for pystray GUI (interactive widget, untestable in CI)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.labels.store import append_label_span, read_label_spans
from taskclf.ui.events import EventBus
from taskclf.ui.tray import (
    ActivityMonitor,
    TrayLabeler,
    _LabelSuggester,
    _make_icon_image,
    _send_desktop_notification,
)


class TestActivityTransitionDetection:
    """Unit tests for the transition detection state machine."""

    def _make_monitor(
        self,
        transition_minutes: int = 3,
        poll_seconds: int = 60,
        transitions: list | None = None,
    ) -> ActivityMonitor:
        captured = transitions if transitions is not None else []

        def on_transition(prev: str, new: str, start: dt.datetime, end: dt.datetime) -> None:
            captured.append((prev, new, start, end))

        return ActivityMonitor(
            poll_seconds=poll_seconds,
            transition_minutes=transition_minutes,
            on_transition=on_transition,
        )

    def test_no_transition_on_same_app(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(transitions=transitions)

        for _ in range(10):
            monitor.check_transition("com.apple.Terminal")

        assert len(transitions) == 0

    def test_transition_fires_after_threshold(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal")
        assert monitor.current_app == "com.apple.Terminal"

        # 3 consecutive polls with a new app (3 * 60s = 180s = 3 min)
        monitor.check_transition("us.zoom.xos")
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos")
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos")
        assert len(transitions) == 1

        prev, new, _start, _end = transitions[0]
        assert prev == "com.apple.Terminal"
        assert new == "us.zoom.xos"
        assert monitor.current_app == "us.zoom.xos"

    def test_no_transition_if_app_flips_back(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal")

        # Switch to Zoom for 2 polls (under threshold), then back
        monitor.check_transition("us.zoom.xos")
        monitor.check_transition("us.zoom.xos")
        monitor.check_transition("com.apple.Terminal")

        assert len(transitions) == 0
        assert monitor.current_app == "com.apple.Terminal"

    def test_transition_resets_after_firing(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=2, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal")
        monitor.check_transition("us.zoom.xos")
        monitor.check_transition("us.zoom.xos")
        assert len(transitions) == 1

        # Now stable on Zoom, transition to Slack
        monitor.check_transition("us.zoom.xos")
        monitor.check_transition("com.tinyspeck.slackmacgap")
        monitor.check_transition("com.tinyspeck.slackmacgap")
        assert len(transitions) == 2
        assert transitions[1][0] == "us.zoom.xos"
        assert transitions[1][1] == "com.tinyspeck.slackmacgap"

    def test_candidate_resets_on_third_app(self) -> None:
        """If the app changes to a third app during candidate phase, candidate resets."""
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal")
        monitor.check_transition("us.zoom.xos")  # candidate = Zoom
        monitor.check_transition("com.tinyspeck.slackmacgap")  # candidate resets to Slack
        monitor.check_transition("com.tinyspeck.slackmacgap")
        monitor.check_transition("com.tinyspeck.slackmacgap")
        assert len(transitions) == 1
        assert transitions[0][1] == "com.tinyspeck.slackmacgap"

    def test_first_app_sets_current(self) -> None:
        monitor = self._make_monitor()
        assert monitor.current_app is None
        monitor.check_transition("com.apple.Terminal")
        assert monitor.current_app == "com.apple.Terminal"

    def test_short_threshold(self) -> None:
        """With transition_minutes=1 and poll_seconds=60, two polls trigger (first sets candidate, second confirms)."""
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=1, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal")
        monitor.check_transition("us.zoom.xos")  # candidate set, duration=60
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos")  # duration=120 >= 60, fires
        assert len(transitions) == 1


class TestLabelFromTray:
    """Label span creation as the tray callbacks would invoke it."""

    def test_creates_span_for_minutes_back(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 15, 0, 0)
        minutes = 10
        start_ts = now - dt.timedelta(minutes=minutes)

        span = LabelSpan(
            start_ts=start_ts,
            end_ts=now,
            label="Build",
            provenance="manual",
            user_id="default-user",
        )
        append_label_span(span, labels_path)

        loaded = read_label_spans(labels_path)
        assert len(loaded) == 1
        assert loaded[0].label == "Build"
        assert loaded[0].end_ts - loaded[0].start_ts == dt.timedelta(minutes=10)

    def test_overlap_rejected(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 15, 0, 0)

        span1 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=10),
            end_ts=now,
            label="Build",
            provenance="manual",
            user_id="default-user",
        )
        append_label_span(span1, labels_path)

        span2 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=5),
            end_ts=now + dt.timedelta(minutes=5),
            label="Meet",
            provenance="manual",
            user_id="default-user",
        )
        with pytest.raises(ValueError, match="overlaps"):
            append_label_span(span2, labels_path)

    def test_non_overlapping_spans_accepted(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 15, 0, 0)

        for i, label in enumerate(["Build", "Meet", "Write"]):
            span = LabelSpan(
                start_ts=now + dt.timedelta(minutes=i * 10),
                end_ts=now + dt.timedelta(minutes=(i + 1) * 10),
                label=label,
                provenance="manual",
                user_id="default-user",
            )
            append_label_span(span, labels_path)

        loaded = read_label_spans(labels_path)
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# Helpers for TrayLabeler / ActivityMonitor event tests
# ---------------------------------------------------------------------------

def _capture_bus() -> tuple[EventBus, list[dict]]:
    """Return an EventBus whose publish_threadsafe appends to a list."""
    bus = EventBus()
    captured: list[dict] = []
    bus.publish_threadsafe = lambda event: captured.append(event)  # type: ignore[assignment]
    return bus, captured


def _make_tray_labeler(
    tmp_path: Path,
    event_bus: EventBus | None = None,
) -> TrayLabeler:
    """Build a TrayLabeler with no model and a tmp data dir."""
    return TrayLabeler(
        data_dir=tmp_path,
        model_dir=None,
        event_bus=event_bus,
    )


_BLOCK_START = dt.datetime(2026, 3, 1, 10, 0, 0)
_BLOCK_END = dt.datetime(2026, 3, 1, 10, 15, 0)


# ---------------------------------------------------------------------------
# 46a — TrayLabeler._handle_transition event publishing
# ---------------------------------------------------------------------------


class TestHandleTransition:

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_transition_without_model_publishes_prompt_and_prediction(
        self, _mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """TC-UI-TRAY-001"""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._handle_transition(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )

        types = [e["type"] for e in captured]
        assert "prompt_label" in types
        assert "prediction" in types

        prompt = next(e for e in captured if e["type"] == "prompt_label")
        assert prompt["suggested_label"] is None
        assert prompt["suggested_confidence"] is None
        assert prompt["prev_app"] == "com.apple.Terminal"
        assert prompt["new_app"] == "us.zoom.xos"

        pred = next(e for e in captured if e["type"] == "prediction")
        assert pred["label"] == "unknown"
        assert pred["confidence"] == 0.0
        assert pred["current_app"] == "us.zoom.xos"

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_transition_with_suggestion_publishes_suggest_label(
        self, _mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """TC-UI-TRAY-002"""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        mock_suggester = MagicMock()
        mock_suggester.suggest.return_value = ("Build", 0.85)
        labeler._suggester = mock_suggester

        labeler._handle_transition(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )

        types = [e["type"] for e in captured]
        assert "prompt_label" in types
        assert "suggest_label" in types
        assert "prediction" not in types

        prompt = next(e for e in captured if e["type"] == "prompt_label")
        assert prompt["suggested_label"] == "Build"
        assert prompt["suggested_confidence"] == 0.85

        suggest = next(e for e in captured if e["type"] == "suggest_label")
        assert suggest["reason"] == "app_switch"
        assert suggest["suggested"] == "Build"
        assert suggest["confidence"] == 0.85

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_transition_count_incremented(
        self, _mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """TC-UI-TRAY-003"""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._transition_count == 0

        labeler._handle_transition(
            "A", "B", _BLOCK_START, _BLOCK_END,
        )
        assert labeler._transition_count == 1

        labeler._handle_transition(
            "B", "C", _BLOCK_START, _BLOCK_END,
        )
        assert labeler._transition_count == 2

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_last_transition_dict_shape(
        self, _mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """TC-UI-TRAY-004"""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._last_transition is None

        labeler._handle_transition(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )

        lt = labeler._last_transition
        assert lt is not None
        assert set(lt.keys()) == {
            "prev_app", "new_app", "block_start", "block_end", "fired_at",
        }
        assert lt["prev_app"] == "com.apple.Terminal"
        assert lt["new_app"] == "us.zoom.xos"
        assert lt["block_start"] == _BLOCK_START.isoformat()
        assert lt["block_end"] == _BLOCK_END.isoformat()
        dt.datetime.fromisoformat(lt["fired_at"])


# ---------------------------------------------------------------------------
# 46b — TrayLabeler._handle_poll tray_state event
# ---------------------------------------------------------------------------


class TestHandlePoll:

    def test_poll_publishes_tray_state_with_expected_keys(
        self, tmp_path: Path,
    ) -> None:
        """TC-UI-TRAY-005"""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._handle_poll("com.apple.Safari")

        assert len(captured) == 1
        event = captured[0]
        assert event["type"] == "tray_state"
        expected_keys = {
            "type", "model_loaded", "model_dir", "model_schema_hash",
            "suggested_label", "suggested_confidence",
            "transition_count", "last_transition",
            "labels_saved_count", "data_dir", "ui_port", "dev_mode",
        }
        assert set(event.keys()) == expected_keys
        assert event["model_loaded"] is False
        assert event["transition_count"] == 0
        assert event["data_dir"] == str(tmp_path)

    def test_current_app_updated_after_poll(self, tmp_path: Path) -> None:
        """TC-UI-TRAY-006"""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._current_app == "unknown"

        labeler._handle_poll("com.apple.Safari")
        assert labeler._current_app == "com.apple.Safari"


# ---------------------------------------------------------------------------
# 46c — ActivityMonitor._publish_status
# ---------------------------------------------------------------------------


class TestPublishStatus:

    def test_status_event_shape(self) -> None:
        """TC-UI-TRAY-007"""
        bus, captured = _capture_bus()
        monitor = ActivityMonitor(
            poll_seconds=30, transition_minutes=3, event_bus=bus,
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0)

        monitor._publish_status("com.apple.Terminal")

        assert len(captured) == 1
        event = captured[0]
        assert event["type"] == "status"
        assert event["state"] == "collecting"
        assert event["current_app"] == "com.apple.Terminal"
        assert event["poll_seconds"] == 30
        assert event["poll_count"] == 1
        assert isinstance(event["uptime_s"], int)
        assert event["aw_connected"] is False
        assert event["aw_host"] is not None
        assert "last_event_count" in event
        assert "last_app_counts" in event

    def test_poll_count_increments(self) -> None:
        """TC-UI-TRAY-008"""
        bus, captured = _capture_bus()
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3, event_bus=bus,
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0)

        monitor._publish_status("app1")
        monitor._publish_status("app1")

        assert len(captured) == 2
        assert captured[0]["poll_count"] == 1
        assert captured[1]["poll_count"] == 2

    def test_candidate_app_included(self) -> None:
        """TC-UI-TRAY-009"""
        bus, captured = _capture_bus()
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3, event_bus=bus,
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0)
        monitor._candidate_app = "us.zoom.xos"
        monitor._candidate_duration = 120

        monitor._publish_status("com.apple.Terminal")

        event = captured[0]
        assert event["candidate_app"] == "us.zoom.xos"
        assert event["candidate_duration_s"] == 120


# ---------------------------------------------------------------------------
# 47 — _send_desktop_notification
# ---------------------------------------------------------------------------


class TestSendDesktopNotification:

    @patch("taskclf.ui.tray.subprocess.run")
    @patch("taskclf.ui.tray.platform.system", return_value="Darwin")
    def test_macos_calls_osascript(
        self, _mock_sys: MagicMock, mock_run: MagicMock,
    ) -> None:
        """TC-UI-NOTIF-001"""
        _send_desktop_notification("Test Title", "Hello world")

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "osascript"
        assert cmd[1] == "-e"
        assert "Test Title" in cmd[2]
        assert "Hello world" in cmd[2]

    @patch("taskclf.ui.tray.logger")
    @patch("taskclf.ui.tray.platform.system", return_value="Linux")
    def test_non_macos_falls_back_to_logger(
        self, _mock_sys: MagicMock, mock_logger: MagicMock,
    ) -> None:
        """TC-UI-NOTIF-002"""
        _send_desktop_notification("Test Title", "Hello world")

        mock_logger.info.assert_called_once()
        log_args = mock_logger.info.call_args[0]
        assert "Test Title" in str(log_args)
        assert "Hello world" in str(log_args)

    @patch("taskclf.ui.tray.logger")
    @patch(
        "taskclf.ui.tray.subprocess.run",
        side_effect=OSError("osascript not found"),
    )
    @patch("taskclf.ui.tray.platform.system", return_value="Darwin")
    def test_subprocess_failure_falls_back_to_logger(
        self, _mock_sys: MagicMock, _mock_run: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """TC-UI-NOTIF-003"""
        _send_desktop_notification("Title", "Body")

        mock_logger.info.assert_called_once()


# ---------------------------------------------------------------------------
# 48 — _make_icon_image
# ---------------------------------------------------------------------------


class TestMakeIconImage:

    def test_default_returns_rgba_64x64(self) -> None:
        """TC-UI-ICON-001"""
        img = _make_icon_image()
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)
        assert img.mode == "RGBA"

    def test_custom_color_and_size(self) -> None:
        """TC-UI-ICON-002"""
        img = _make_icon_image("#FF0000", size=32)
        assert img.size == (32, 32)
        assert img.mode == "RGBA"


# ---------------------------------------------------------------------------
# 49 — _LabelSuggester.suggest
# ---------------------------------------------------------------------------


def _make_suggester(predictor_mock: MagicMock) -> _LabelSuggester:
    """Bypass __init__ and wire a mock predictor directly."""
    suggester = object.__new__(_LabelSuggester)
    suggester._predictor = predictor_mock
    suggester._aw_host = "http://localhost:5600"
    suggester._title_salt = "test-salt"
    return suggester


class TestLabelSuggester:

    @patch("taskclf.adapters.activitywatch.client.find_window_bucket_id", return_value="aw-watcher-window_test")
    @patch("taskclf.adapters.activitywatch.client.fetch_aw_events")
    @patch("taskclf.features.build.build_features_from_aw_events")
    def test_successful_suggestion(
        self,
        mock_build: MagicMock,
        mock_fetch: MagicMock,
        _mock_find: MagicMock,
    ) -> None:
        """TC-UI-SUG-001"""
        mock_event = MagicMock()
        mock_fetch.return_value = [mock_event]

        mock_row = MagicMock()
        mock_build.return_value = [mock_row]

        mock_prediction = MagicMock()
        mock_prediction.core_label_name = "Build"
        mock_prediction.confidence = 0.92

        predictor = MagicMock()
        predictor.predict_bucket.return_value = mock_prediction

        suggester = _make_suggester(predictor)
        result = suggester.suggest(_BLOCK_START, _BLOCK_END)

        assert result is not None
        label, confidence = result
        assert label == "Build"
        assert confidence == 0.92
        predictor.predict_bucket.assert_called_once_with(mock_row)

    @patch("taskclf.adapters.activitywatch.client.find_window_bucket_id", return_value="aw-watcher-window_test")
    @patch("taskclf.adapters.activitywatch.client.fetch_aw_events", return_value=[])
    def test_no_events_returns_none(
        self, _mock_fetch: MagicMock, _mock_find: MagicMock,
    ) -> None:
        """TC-UI-SUG-002"""
        predictor = MagicMock()
        suggester = _make_suggester(predictor)

        result = suggester.suggest(_BLOCK_START, _BLOCK_END)

        assert result is None
        predictor.predict_bucket.assert_not_called()

    @patch("taskclf.adapters.activitywatch.client.find_window_bucket_id", return_value="aw-watcher-window_test")
    @patch("taskclf.adapters.activitywatch.client.fetch_aw_events")
    @patch("taskclf.features.build.build_features_from_aw_events", return_value=[])
    def test_no_features_returns_none(
        self,
        _mock_build: MagicMock,
        mock_fetch: MagicMock,
        _mock_find: MagicMock,
    ) -> None:
        """TC-UI-SUG-003"""
        mock_fetch.return_value = [MagicMock()]

        predictor = MagicMock()
        suggester = _make_suggester(predictor)

        result = suggester.suggest(_BLOCK_START, _BLOCK_END)

        assert result is None
        predictor.predict_bucket.assert_not_called()

    @patch(
        "taskclf.adapters.activitywatch.client.find_window_bucket_id",
        side_effect=ConnectionError("AW unavailable"),
    )
    def test_prediction_exception_returns_none(
        self, _mock_find: MagicMock,
    ) -> None:
        """TC-UI-SUG-004"""
        predictor = MagicMock()
        suggester = _make_suggester(predictor)

        result = suggester.suggest(_BLOCK_START, _BLOCK_END)

        assert result is None
