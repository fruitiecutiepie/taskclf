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


def _t(seconds: int) -> dt.datetime:
    """Return a test timestamp offset by *seconds* from a fixed epoch (UTC-aware)."""
    return dt.datetime(2026, 3, 1, 10, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=seconds)


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

        for i in range(10):
            monitor.check_transition("com.apple.Terminal", _now=_t(i * 60))

        assert len(transitions) == 0

    def test_transition_fires_after_threshold(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        assert monitor.current_app == "com.apple.Terminal"

        # 3 consecutive polls with a new app (3 * 60s = 180s = 3 min)
        monitor.check_transition("us.zoom.xos", _now=_t(60))
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos", _now=_t(120))
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos", _now=_t(180))
        assert len(transitions) == 1

        prev, new, start, end = transitions[0]
        assert prev == "com.apple.Terminal"
        assert new == "us.zoom.xos"
        assert start == _t(0)
        assert end == _t(60)
        assert monitor.current_app == "us.zoom.xos"

    def test_no_transition_if_app_flips_back(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))

        # Switch to Zoom for 2 polls (under threshold), then back
        monitor.check_transition("us.zoom.xos", _now=_t(60))
        monitor.check_transition("us.zoom.xos", _now=_t(120))
        monitor.check_transition("com.apple.Terminal", _now=_t(180))

        assert len(transitions) == 0
        assert monitor.current_app == "com.apple.Terminal"

    def test_transition_resets_after_firing(self) -> None:
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=2, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        monitor.check_transition("us.zoom.xos", _now=_t(60))
        monitor.check_transition("us.zoom.xos", _now=_t(120))
        assert len(transitions) == 1

        # Now stable on Zoom, transition to Slack
        monitor.check_transition("us.zoom.xos", _now=_t(180))
        monitor.check_transition("com.tinyspeck.slackmacgap", _now=_t(240))
        monitor.check_transition("com.tinyspeck.slackmacgap", _now=_t(300))
        assert len(transitions) == 2
        assert transitions[1][0] == "us.zoom.xos"
        assert transitions[1][1] == "com.tinyspeck.slackmacgap"

    def test_candidate_resets_on_third_app(self) -> None:
        """If the app changes to a third app during candidate phase, candidate resets."""
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        monitor.check_transition("us.zoom.xos", _now=_t(60))  # candidate = Zoom
        monitor.check_transition("com.tinyspeck.slackmacgap", _now=_t(120))  # candidate resets to Slack
        monitor.check_transition("com.tinyspeck.slackmacgap", _now=_t(180))
        monitor.check_transition("com.tinyspeck.slackmacgap", _now=_t(240))
        assert len(transitions) == 1
        assert transitions[0][1] == "com.tinyspeck.slackmacgap"

    def test_first_app_sets_current(self) -> None:
        monitor = self._make_monitor()
        assert monitor.current_app is None
        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        assert monitor.current_app == "com.apple.Terminal"

    def test_short_threshold(self) -> None:
        """With transition_minutes=1 and poll_seconds=60, two polls trigger (first sets candidate, second confirms)."""
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=1, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        monitor.check_transition("us.zoom.xos", _now=_t(60))  # candidate set, duration=60
        assert len(transitions) == 0
        monitor.check_transition("us.zoom.xos", _now=_t(120))  # duration=120 >= 60, fires
        assert len(transitions) == 1

    def test_delayed_poll_accumulates_wall_clock_time(self) -> None:
        """A 90-second gap between polls accumulates 90s, not poll_seconds (60s)."""
        transitions: list = []
        monitor = self._make_monitor(
            transition_minutes=3, poll_seconds=60, transitions=transitions,
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        # Each poll delayed to 90s apart instead of 60s
        monitor.check_transition("us.zoom.xos", _now=_t(90))   # candidate, duration=90
        monitor.check_transition("us.zoom.xos", _now=_t(180))  # duration=180 >= 180, fires
        assert len(transitions) == 1
        prev, new, start, end = transitions[0]
        assert new == "us.zoom.xos"
        assert start == _t(0)
        assert end == _t(90)


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

    def test_none_user_overlaps_concrete_user(self, tmp_path: Path) -> None:
        """A span with user_id=None must overlap-check against concrete user spans."""
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 15, 0, 0)

        span1 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=10),
            end_ts=now,
            label="Build",
            provenance="manual",
            user_id="alice",
        )
        append_label_span(span1, labels_path)

        span2 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=5),
            end_ts=now + dt.timedelta(minutes=5),
            label="Meet",
            provenance="weak:app_rule",
            user_id=None,
        )
        with pytest.raises(ValueError, match="overlaps"):
            append_label_span(span2, labels_path)


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


_BLOCK_START = dt.datetime(2026, 3, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
_BLOCK_END = dt.datetime(2026, 3, 1, 10, 15, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# 46a — TrayLabeler._handle_transition event publishing
# ---------------------------------------------------------------------------


class TestHandleTransition:

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_transition_without_model_publishes_prompt_and_no_model_transition(
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
        assert "no_model_transition" in types
        assert "prediction" not in types

        prompt = next(e for e in captured if e["type"] == "prompt_label")
        assert prompt["suggested_label"] is None
        assert prompt["suggested_confidence"] is None
        assert prompt["prev_app"] == "com.apple.Terminal"
        assert prompt["new_app"] == "us.zoom.xos"

        nmt = next(e for e in captured if e["type"] == "no_model_transition")
        assert nmt["current_app"] == "us.zoom.xos"
        assert nmt["ts"] == _BLOCK_END.isoformat()
        assert nmt["block_start"] == _BLOCK_START.isoformat()
        assert nmt["block_end"] == _BLOCK_END.isoformat()

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
            "paused",
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
# 46b′ — TrayLabeler._on_label_saved counter (Item 9)
# ---------------------------------------------------------------------------


class TestOnLabelSaved:

    def test_counter_increments(self, tmp_path: Path) -> None:
        """TC-UI-TRAY-010: _on_label_saved increments _labels_saved_count."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._labels_saved_count == 0

        labeler._on_label_saved()
        assert labeler._labels_saved_count == 1

        labeler._on_label_saved()
        labeler._on_label_saved()
        assert labeler._labels_saved_count == 3

    def test_counter_reflected_in_tray_state(self, tmp_path: Path) -> None:
        """TC-UI-TRAY-011: tray_state event includes updated labels_saved_count."""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._on_label_saved()
        labeler._on_label_saved()
        labeler._handle_poll("com.apple.Safari")

        tray_state = next(e for e in captured if e["type"] == "tray_state")
        assert tray_state["labels_saved_count"] == 2


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
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)

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
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)

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
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)
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

    @patch("taskclf.ui.tray.subprocess.run")
    @patch("taskclf.ui.tray.platform.system", return_value="Darwin")
    def test_newlines_and_quotes_escaped(
        self, _mock_sys: MagicMock, mock_run: MagicMock,
    ) -> None:
        """Regression: newlines and double quotes must not break the AppleScript."""
        _send_desktop_notification('Has "quotes"', "Line1\nLine2")

        mock_run.assert_called_once()
        script = mock_run.call_args[0][0][2]
        assert "\n" not in script
        assert '"Has \\"quotes\\""' in script or '\\"quotes\\"' in script
        assert "Line1 Line2" in script


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


# ---------------------------------------------------------------------------
# Pause/resume  (Item 4)
# ---------------------------------------------------------------------------


class TestPauseResume:

    def test_monitor_starts_unpaused(self) -> None:
        monitor = ActivityMonitor(poll_seconds=60, transition_minutes=3)
        assert monitor.is_paused is False

    def test_pause_sets_flag(self) -> None:
        monitor = ActivityMonitor(poll_seconds=60, transition_minutes=3)
        monitor.pause()
        assert monitor.is_paused is True

    def test_resume_clears_flag(self) -> None:
        monitor = ActivityMonitor(poll_seconds=60, transition_minutes=3)
        monitor.pause()
        monitor.resume()
        assert monitor.is_paused is False

    def test_publish_status_paused_state(self) -> None:
        """When paused, _publish_status emits state='paused'."""
        bus, captured = _capture_bus()
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3, event_bus=bus,
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)

        monitor._publish_status("app1", state="paused")

        assert len(captured) == 1
        assert captured[0]["state"] == "paused"
        assert captured[0]["current_app"] == "app1"

    def test_publish_status_defaults_to_collecting(self) -> None:
        bus, captured = _capture_bus()
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3, event_bus=bus,
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)

        monitor._publish_status("app1")

        assert captured[0]["state"] == "collecting"

    def test_session_state_preserved_across_pause(self) -> None:
        """poll_count and transition state survive pause/resume."""
        bus, captured = _capture_bus()
        transitions: list = []
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=2,
            event_bus=bus, on_transition=lambda *a: transitions.append(a),
        )
        monitor._started_at = dt.datetime(2026, 3, 1, 9, 0, 0, tzinfo=dt.timezone.utc)

        monitor._publish_status("app1")
        monitor.check_transition("app1", _now=_t(0))
        assert monitor._poll_count == 1

        monitor.pause()
        assert monitor.is_paused is True
        assert monitor._poll_count == 1

        monitor.resume()
        monitor._publish_status("app1")
        assert monitor._poll_count == 2
        assert monitor.current_app == "app1"

    def test_pause_resume_does_not_corrupt_elapsed(self) -> None:
        """After a long pause, elapsed should not include the pause gap."""
        transitions: list = []
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3,
            on_transition=lambda *a: transitions.append(a),
        )

        monitor.check_transition("app1", _now=_t(0))
        monitor.check_transition("app2", _now=_t(60))
        assert monitor._candidate_app == "app2"
        assert monitor._candidate_duration == 60

        monitor.pause()
        # Simulate 30 minutes passing while paused
        monitor.resume()

        # After resume, _last_check_time is None so elapsed defaults
        # to poll_seconds (60), not the 1800s pause gap.
        monitor.check_transition("app2", _now=_t(1860))
        assert monitor._candidate_duration == 120
        assert len(transitions) == 0

    def test_tray_toggle_pause(self, tmp_path: Path) -> None:
        """TrayLabeler._toggle_pause toggles the monitor's paused state."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        assert labeler._monitor.is_paused is False
        result = labeler._toggle_pause()
        assert result is True
        assert labeler._monitor.is_paused is True

        result = labeler._toggle_pause()
        assert result is False
        assert labeler._monitor.is_paused is False

    def test_tray_state_includes_paused_field(self, tmp_path: Path) -> None:
        """tray_state event includes 'paused' key."""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._handle_poll("app1")
        tray_state = next(e for e in captured if e["type"] == "tray_state")
        assert "paused" in tray_state
        assert tray_state["paused"] is False

        captured.clear()
        labeler._toggle_pause()
        labeler._handle_poll("app1")
        tray_state = next(e for e in captured if e["type"] == "tray_state")
        assert tray_state["paused"] is True


# ---------------------------------------------------------------------------
# Notification privacy  (Item 5)
# ---------------------------------------------------------------------------


class TestNotificationPrivacy:

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_notifications_disabled_skips_call(
        self, mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None, event_bus=bus,
            notifications_enabled=False,
        )
        labeler._send_notification(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )
        mock_notif.assert_not_called()

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_privacy_mode_redacts_app_names(
        self, mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """Default privacy_notifications=True hides app identifiers."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None, event_bus=bus,
            privacy_notifications=True,
        )
        labeler._send_notification(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )
        mock_notif.assert_called_once()
        message = mock_notif.call_args[1].get("message") or mock_notif.call_args[0][1]
        assert "com.apple.Terminal" not in message
        assert "us.zoom.xos" not in message
        assert "Activity changed" in message
        assert "15 min" in message

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_privacy_off_shows_raw_app_names(
        self, mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None, event_bus=bus,
            privacy_notifications=False,
        )
        labeler._send_notification(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )
        mock_notif.assert_called_once()
        message = mock_notif.call_args[0][1]
        assert "com.apple.Terminal" in message
        assert "us.zoom.xos" in message

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_privacy_default_is_true(
        self, mock_notif: MagicMock, tmp_path: Path,
    ) -> None:
        """Default TrayLabeler has privacy_notifications=True."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._send_notification(
            "com.apple.Terminal", "us.zoom.xos", _BLOCK_START, _BLOCK_END,
        )
        message = mock_notif.call_args[0][1]
        assert "com.apple.Terminal" not in message


# ---------------------------------------------------------------------------
# Cold start gap  (Item 7)
# ---------------------------------------------------------------------------


class TestColdStart:

    def test_first_poll_fires_on_initial_app(self) -> None:
        """on_initial_app fires on the very first check_transition call."""
        initial_calls: list = []
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3,
            on_initial_app=lambda app, ts: initial_calls.append((app, ts)),
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))

        assert len(initial_calls) == 1
        assert initial_calls[0][0] == "com.apple.Terminal"
        assert initial_calls[0][1] == _t(0)

    def test_subsequent_polls_do_not_refire(self) -> None:
        """on_initial_app fires only once, not on subsequent polls."""
        initial_calls: list = []
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3,
            on_initial_app=lambda app, ts: initial_calls.append((app, ts)),
        )

        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        monitor.check_transition("com.apple.Terminal", _now=_t(60))
        monitor.check_transition("us.zoom.xos", _now=_t(120))

        assert len(initial_calls) == 1

    def test_no_callback_is_safe(self) -> None:
        """on_initial_app=None doesn't raise on first poll."""
        monitor = ActivityMonitor(
            poll_seconds=60, transition_minutes=3,
            on_initial_app=None,
        )
        monitor.check_transition("com.apple.Terminal", _now=_t(0))
        assert monitor.current_app == "com.apple.Terminal"

    def test_tray_publishes_initial_app_event(self, tmp_path: Path) -> None:
        """TrayLabeler._handle_initial_app publishes an initial_app event."""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._handle_initial_app("com.apple.Terminal", _t(0))

        assert len(captured) == 1
        event = captured[0]
        assert event["type"] == "initial_app"
        assert event["app"] == "com.apple.Terminal"
        assert event["ts"] == _t(0).isoformat()

    def test_monitor_wired_to_tray_initial_app(self, tmp_path: Path) -> None:
        """ActivityMonitor fires on_initial_app which TrayLabeler publishes."""
        bus, captured = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labeler._monitor.check_transition("com.apple.Safari", _now=_t(0))

        initial_events = [e for e in captured if e["type"] == "initial_app"]
        assert len(initial_events) == 1
        assert initial_events[0]["app"] == "com.apple.Safari"


# ---------------------------------------------------------------------------
# Server startup unification  (Item 1)
# ---------------------------------------------------------------------------


class TestServerStartup:
    """Verify that both --browser and native-window modes run FastAPI
    in-process with the tray's shared EventBus."""

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    @patch("taskclf.ui.server.create_app")
    def test_start_server_passes_shared_event_bus(
        self, mock_create_app: MagicMock, mock_config: MagicMock,
        mock_server_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """_start_server() must pass self._event_bus to create_app()."""
        mock_create_app.return_value = MagicMock()
        mock_server_cls.return_value = MagicMock()

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._start_server()

        mock_create_app.assert_called_once()
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["event_bus"] is bus

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    @patch("taskclf.ui.server.create_app")
    def test_start_server_passes_callbacks(
        self, mock_create_app: MagicMock, mock_config: MagicMock,
        mock_server_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """_start_server() must wire on_label_saved, pause_toggle, is_paused."""
        mock_create_app.return_value = MagicMock()
        mock_server_cls.return_value = MagicMock()

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._start_server()

        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["on_label_saved"] == labeler._on_label_saved
        assert call_kwargs["pause_toggle"] == labeler._toggle_pause
        assert callable(call_kwargs["is_paused"])

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    @patch("taskclf.ui.server.create_app")
    def test_start_server_sets_ui_server_running(
        self, mock_create_app: MagicMock, mock_config: MagicMock,
        mock_server_cls: MagicMock, tmp_path: Path,
    ) -> None:
        mock_create_app.return_value = MagicMock()
        mock_server_cls.return_value = MagicMock()

        labeler = _make_tray_labeler(tmp_path)
        assert labeler._ui_server_running is False
        labeler._start_server()
        assert labeler._ui_server_running is True

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    @patch("taskclf.ui.server.create_app")
    def test_start_server_returns_ui_port(
        self, mock_create_app: MagicMock, mock_config: MagicMock,
        mock_server_cls: MagicMock, tmp_path: Path,
    ) -> None:
        mock_create_app.return_value = MagicMock()
        mock_server_cls.return_value = MagicMock()

        labeler = TrayLabeler(data_dir=tmp_path, ui_port=9999)
        port = labeler._start_server()
        assert port == 9999

    @patch("taskclf.ui.tray.TrayLabeler._spawn_window")
    @patch("taskclf.ui.tray.TrayLabeler._start_server")
    def test_subprocess_mode_calls_start_server_then_spawn_window(
        self, mock_start: MagicMock, mock_spawn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_start_ui_subprocess() must call _start_server() then _spawn_window()."""
        labeler = _make_tray_labeler(tmp_path)
        labeler._start_ui_subprocess()

        mock_start.assert_called_once()
        mock_spawn.assert_called_once()

    @patch("taskclf.ui.tray.TrayLabeler._start_server", return_value=8741)
    def test_embedded_mode_calls_start_server(
        self, mock_start: MagicMock, tmp_path: Path,
    ) -> None:
        """_start_ui_embedded() must call _start_server()."""
        labeler = TrayLabeler(data_dir=tmp_path, browser=True)
        with patch("webbrowser.open"):
            labeler._start_ui_embedded()
        mock_start.assert_called_once()

    @patch("subprocess.Popen")
    def test_spawn_window_launches_pywebview_module(
        self, mock_popen: MagicMock, tmp_path: Path,
    ) -> None:
        """_spawn_window() must invoke python -m taskclf.ui.window."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        labeler = TrayLabeler(data_dir=tmp_path, ui_port=8741)
        labeler._spawn_window()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "taskclf.ui.window" in cmd
        assert "--port" in cmd
        assert "8741" in cmd
        assert "taskclf.cli.main" not in cmd

    @patch("subprocess.Popen")
    def test_spawn_window_failure_does_not_raise(
        self, mock_popen: MagicMock, tmp_path: Path,
    ) -> None:
        """_spawn_window() gracefully handles subprocess failures."""
        mock_popen.side_effect = OSError("pywebview not installed")

        labeler = _make_tray_labeler(tmp_path)
        labeler._spawn_window()
        assert labeler._ui_proc is None


class TestOpenDashboard:
    """Verify _open_dashboard behaviour for both browser and native modes."""

    def test_browser_mode_opens_webbrowser(self, tmp_path: Path) -> None:
        bus, _ = _capture_bus()
        labeler = TrayLabeler(data_dir=tmp_path, browser=True, event_bus=bus)
        with patch("webbrowser.open") as mock_open:
            labeler._open_dashboard()
            mock_open.assert_called_once()
            url = mock_open.call_args[0][0]
            assert "127.0.0.1" in url

    def test_native_mode_noop_when_window_alive(self, tmp_path: Path) -> None:
        """If pywebview subprocess is still running, _open_dashboard is a no-op."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        labeler._ui_proc = mock_proc

        with patch.object(labeler, "_spawn_window") as mock_spawn:
            labeler._open_dashboard()
            mock_spawn.assert_not_called()

    def test_native_mode_respawns_when_window_exited(self, tmp_path: Path) -> None:
        """If pywebview subprocess exited, _open_dashboard restarts it."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # exited
        labeler._ui_proc = mock_proc

        with patch.object(labeler, "_spawn_window") as mock_spawn:
            labeler._open_dashboard()
            mock_spawn.assert_called_once()

    def test_native_mode_spawns_when_no_prior_proc(self, tmp_path: Path) -> None:
        """If no subprocess was ever started, _open_dashboard spawns one."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._ui_proc is None

        with patch.object(labeler, "_spawn_window") as mock_spawn:
            labeler._open_dashboard()
            mock_spawn.assert_called_once()


class TestTrayMenuExportLabels:
    """Menu includes 'Export Labels' item and the callback works."""

    def test_build_menu_contains_export_labels(self, tmp_path: Path) -> None:
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        menu = labeler._build_menu()
        labels = [
            item.text if isinstance(item.text, str) else item.text(None)
            for item in menu.items
            if hasattr(item, "text") and item.text is not None
        ]
        assert "Export Labels" in labels

    def test_export_labels_success(self, tmp_path: Path) -> None:
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labels_dir = tmp_path / "labels_v1"
        labels_dir.mkdir(parents=True)
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
        )
        from taskclf.labels.store import write_label_spans
        write_label_spans([span], labels_dir / "labels.parquet")

        csv_dest = tmp_path / "out.csv"
        mock_tk = MagicMock()
        mock_tk.filedialog.asksaveasfilename.return_value = str(csv_dest)

        with (
            patch.dict(sys.modules, {"tkinter": mock_tk, "tkinter.filedialog": mock_tk.filedialog}),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._export_labels()

        assert csv_dest.exists()
        mock_notify.assert_called_once()
        assert "exported" in mock_notify.call_args[0][0].lower()

    def test_export_labels_no_file_notifies_error(self, tmp_path: Path) -> None:
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        csv_dest = tmp_path / "out.csv"
        mock_tk = MagicMock()
        mock_tk.filedialog.asksaveasfilename.return_value = str(csv_dest)

        with (
            patch.dict(sys.modules, {"tkinter": mock_tk, "tkinter.filedialog": mock_tk.filedialog}),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._export_labels()

        assert not csv_dest.exists()
        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()

    def test_export_labels_cancel_dialog(self, tmp_path: Path) -> None:
        """User cancels the save dialog — no export happens."""
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        mock_tk = MagicMock()
        mock_tk.filedialog.asksaveasfilename.return_value = ""

        with (
            patch.dict(sys.modules, {"tkinter": mock_tk, "tkinter.filedialog": mock_tk.filedialog}),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._export_labels()

        mock_notify.assert_not_called()


# ---------------------------------------------------------------------------
# Import Labels  (Item 2)
# ---------------------------------------------------------------------------


class TestImportLabels:
    """Tests for TrayLabeler._import_labels tray callback."""

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        import csv
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _sample_csv_rows(self) -> list[dict]:
        return [
            {
                "start_ts": "2026-03-01 09:00:00",
                "end_ts": "2026-03-01 09:30:00",
                "label": "Build",
                "provenance": "manual",
            },
            {
                "start_ts": "2026-03-01 10:00:00",
                "end_ts": "2026-03-01 10:30:00",
                "label": "Meet",
                "provenance": "manual",
            },
        ]

    def test_import_success_merge(self, tmp_path: Path) -> None:
        """Import with merge strategy adds new labels alongside existing."""
        import sys
        from taskclf.labels.store import write_label_spans

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labels_dir = tmp_path / "labels_v1"
        labels_dir.mkdir(parents=True)
        existing_span = LabelSpan(
            start_ts=dt.datetime(2026, 3, 1, 8, 0),
            end_ts=dt.datetime(2026, 3, 1, 8, 30),
            label="Write",
            provenance="manual",
        )
        write_label_spans([existing_span], labels_dir / "labels.parquet")

        csv_file = tmp_path / "import.csv"
        self._write_csv(csv_file, self._sample_csv_rows())

        mock_tk = MagicMock()
        mock_tk.filedialog.askopenfilename.return_value = str(csv_file)
        mock_tk.messagebox.askyesnocancel.return_value = True  # merge

        with (
            patch.dict(sys.modules, {
                "tkinter": mock_tk,
                "tkinter.filedialog": mock_tk.filedialog,
                "tkinter.messagebox": mock_tk.messagebox,
            }),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._import_labels()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "imported" in msg.lower()
        assert "2" in msg

        loaded = read_label_spans(labels_dir / "labels.parquet")
        assert len(loaded) == 3

    def test_import_success_overwrite(self, tmp_path: Path) -> None:
        """Import with overwrite strategy replaces all existing labels."""
        import sys
        from taskclf.labels.store import write_label_spans

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labels_dir = tmp_path / "labels_v1"
        labels_dir.mkdir(parents=True)
        existing_span = LabelSpan(
            start_ts=dt.datetime(2026, 3, 1, 8, 0),
            end_ts=dt.datetime(2026, 3, 1, 8, 30),
            label="Write",
            provenance="manual",
        )
        write_label_spans([existing_span], labels_dir / "labels.parquet")

        csv_file = tmp_path / "import.csv"
        self._write_csv(csv_file, self._sample_csv_rows())

        mock_tk = MagicMock()
        mock_tk.filedialog.askopenfilename.return_value = str(csv_file)
        mock_tk.messagebox.askyesnocancel.return_value = False  # overwrite

        with (
            patch.dict(sys.modules, {
                "tkinter": mock_tk,
                "tkinter.filedialog": mock_tk.filedialog,
                "tkinter.messagebox": mock_tk.messagebox,
            }),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._import_labels()

        mock_notify.assert_called_once()
        assert "imported" in mock_notify.call_args[0][0].lower()

        loaded = read_label_spans(labels_dir / "labels.parquet")
        assert len(loaded) == 2
        labels = {s.label for s in loaded}
        assert "Write" not in labels

    def test_import_cancel_file_dialog(self, tmp_path: Path) -> None:
        """Canceling the file dialog performs no import."""
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        mock_tk = MagicMock()
        mock_tk.filedialog.askopenfilename.return_value = ""

        with (
            patch.dict(sys.modules, {
                "tkinter": mock_tk,
                "tkinter.filedialog": mock_tk.filedialog,
                "tkinter.messagebox": mock_tk.messagebox,
            }),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._import_labels()

        mock_notify.assert_not_called()
        mock_tk.messagebox.askyesnocancel.assert_not_called()

    def test_import_cancel_strategy_dialog(self, tmp_path: Path) -> None:
        """Canceling the strategy dialog performs no import."""
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        csv_file = tmp_path / "import.csv"
        self._write_csv(csv_file, self._sample_csv_rows())

        mock_tk = MagicMock()
        mock_tk.filedialog.askopenfilename.return_value = str(csv_file)
        mock_tk.messagebox.askyesnocancel.return_value = None  # cancel

        with (
            patch.dict(sys.modules, {
                "tkinter": mock_tk,
                "tkinter.filedialog": mock_tk.filedialog,
                "tkinter.messagebox": mock_tk.messagebox,
            }),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._import_labels()

        mock_notify.assert_not_called()

    def test_import_failure_bad_csv(self, tmp_path: Path) -> None:
        """Invalid CSV (missing columns) notifies failure."""
        import sys

        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("col_a,col_b\n1,2\n")

        mock_tk = MagicMock()
        mock_tk.filedialog.askopenfilename.return_value = str(csv_file)
        mock_tk.messagebox.askyesnocancel.return_value = True

        with (
            patch.dict(sys.modules, {
                "tkinter": mock_tk,
                "tkinter.filedialog": mock_tk.filedialog,
                "tkinter.messagebox": mock_tk.messagebox,
            }),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._import_labels()

        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()

    def test_build_menu_contains_import_labels(self, tmp_path: Path) -> None:
        """Menu structure includes Import Labels item."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        menu = labeler._build_menu()
        labels = [
            item.text if isinstance(item.text, str) else item.text(None)
            for item in menu.items
            if hasattr(item, "text") and item.text is not None
        ]
        assert "Import Labels" in labels


# ---------------------------------------------------------------------------
# Label Stats  (Item 1)
# ---------------------------------------------------------------------------


class TestLabelStats:
    """Tests for TrayLabeler._label_stats notification."""

    def test_no_file_notifies_no_labels(self, tmp_path: Path) -> None:
        """When the labels file doesn't exist, show 'No labels yet'."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._label_stats()

        mock_notify.assert_called_once()
        assert "no labels" in mock_notify.call_args[0][0].lower()

    def test_no_labels_today(self, tmp_path: Path) -> None:
        """Labels exist but none from today → 'no labels yet'."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labels_dir = tmp_path / "labels_v1"
        labels_dir.mkdir(parents=True)
        yesterday = dt.datetime(2020, 1, 1, 10, 0, 0)
        span = LabelSpan(
            start_ts=yesterday,
            end_ts=yesterday + dt.timedelta(minutes=10),
            label="Build",
            provenance="manual",
        )
        from taskclf.labels.store import write_label_spans
        write_label_spans([span], labels_dir / "labels.parquet")

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._label_stats()

        mock_notify.assert_called_once()
        assert "no labels" in mock_notify.call_args[0][0].lower()

    def test_today_labels_summary(self, tmp_path: Path) -> None:
        """Labels from today produce a summary with count, time, breakdown."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        labels_dir = tmp_path / "labels_v1"
        labels_dir.mkdir(parents=True)

        now = dt.datetime.now(dt.timezone.utc)
        today = now.replace(tzinfo=None)
        base = dt.datetime(today.year, today.month, today.day, 9, 0, 0)
        spans = [
            LabelSpan(
                start_ts=base,
                end_ts=base + dt.timedelta(minutes=45),
                label="Build",
                provenance="manual",
            ),
            LabelSpan(
                start_ts=base + dt.timedelta(minutes=45),
                end_ts=base + dt.timedelta(minutes=65),
                label="Meet",
                provenance="manual",
            ),
        ]
        from taskclf.labels.store import write_label_spans
        write_label_spans(spans, labels_dir / "labels.parquet")

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._label_stats()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "2 labels" in msg
        assert "Build" in msg
        assert "Meet" in msg
        assert "45m" in msg
        assert "20m" in msg


# ---------------------------------------------------------------------------
# Open Data Directory  (Item 4)
# ---------------------------------------------------------------------------


class TestOpenDataDir:
    """Tests for TrayLabeler._open_data_dir."""

    @patch("taskclf.ui.tray.subprocess.Popen")
    @patch("taskclf.ui.tray.platform.system", return_value="Darwin")
    def test_macos_calls_open(
        self, _mock_sys: MagicMock, mock_popen: MagicMock, tmp_path: Path,
    ) -> None:
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._open_data_dir()

        mock_popen.assert_called_once_with(["open", str(tmp_path)])

    @patch("taskclf.ui.tray.subprocess.Popen")
    @patch("taskclf.ui.tray.platform.system", return_value="Linux")
    def test_linux_calls_xdg_open(
        self, _mock_sys: MagicMock, mock_popen: MagicMock, tmp_path: Path,
    ) -> None:
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._open_data_dir()

        mock_popen.assert_called_once_with(["xdg-open", str(tmp_path)])

    @patch("taskclf.ui.tray.subprocess.Popen", side_effect=OSError("no open"))
    @patch("taskclf.ui.tray.platform.system", return_value="Darwin")
    def test_fallback_notifies_path_on_error(
        self, _mock_sys: MagicMock, _mock_popen: MagicMock, tmp_path: Path,
    ) -> None:
        """When Popen fails, fall back to a notification showing the path."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._open_data_dir()

        mock_notify.assert_called_once()
        assert str(tmp_path) in mock_notify.call_args[0][0]


# ---------------------------------------------------------------------------
# Reload Model  (Item 5)
# ---------------------------------------------------------------------------


class TestReloadModel:
    """Tests for TrayLabeler._reload_model."""

    def test_no_model_dir_notifies(self, tmp_path: Path) -> None:
        """Without model_dir, notify 'No model directory configured'."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._model_dir is None

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._reload_model()

        mock_notify.assert_called_once()
        assert "no model" in mock_notify.call_args[0][0].lower()

    def test_reload_success(self, tmp_path: Path) -> None:
        """Successful reload updates _suggester and notifies."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._model_dir = tmp_path / "models" / "run_test"

        mock_suggester = MagicMock()
        mock_suggester._predictor._metadata.schema_hash = "abc123"

        with (
            patch("taskclf.ui.tray._LabelSuggester", return_value=mock_suggester) as mock_cls,
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._reload_model()

        mock_cls.assert_called_once_with(labeler._model_dir)
        assert labeler._suggester is mock_suggester
        assert labeler._model_schema_hash == "abc123"
        mock_notify.assert_called_once()
        assert "reloaded" in mock_notify.call_args[0][0].lower()

    def test_reload_failure_keeps_old_model(self, tmp_path: Path) -> None:
        """On failure, keep old _suggester and notify error."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._model_dir = tmp_path / "models" / "run_test"
        old_suggester = MagicMock()
        labeler._suggester = old_suggester

        with (
            patch("taskclf.ui.tray._LabelSuggester", side_effect=RuntimeError("bad model")),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._reload_model()

        assert labeler._suggester is old_suggester
        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Connection Status  (Item 6)
# ---------------------------------------------------------------------------


class TestShowStatus:
    """Tests for TrayLabeler._show_status."""

    def test_status_notification_content(self, tmp_path: Path) -> None:
        """Status notification contains AW, polls, transitions, labels, model info."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._monitor._poll_count = 42
        labeler._transition_count = 3
        labeler._labels_saved_count = 7

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._show_status()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "AW: disconnected" in msg
        assert "Polls: 42" in msg
        assert "Transitions: 3" in msg
        assert "Labels: 7" in msg
        assert "Model: none" in msg

    def test_status_connected_with_model(self, tmp_path: Path) -> None:
        """Status shows 'connected' when bucket_id is set and model name."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._monitor._bucket_id = "aw-watcher-window_test"
        labeler._model_dir = Path("models/run_20260301")

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._show_status()

        msg = mock_notify.call_args[0][0]
        assert "AW: connected" in msg
        assert "Model: run_20260301" in msg

    def test_status_paused(self, tmp_path: Path) -> None:
        """Status includes '(paused)' when monitoring is paused."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._monitor.pause()

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._show_status()

        msg = mock_notify.call_args[0][0]
        assert "(paused)" in msg


# ---------------------------------------------------------------------------
# Build menu includes new items
# ---------------------------------------------------------------------------


class TestBuildMenuEnhancements:
    """Verify the updated menu structure includes new items."""

    def _get_top_level_labels(self, menu: object) -> list[str]:
        labels = []
        for item in menu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text is not None:
                text = item.text if isinstance(item.text, str) else item.text(None)
                if text:
                    labels.append(text)
        return labels

    def _find_model_submenu(self, menu: object) -> object | None:
        for item in menu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Model":
                return item.submenu
        return None

    def _get_submenu_labels(self, submenu: object) -> list[str]:
        labels = []
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text is not None:
                text = item.text if isinstance(item.text, str) else item.text(None)
                if text:
                    labels.append(text)
        return labels

    def test_menu_contains_new_items(self, tmp_path: Path) -> None:
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        menu = labeler._build_menu()

        labels = self._get_top_level_labels(menu)

        assert "Label Stats" in labels
        assert "Status" in labels
        assert "Open Data Folder" in labels
        assert "Model" in labels
        assert "Export Labels" in labels

    def test_model_submenu_contains_reload_and_check_retrain(self, tmp_path: Path) -> None:
        """Reload Model and Check Retrain live inside the Model submenu, not at top level."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        menu = labeler._build_menu()

        top_labels = self._get_top_level_labels(menu)
        assert "Reload Model" not in top_labels
        assert "Check Retrain" not in top_labels

        submenu = self._find_model_submenu(menu)
        assert submenu is not None
        sub_labels = self._get_submenu_labels(submenu)
        assert "Reload Model" in sub_labels
        assert "Check Retrain" in sub_labels

    def test_reload_model_disabled_without_model_dir(self, tmp_path: Path) -> None:
        """Reload Model inside Model submenu is disabled when no model_dir."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._model_dir is None

        menu = labeler._build_menu()
        submenu = self._find_model_submenu(menu)
        assert submenu is not None

        reload_item = None
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Reload Model":
                reload_item = item
                break

        assert reload_item is not None
        assert reload_item.enabled is False

    def test_reload_model_enabled_with_model_dir(self, tmp_path: Path) -> None:
        """Reload Model inside Model submenu is enabled when model_dir is set."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        labeler._model_dir = Path("models/run_test")

        menu = labeler._build_menu()
        submenu = self._find_model_submenu(menu)
        assert submenu is not None

        reload_item = None
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Reload Model":
                reload_item = item
                break

        assert reload_item is not None
        assert reload_item.enabled is True


# ---------------------------------------------------------------------------
# Switch Model  (Item 3)
# ---------------------------------------------------------------------------


class TestSwitchModel:
    """Tests for the Model submenu, _switch_model, and _unload_model."""

    def _make_labeler_with_models_dir(
        self, tmp_path: Path, *, event_bus: EventBus | None = None,
    ) -> TrayLabeler:
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        bus = event_bus or _capture_bus()[0]
        return TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            models_dir=models_dir,
            event_bus=bus,
        )

    def _submenu_labels(self, submenu: object) -> list[str]:
        labels = []
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text is not None:
                text = item.text if isinstance(item.text, str) else item.text(None)
                if text:
                    labels.append(text)
        return labels

    def _find_model_submenu(self, menu: object) -> object:
        for item in menu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Model":
                return item.submenu
        pytest.fail("Model submenu not found in menu")

    # -- submenu listing --

    def test_submenu_lists_valid_bundles(self, tmp_path: Path) -> None:
        """Model submenu shows one item per valid bundle."""
        from taskclf.model_registry import ModelBundle

        labeler = self._make_labeler_with_models_dir(tmp_path)
        bundles = [
            ModelBundle(model_id="run_20260301", path=tmp_path / "models" / "run_20260301", valid=True),
            ModelBundle(model_id="run_20260226", path=tmp_path / "models" / "run_20260226", valid=True),
        ]

        with patch("taskclf.model_registry.list_bundles", return_value=bundles):
            submenu = labeler._build_model_submenu()

        labels = self._submenu_labels(submenu)
        assert "run_20260301" in labels
        assert "run_20260226" in labels
        assert "(No Model)" in labels
        assert "Reload Model" in labels

    def test_submenu_excludes_invalid_bundles(self, tmp_path: Path) -> None:
        """Invalid bundles are filtered out of the submenu."""
        from taskclf.model_registry import ModelBundle

        labeler = self._make_labeler_with_models_dir(tmp_path)
        bundles = [
            ModelBundle(model_id="run_good", path=tmp_path / "models" / "run_good", valid=True),
            ModelBundle(model_id="run_bad", path=tmp_path / "models" / "run_bad", valid=False, invalid_reason="missing metadata"),
        ]

        with patch("taskclf.model_registry.list_bundles", return_value=bundles):
            submenu = labeler._build_model_submenu()

        labels = self._submenu_labels(submenu)
        assert "run_good" in labels
        assert "run_bad" not in labels

    # -- checked state (radio-button effect) --

    def test_current_model_is_checked(self, tmp_path: Path) -> None:
        """The currently loaded model shows a check mark."""
        from taskclf.model_registry import ModelBundle

        model_path = tmp_path / "models" / "run_20260301"
        model_path.mkdir(parents=True)
        labeler = self._make_labeler_with_models_dir(tmp_path)
        labeler._model_dir = model_path

        bundles = [
            ModelBundle(model_id="run_20260301", path=model_path, valid=True),
            ModelBundle(model_id="run_20260226", path=tmp_path / "models" / "run_20260226", valid=True),
        ]

        with patch("taskclf.model_registry.list_bundles", return_value=bundles):
            submenu = labeler._build_model_submenu()

        for item in submenu.items:
            if hasattr(item, "text") and item.text == "run_20260301":
                assert item.checked is True
            elif hasattr(item, "text") and item.text == "run_20260226":
                assert item.checked is False
            elif hasattr(item, "text") and item.text == "(No Model)":
                assert item.checked is False

    def test_no_model_checked_when_unloaded(self, tmp_path: Path) -> None:
        """'(No Model)' is checked when no model is loaded."""
        from taskclf.model_registry import ModelBundle

        labeler = self._make_labeler_with_models_dir(tmp_path)
        assert labeler._model_dir is None

        bundles = [
            ModelBundle(model_id="run_20260301", path=tmp_path / "models" / "run_20260301", valid=True),
        ]

        with patch("taskclf.model_registry.list_bundles", return_value=bundles):
            submenu = labeler._build_model_submenu()

        for item in submenu.items:
            if hasattr(item, "text") and item.text == "(No Model)":
                assert item.checked is True
            elif hasattr(item, "text") and item.text == "run_20260301":
                assert item.checked is False

    # -- switch model --

    def test_switch_model_success(self, tmp_path: Path) -> None:
        """Switching to a new model updates _suggester, _model_dir, _model_schema_hash."""
        labeler = self._make_labeler_with_models_dir(tmp_path)
        new_path = tmp_path / "models" / "run_20260301"
        new_path.mkdir(parents=True)

        mock_suggester = MagicMock()
        mock_suggester._predictor._metadata.schema_hash = "hash_301"

        with (
            patch("taskclf.ui.tray._LabelSuggester", return_value=mock_suggester) as mock_cls,
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._switch_model(new_path)

        mock_cls.assert_called_once_with(new_path)
        assert labeler._suggester is mock_suggester
        assert labeler._model_dir == new_path
        assert labeler._model_schema_hash == "hash_301"
        mock_notify.assert_called_once()
        assert "switched" in mock_notify.call_args[0][0].lower()

    def test_switch_model_failure_keeps_old(self, tmp_path: Path) -> None:
        """On load failure, the previous model is preserved."""
        labeler = self._make_labeler_with_models_dir(tmp_path)
        old_suggester = MagicMock()
        old_path = tmp_path / "models" / "run_old"
        labeler._suggester = old_suggester
        labeler._model_dir = old_path
        labeler._model_schema_hash = "old_hash"

        new_path = tmp_path / "models" / "run_broken"

        with (
            patch("taskclf.ui.tray._LabelSuggester", side_effect=RuntimeError("corrupt model")),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._switch_model(new_path)

        assert labeler._suggester is old_suggester
        assert labeler._model_dir == old_path
        assert labeler._model_schema_hash == "old_hash"
        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()

    def test_switch_model_noop_if_same(self, tmp_path: Path) -> None:
        """Switching to the already-loaded model is a no-op."""
        labeler = self._make_labeler_with_models_dir(tmp_path)
        model_path = tmp_path / "models" / "run_20260301"
        model_path.mkdir(parents=True)
        labeler._model_dir = model_path

        with (
            patch("taskclf.ui.tray._LabelSuggester") as mock_cls,
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._switch_model(model_path)

        mock_cls.assert_not_called()
        mock_notify.assert_not_called()

    # -- unload model --

    def test_unload_model(self, tmp_path: Path) -> None:
        """Unloading clears _suggester, _model_dir, and suggestions."""
        labeler = self._make_labeler_with_models_dir(tmp_path)
        labeler._suggester = MagicMock()
        labeler._model_dir = tmp_path / "models" / "run_test"
        labeler._model_schema_hash = "some_hash"
        labeler._suggested_label = "Build"
        labeler._suggested_confidence = 0.9

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._unload_model()

        assert labeler._suggester is None
        assert labeler._model_dir is None
        assert labeler._model_schema_hash is None
        assert labeler._suggested_label is None
        assert labeler._suggested_confidence is None
        mock_notify.assert_called_once()
        assert "unloaded" in mock_notify.call_args[0][0].lower()

    # -- fallback: no models_dir --

    def test_no_models_dir_shows_fallback(self, tmp_path: Path) -> None:
        """When models_dir is None, submenu shows '(no models found)'."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            models_dir=None,
            event_bus=bus,
        )

        submenu = labeler._build_model_submenu()
        labels = self._submenu_labels(submenu)
        assert "(no models found)" in labels
        assert "Reload Model" in labels

    def test_empty_models_dir_shows_fallback(self, tmp_path: Path) -> None:
        """When models_dir exists but is empty, submenu shows '(no models found)'."""
        labeler = self._make_labeler_with_models_dir(tmp_path)

        with patch("taskclf.model_registry.list_bundles", return_value=[]):
            submenu = labeler._build_model_submenu()

        labels = self._submenu_labels(submenu)
        assert "(no models found)" in labels

    def test_all_invalid_bundles_shows_fallback(self, tmp_path: Path) -> None:
        """When all bundles are invalid, submenu shows '(no models found)'."""
        from taskclf.model_registry import ModelBundle

        labeler = self._make_labeler_with_models_dir(tmp_path)
        bundles = [
            ModelBundle(model_id="run_bad", path=tmp_path / "models" / "run_bad", valid=False, invalid_reason="broken"),
        ]

        with patch("taskclf.model_registry.list_bundles", return_value=bundles):
            submenu = labeler._build_model_submenu()

        labels = self._submenu_labels(submenu)
        assert "(no models found)" in labels

    # -- menu structure --

    def test_model_submenu_in_main_menu(self, tmp_path: Path) -> None:
        """Top-level menu contains a 'Model' item with a submenu."""
        labeler = self._make_labeler_with_models_dir(tmp_path)

        with patch("taskclf.model_registry.list_bundles", return_value=[]):
            menu = labeler._build_menu()

        submenu = self._find_model_submenu(menu)
        labels = self._submenu_labels(submenu)
        assert "(no models found)" in labels
        assert "Reload Model" in labels
        assert "Check Retrain" in labels


# ---------------------------------------------------------------------------
# Check Retrain  (Item 7)
# ---------------------------------------------------------------------------


class TestCheckRetrain:
    """Tests for TrayLabeler._check_retrain."""

    def test_no_models_dir_notifies(self, tmp_path: Path) -> None:
        """Without models_dir, notify 'No models directory configured'."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._models_dir is None

        with patch.object(labeler, "_notify") as mock_notify:
            labeler._check_retrain()

        mock_notify.assert_called_once()
        assert "no models directory" in mock_notify.call_args[0][0].lower()

    def test_retrain_due_no_models_found(self, tmp_path: Path) -> None:
        """When no models exist, notify 'Retrain recommended: no models found'."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=tmp_path / "models", event_bus=bus,
        )
        (tmp_path / "models").mkdir(exist_ok=True)

        with (
            patch("taskclf.train.retrain.find_latest_model", return_value=None),
            patch("taskclf.train.retrain.check_retrain_due", return_value=True),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._check_retrain()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "no models found" in msg.lower()

    def test_retrain_due_with_model(self, tmp_path: Path) -> None:
        """When retrain is due with an existing model, show model name and date."""
        import json

        bus, _ = _capture_bus()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "run_20260226"
        model_path.mkdir()
        (model_path / "metadata.json").write_text(
            json.dumps({"created_at": "2026-02-26T12:00:00+00:00"})
        )

        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=models_dir, event_bus=bus,
        )

        with (
            patch("taskclf.train.retrain.find_latest_model", return_value=model_path),
            patch("taskclf.train.retrain.check_retrain_due", return_value=True),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._check_retrain()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "retrain recommended" in msg.lower()
        assert "run_20260226" in msg
        assert "2026-02-26" in msg

    def test_model_is_current(self, tmp_path: Path) -> None:
        """When model is current, notify with model name and date."""
        import json

        bus, _ = _capture_bus()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "run_20260301"
        model_path.mkdir()
        (model_path / "metadata.json").write_text(
            json.dumps({"created_at": "2026-03-01T12:00:00+00:00"})
        )

        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=models_dir, event_bus=bus,
        )

        with (
            patch("taskclf.train.retrain.find_latest_model", return_value=model_path),
            patch("taskclf.train.retrain.check_retrain_due", return_value=False),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._check_retrain()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "current" in msg.lower()
        assert "run_20260301" in msg

    def test_with_retrain_config_file(self, tmp_path: Path) -> None:
        """When retrain_config is provided, load_retrain_config is used."""
        import json

        bus, _ = _capture_bus()
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        config_path = tmp_path / "retrain.yaml"
        config_path.write_text("global_retrain_cadence_days: 14\n")

        model_path = models_dir / "run_20260301"
        model_path.mkdir()
        (model_path / "metadata.json").write_text(
            json.dumps({"created_at": "2026-03-01T12:00:00+00:00"})
        )

        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=models_dir, event_bus=bus,
            retrain_config=config_path,
        )

        with (
            patch("taskclf.train.retrain.find_latest_model", return_value=model_path),
            patch("taskclf.train.retrain.check_retrain_due", return_value=False) as mock_check,
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._check_retrain()

        mock_check.assert_called_once_with(models_dir, 14)
        mock_notify.assert_called_once()

    def test_exception_notifies_check_failed(self, tmp_path: Path) -> None:
        """On exception, notify 'Check failed: ...'."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=tmp_path / "models", event_bus=bus,
        )
        (tmp_path / "models").mkdir(exist_ok=True)

        with (
            patch(
                "taskclf.train.retrain.find_latest_model",
                side_effect=RuntimeError("disk error"),
            ),
            patch.object(labeler, "_notify") as mock_notify,
        ):
            labeler._check_retrain()

        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][0]
        assert "check failed" in msg.lower()
        assert "disk error" in msg

    def test_check_retrain_in_submenu_enabled(self, tmp_path: Path) -> None:
        """Check Retrain is enabled when models_dir is set."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path, model_dir=None,
            models_dir=tmp_path / "models", event_bus=bus,
        )
        (tmp_path / "models").mkdir(exist_ok=True)

        menu = labeler._build_menu()
        for item in menu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Model":
                submenu = item.submenu
                break
        else:
            pytest.fail("Model submenu not found")

        retrain_item = None
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Check Retrain":
                retrain_item = item
                break

        assert retrain_item is not None
        assert retrain_item.enabled is True

    def test_check_retrain_in_submenu_disabled_without_models_dir(self, tmp_path: Path) -> None:
        """Check Retrain is disabled when models_dir is None."""
        bus, _ = _capture_bus()
        labeler = _make_tray_labeler(tmp_path, event_bus=bus)
        assert labeler._models_dir is None

        menu = labeler._build_menu()
        for item in menu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Model":
                submenu = item.submenu
                break
        else:
            pytest.fail("Model submenu not found")

        retrain_item = None
        for item in submenu.items:  # type: ignore[attr-defined]
            if hasattr(item, "text") and item.text == "Check Retrain":
                retrain_item = item
                break

        assert retrain_item is not None
        assert retrain_item.enabled is False
