"""Tests for surface separation between transition suggestions and live status.

Covers:
- SRF-001: Transition notification text matches "Was this {label}? {start}–{end}".
- SRF-002: Live status text matches "Now: {label}".
- SRF-003: _LabelSuggester and live-status are separate methods/code paths.
- SRF-004: Notification payload does not contain numeric confidence.
- SRF-008: All user-facing strings are imported from ui/copy.py.
- CNF-004: Settings schema does not expose reject_threshold.
- SEM-002: Live status uses only the latest bucket.
"""

from __future__ import annotations

import datetime as dt
import inspect
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from taskclf.ui.copy import (
    gap_fill_detail,
    gap_fill_prompt,
    live_status_text,
    transition_suggestion_text,
)
from taskclf.ui.events import EventBus
from taskclf.ui.tray import TrayLabeler


def _capture_bus() -> tuple[EventBus, list[dict]]:
    bus = EventBus()
    captured: list[dict] = []
    bus.publish_threadsafe = lambda event: captured.append(event)  # type: ignore[assignment]
    return bus, captured


_BLOCK_START = dt.datetime(2026, 3, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
_BLOCK_END = dt.datetime(2026, 3, 1, 10, 15, 0, tzinfo=dt.timezone.utc)


def _transition_text_for_interval(
    label: str,
    block_start: dt.datetime,
    block_end: dt.datetime,
) -> str:
    return transition_suggestion_text(
        label,
        block_start.astimezone().strftime("%H:%M"),
        block_end.astimezone().strftime("%H:%M"),
    )


@contextmanager
def _local_timezone(name: str):
    if not hasattr(time, "tzset"):
        pytest.skip("Local timezone override requires time.tzset()")

    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


class TestCopyStrings:
    """SRF-001, SRF-002, SRF-008: copy string format and centralization."""

    def test_srf001_transition_suggestion_text(self) -> None:
        """SRF-001: Transition text matches 'Was this {label}? {start}–{end}'."""
        text = transition_suggestion_text("Coding", "12:00", "12:47")
        assert text == "Was this Coding? 12:00\u201312:47"

    def test_srf002_live_status_text(self) -> None:
        """SRF-002: Live status text matches 'Now: {label}'."""
        assert live_status_text("Coding") == "Now: Coding"

    def test_gap_fill_prompt(self) -> None:
        assert gap_fill_prompt("2h 30m") == "You have 2h 30m unlabeled. Review?"

    def test_gap_fill_detail(self) -> None:
        assert gap_fill_detail("9:00", "11:30") == "Review unlabeled: 9:00\u201311:30"

    def test_srf008_copy_module_is_sole_source(self) -> None:
        """SRF-008: All four user-facing copy functions live in ui.copy."""
        import taskclf.ui.copy as copy_mod

        public = [
            name
            for name, obj in inspect.getmembers(copy_mod, inspect.isfunction)
            if not name.startswith("_")
        ]
        assert set(public) == {
            "transition_suggestion_text",
            "live_status_text",
            "gap_fill_prompt",
            "gap_fill_detail",
        }


class TestTransitionSurface:
    """SRF-004: No confidence in transition notification."""

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_srf004_notification_excludes_confidence(
        self,
        mock_notif: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SRF-004: Desktop notification body must not contain numeric confidence."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
            privacy_notifications=False,
        )
        labeler._suggested_label = "Build"
        labeler._suggested_confidence = 0.92

        labeler._send_notification(
            "com.apple.Terminal",
            "us.zoom.xos",
            _BLOCK_START,
            _BLOCK_END,
        )

        mock_notif.assert_called_once()
        message = mock_notif.call_args[0][1]
        assert "92%" not in message
        assert "0.92" not in message
        assert "Build" in message
        assert "Was this Build?" in message

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_transition_notification_uses_local_display_time(
        self,
        mock_notif: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Transition notification copy uses local clock time, not raw UTC."""
        bus, _ = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
            privacy_notifications=False,
        )
        labeler._suggested_label = "Build"

        with _local_timezone("America/Los_Angeles"):
            labeler._send_notification(
                "com.apple.Terminal",
                "us.zoom.xos",
                _BLOCK_START,
                _BLOCK_END,
            )

        mock_notif.assert_called_once()
        message = mock_notif.call_args[0][1]
        assert message == "Was this Build? 02:00\u201302:15"

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_srf004_prompt_label_event_excludes_confidence(
        self,
        _mock_notif: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SRF-004: prompt_label event must not contain suggested_confidence."""
        bus, captured = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
        )

        mock_suggester = MagicMock()
        mock_suggester.suggest.return_value = ("Build", 0.85)
        labeler._suggester = mock_suggester

        labeler._handle_transition(
            "com.apple.Terminal",
            "us.zoom.xos",
            _BLOCK_START,
            _BLOCK_END,
        )

        prompt = next(e for e in captured if e["type"] == "prompt_label")
        assert "suggested_confidence" not in prompt
        assert prompt["suggestion_text"] == _transition_text_for_interval(
            "Build",
            _BLOCK_START,
            _BLOCK_END,
        )

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_prompt_label_event_uses_local_display_time(
        self,
        _mock_notif: MagicMock,
        tmp_path: Path,
    ) -> None:
        """prompt_label copy uses local time while structured bounds stay UTC."""
        bus, captured = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
        )

        mock_suggester = MagicMock()
        mock_suggester.suggest.return_value = ("Build", 0.85)
        labeler._suggester = mock_suggester

        with _local_timezone("America/Los_Angeles"):
            labeler._handle_transition(
                "com.apple.Terminal",
                "us.zoom.xos",
                _BLOCK_START,
                _BLOCK_END,
            )

        prompt = next(e for e in captured if e["type"] == "prompt_label")
        assert prompt["block_start"] == _BLOCK_START.isoformat()
        assert prompt["block_end"] == _BLOCK_END.isoformat()
        assert prompt["suggestion_text"] == "Was this Build? 02:00\u201302:15"


class TestSurfaceSeparation:
    """SRF-003: Transition suggestions and live status use separate code paths."""

    def test_srf003_separate_methods(self) -> None:
        """SRF-003: _handle_transition and _publish_live_status are distinct methods."""
        assert hasattr(TrayLabeler, "_handle_transition")
        assert hasattr(TrayLabeler, "_publish_live_status")
        assert TrayLabeler._handle_transition is not TrayLabeler._publish_live_status

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_sem002_live_status_uses_latest_bucket_only(
        self,
        _mock_notif: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SEM-002: Live status publishes a prediction for only the current bucket."""
        bus, captured = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
        )

        mock_suggester = MagicMock()
        mock_suggester.suggest.return_value = ("Debug", 0.75)
        labeler._suggester = mock_suggester

        labeler._publish_live_status()

        live_events = [e for e in captured if e["type"] == "live_status"]
        assert len(live_events) == 1
        assert live_events[0]["label"] == "Debug"
        assert live_events[0]["text"] == "Now: Debug"

        call_args = mock_suggester.suggest.call_args[0]
        start, end = call_args
        assert (end - start).total_seconds() <= 60

    def test_live_status_skipped_without_model(self, tmp_path: Path) -> None:
        bus, captured = _capture_bus()
        labeler = TrayLabeler(
            data_dir=tmp_path,
            model_dir=None,
            event_bus=bus,
        )
        labeler._publish_live_status()
        live_events = [e for e in captured if e["type"] == "live_status"]
        assert len(live_events) == 0


class TestSettingsSchema:
    """CNF-004: reject_threshold is not user-configurable."""

    def test_cnf004_reject_threshold_not_in_settings(self) -> None:
        """CNF-004: Settings schema does not expose reject_threshold."""
        from taskclf.core.config import UserConfig

        config = (
            UserConfig.__dataclass_fields__
            if hasattr(UserConfig, "__dataclass_fields__")
            else {}
        )
        pydantic_fields = getattr(UserConfig, "model_fields", {})
        all_fields = set(config.keys()) | set(pydantic_fields.keys())
        assert "reject_threshold" not in all_fields
