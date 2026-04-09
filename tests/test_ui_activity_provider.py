"""Tests for provider-neutral activity-source helpers."""

from __future__ import annotations

import datetime as dt

from taskclf.adapters.activitywatch.client import AWConnectionError
from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.ui.activity_provider import (
    ActivityProviderUnavailableError,
    ActivityWatchProvider,
)
from taskclf.ui.runtime import ActivityMonitor


def _aw_event(app_id: str) -> AWEvent:
    return AWEvent(
        timestamp=dt.datetime(2026, 4, 9, 8, 0, tzinfo=dt.timezone.utc),
        duration_seconds=60.0,
        app_id=app_id,
        window_title_hash="hashed",
        is_browser=False,
        is_editor=False,
        is_terminal=False,
        app_category="other",
    )


class TestActivityWatchProvider:
    def test_probe_status_ready(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "taskclf.ui.activity_provider.find_window_bucket_id",
            lambda host, timeout=10: "aw-window-test",
        )

        provider = ActivityWatchProvider(
            endpoint="http://localhost:5600",
            title_salt="salt",
        )
        status = provider.probe_status(timeout_seconds=2)

        assert status.state == "ready"
        assert status.summary_available is True
        assert status.source_id == "aw-window-test"

    def test_probe_status_setup_required_includes_setup_guidance(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            "taskclf.ui.activity_provider.find_window_bucket_id",
            lambda host, timeout=10: (_ for _ in ()).throw(AWConnectionError(host)),
        )

        provider = ActivityWatchProvider(
            endpoint="http://localhost:5600",
            title_salt="salt",
        )
        status = provider.probe_status(timeout_seconds=2)

        assert status.state == "setup_required"
        assert status.summary_available is False
        assert status.setup_title == "Activity source unavailable"
        assert "Manual labeling still works" in status.setup_message
        assert status.setup_steps[1] == (
            "Confirm the local server is reachable at http://localhost:5600."
        )
        assert status.help_url == "https://activitywatch.net/"

    def test_recent_app_summary_uses_provider_breakdown(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "taskclf.ui.activity_provider.find_window_bucket_id",
            lambda host, timeout=10: "aw-window-test",
        )
        monkeypatch.setattr(
            "taskclf.ui.activity_provider.fetch_aw_events",
            lambda *args, **kwargs: [
                _aw_event("com.apple.Terminal"),
                _aw_event("com.apple.Terminal"),
                _aw_event("com.apple.Safari"),
            ],
        )

        provider = ActivityWatchProvider(
            endpoint="http://localhost:5600",
            title_salt="salt",
        )
        status, apps = provider.recent_app_summary(
            dt.datetime(2026, 4, 9, 8, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 4, 9, 8, 5, tzinfo=dt.timezone.utc),
        )

        assert status.state == "ready"
        assert status.last_sample_count == 3
        assert status.last_sample_breakdown == {
            "com.apple.Terminal": 2,
            "com.apple.Safari": 1,
        }
        assert [(entry.app, entry.events) for entry in apps] == [
            ("com.apple.Terminal", 2),
            ("com.apple.Safari", 1),
        ]


class TestActivityMonitorProviderState:
    def test_monitor_starts_in_checking_state(self) -> None:
        monitor = ActivityMonitor()
        assert monitor.activity_provider_status["state"] == "checking"

    def test_first_probe_uses_short_timeout_and_marks_setup_required(
        self,
        monkeypatch,
    ) -> None:
        captured: dict[str, float | int | None] = {}

        def _fail_discover(
            self,
            *,
            timeout_seconds: float | int | None = None,
        ) -> str:
            captured["timeout_seconds"] = timeout_seconds
            raise ActivityProviderUnavailableError("not ready", retryable=True)

        monkeypatch.setattr(
            ActivityWatchProvider,
            "discover_source_id",
            _fail_discover,
        )

        monitor = ActivityMonitor(aw_timeout_seconds=10, poll_seconds=60)
        dominant = monitor._poll_dominant_app()

        assert dominant is None
        assert captured["timeout_seconds"] == 2
        assert monitor.activity_provider_status["state"] == "setup_required"
        assert monitor.activity_provider_status["summary_available"] is False
