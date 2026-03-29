"""Tests for the gap-fill surface (Phase 4b).

Covers:
- SEM-003: Gap-fill interval spans from last label end to current time.
- SRF-005: Badge text includes unlabeled time duration.
- SRF-006: Prompt fires only at idle return, session start, or post-acceptance.
- SRF-007: Escalation changes tray icon state when threshold exceeded.
- P4b-001: Idle return (>5 min) triggers gap-fill prompt event.
- P4b-002: New session with unlabeled time triggers prompt.
- P4b-003: After accepting transition suggestion with adjacent gap, prompt fires.
- P4b-004: Escalation does not call the notification API (no popup).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

from taskclf.core.types import LabelSpan
from taskclf.labels.store import write_label_spans
from taskclf.ui.events import EventBus
from taskclf.ui.tray import TrayLabeler


def _capture_bus() -> tuple[EventBus, list[dict]]:
    bus = EventBus()
    captured: list[dict] = []
    bus.publish_threadsafe = lambda event: captured.append(event)  # type: ignore[assignment]
    return bus, captured


def _write_label(tmp_path: Path, end_ts: dt.datetime) -> Path:
    """Write a single label ending at *end_ts* and return the parquet path."""
    labels_dir = tmp_path / "labels_v1"
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / "labels.parquet"
    span = LabelSpan(
        start_ts=end_ts - dt.timedelta(minutes=30),
        end_ts=end_ts,
        label="Build",
        provenance="manual",
        user_id="test-user",
        confidence=1.0,
    )
    write_label_spans([span], path)
    return path


def _make_labeler(
    tmp_path: Path,
    bus: EventBus,
    **kwargs: object,
) -> TrayLabeler:
    return TrayLabeler(
        data_dir=tmp_path,
        model_dir=None,
        event_bus=bus,
        **kwargs,  # type: ignore[arg-type]
    )


class TestUnlabeledTimeIndicator:
    """Step 4b.1: passive unlabeled-time indicator."""

    def test_sem003_gap_fill_interval_spans_last_label_to_now(
        self, tmp_path: Path
    ) -> None:
        """SEM-003: Unlabeled interval spans from last label end to current time."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        fake_now = dt.datetime(2026, 3, 28, 11, 30, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        unlabeled = [e for e in captured if e["type"] == "unlabeled_time"]
        assert len(unlabeled) == 1
        assert unlabeled[0]["last_label_end"] == label_end.isoformat()
        assert unlabeled[0]["unlabeled_minutes"] == 150.0

    def test_srf005_badge_text_includes_duration(self, tmp_path: Path) -> None:
        """SRF-005: Badge text includes unlabeled time duration."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        fake_now = dt.datetime(2026, 3, 28, 11, 30, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        unlabeled = [e for e in captured if e["type"] == "unlabeled_time"]
        assert len(unlabeled) == 1
        assert "2h 30m" in unlabeled[0]["text"]

    def test_no_unlabeled_event_when_no_labels_exist(self, tmp_path: Path) -> None:
        bus, captured = _capture_bus()
        labeler = _make_labeler(tmp_path, bus)
        labeler._publish_unlabeled_time()

        unlabeled = [e for e in captured if e["type"] == "unlabeled_time"]
        assert len(unlabeled) == 0

    def test_cache_invalidated_on_label_save(self, tmp_path: Path) -> None:
        """Cache updates when _labels_saved_count changes."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        labeler._get_last_label_end()
        assert labeler._last_label_end_cache == label_end

        new_end = dt.datetime(2026, 3, 28, 11, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, new_end)
        labeler._labels_saved_count += 1
        result = labeler._get_last_label_end()
        assert result == new_end


class TestGapFillPrompting:
    """Step 4b.2: contextual gap-fill prompting."""

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_p4b001_idle_return_triggers_gap_fill(
        self, _mock_notif: object, tmp_path: Path
    ) -> None:
        """P4b-001: Idle return >5 min triggers gap_fill_prompt event."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        block_start = dt.datetime(2026, 3, 28, 10, 0, 0, tzinfo=dt.timezone.utc)
        block_end = dt.datetime(2026, 3, 28, 10, 10, 0, tzinfo=dt.timezone.utc)
        fake_now = dt.datetime(2026, 3, 28, 10, 15, 0, tzinfo=dt.timezone.utc)

        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.datetime.fromisoformat = dt.datetime.fromisoformat
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._handle_transition(
                "com.apple.loginwindow", "com.apple.Terminal", block_start, block_end
            )

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 1
        assert prompts[0]["trigger"] == "idle_return"

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_idle_return_under_5min_no_prompt(
        self, _mock_notif: object, tmp_path: Path
    ) -> None:
        """Short idle (<= 5 min) does not trigger gap_fill_prompt."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        block_start = dt.datetime(2026, 3, 28, 10, 0, 0, tzinfo=dt.timezone.utc)
        block_end = dt.datetime(2026, 3, 28, 10, 3, 0, tzinfo=dt.timezone.utc)
        fake_now = dt.datetime(2026, 3, 28, 10, 5, 0, tzinfo=dt.timezone.utc)

        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.datetime.fromisoformat = dt.datetime.fromisoformat
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._handle_transition(
                "com.apple.loginwindow", "com.apple.Terminal", block_start, block_end
            )

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 0

    def test_p4b002_session_start_triggers_prompt(self, tmp_path: Path) -> None:
        """P4b-002: New session with existing unlabeled time triggers prompt."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        ts = dt.datetime(2026, 3, 28, 11, 0, 0, tzinfo=dt.timezone.utc)
        fake_now = dt.datetime(2026, 3, 28, 11, 0, 0, tzinfo=dt.timezone.utc)

        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._handle_initial_app("com.apple.Terminal", ts)

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 1
        assert prompts[0]["trigger"] == "session_start"

    def test_session_start_no_prompt_without_labels(self, tmp_path: Path) -> None:
        """No prompt at session start when no labels exist."""
        bus, captured = _capture_bus()
        labeler = _make_labeler(tmp_path, bus)

        ts = dt.datetime(2026, 3, 28, 11, 0, 0, tzinfo=dt.timezone.utc)
        labeler._handle_initial_app("com.apple.Terminal", ts)

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 0

    def test_p4b003_post_acceptance_triggers_prompt(self, tmp_path: Path) -> None:
        """P4b-003: After accepting transition suggestion with adjacent gap, prompt fires."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        fake_now = dt.datetime(2026, 3, 28, 11, 0, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._on_suggestion_accepted()

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 1
        assert prompts[0]["trigger"] == "post_acceptance"
        assert prompts[0]["unlabeled_minutes"] == 120.0

    def test_srf006_prompt_only_at_defined_triggers(self, tmp_path: Path) -> None:
        """SRF-006: gap_fill_prompt fires only at idle return, session start, or post-acceptance."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 9, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus)

        labeler._handle_poll("com.apple.Terminal")
        labeler._handle_poll("com.apple.Terminal")
        labeler._handle_poll("com.apple.Terminal")

        prompts = [e for e in captured if e["type"] == "gap_fill_prompt"]
        assert len(prompts) == 0, "Regular polls must not trigger gap_fill_prompt"


class TestEscalation:
    """Step 4b.3: escalation threshold."""

    def test_srf007_escalation_changes_icon_state(self, tmp_path: Path) -> None:
        """SRF-007: Escalation changes tray icon state when threshold exceeded."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 1, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus, gap_fill_escalation_minutes=60)

        from unittest.mock import MagicMock

        mock_icon = MagicMock()
        labeler._icon = mock_icon

        fake_now = dt.datetime(2026, 3, 28, 3, 0, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        assert labeler._escalated is True
        assert mock_icon.icon is not None

        escalated = [e for e in captured if e["type"] == "gap_fill_escalated"]
        assert len(escalated) == 1
        assert escalated[0]["unlabeled_minutes"] == 120.0
        assert escalated[0]["threshold_minutes"] == 60

    def test_escalation_reverts_when_below_threshold(self, tmp_path: Path) -> None:
        """Icon reverts when unlabeled time drops below threshold."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 1, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus, gap_fill_escalation_minutes=60)
        labeler._escalated = True

        from unittest.mock import MagicMock

        mock_icon = MagicMock()
        labeler._icon = mock_icon

        new_end = dt.datetime(2026, 3, 28, 2, 50, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, new_end)
        labeler._labels_saved_count += 1

        fake_now = dt.datetime(2026, 3, 28, 3, 0, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        assert labeler._escalated is False

    @patch("taskclf.ui.tray._send_desktop_notification")
    def test_p4b004_escalation_no_popup(
        self, mock_notif: object, tmp_path: Path
    ) -> None:
        """P4b-004: Escalation does not call the notification API (no popup)."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 1, 0, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus, gap_fill_escalation_minutes=60)

        fake_now = dt.datetime(2026, 3, 28, 3, 0, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        assert labeler._escalated is True
        from unittest.mock import MagicMock

        assert isinstance(mock_notif, MagicMock)
        mock_notif.assert_not_called()  # type: ignore[union-attr]

    def test_no_escalation_below_threshold(self, tmp_path: Path) -> None:
        """No escalation when unlabeled time is below threshold."""
        bus, captured = _capture_bus()
        label_end = dt.datetime(2026, 3, 28, 2, 30, 0, tzinfo=dt.timezone.utc)
        _write_label(tmp_path, label_end)
        labeler = _make_labeler(tmp_path, bus, gap_fill_escalation_minutes=480)

        fake_now = dt.datetime(2026, 3, 28, 3, 0, 0, tzinfo=dt.timezone.utc)
        with patch("taskclf.ui.tray.dt") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = dt.timezone
            mock_dt.timedelta = dt.timedelta
            labeler._publish_unlabeled_time()

        assert labeler._escalated is False
        escalated = [e for e in captured if e["type"] == "gap_fill_escalated"]
        assert len(escalated) == 0


class TestFormatDuration:
    def test_hours_and_minutes(self) -> None:
        assert TrayLabeler._format_duration(150) == "2h 30m"

    def test_hours_only(self) -> None:
        assert TrayLabeler._format_duration(120) == "2h"

    def test_minutes_only(self) -> None:
        assert TrayLabeler._format_duration(45) == "45m"

    def test_zero(self) -> None:
        assert TrayLabeler._format_duration(0) == "0m"
