"""Tests for the system tray labeling app.

Covers:
- ActivityMonitor transition detection logic
- Label span creation via tray callbacks
- No tests for pystray GUI (interactive widget, untestable in CI)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.labels.store import append_label_span, read_label_spans
from taskclf.ui.tray import ActivityMonitor


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
