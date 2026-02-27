"""System tray labeling app: persistent background labeler with activity transition detection.

Launch with::

    taskclf tray
    taskclf tray --model-dir models/run_20260226

Runs a pystray icon in the system tray that:

- Polls ActivityWatch for the current foreground app
- Detects activity transitions (dominant app changes persisting >= N minutes)
- Sends desktop notifications prompting the user to label completed blocks
- If a trained model is provided, suggests a label; otherwise shows all 8 core labels
- Allows quick labeling from the tray menu at any time
- Publishes prediction/suggestion events to a shared EventBus for the web UI
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)
from taskclf.core.types import CoreLabel, LabelSpan
from taskclf.labels.store import append_label_span
from taskclf.ui.events import EventBus

logger = logging.getLogger(__name__)

_ALL_LABELS = [cl.value for cl in CoreLabel]


# ---------------------------------------------------------------------------
# Activity transition detection
# ---------------------------------------------------------------------------


class ActivityMonitor:
    """Polls ActivityWatch and fires a callback when the dominant app changes.

    The transition rule: the dominant app must change and the new app must
    persist as dominant for >= *transition_minutes* consecutive polls before
    a transition is confirmed.

    Args:
        aw_host: ActivityWatch server base URL.
        title_salt: Salt for hashing window titles.
        poll_seconds: Seconds between polls.
        transition_minutes: Minutes a new app must persist before
            a transition is confirmed.
        on_transition: Callback invoked with
            ``(prev_app, new_app, block_start, block_end)`` when a
            transition is confirmed.
        on_poll: Callback invoked with ``(dominant_app,)`` after every
            successful poll (for status display updates).
        event_bus: Optional shared event bus for broadcasting events.
    """

    def __init__(
        self,
        *,
        aw_host: str = DEFAULT_AW_HOST,
        title_salt: str = DEFAULT_TITLE_SALT,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
        transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
        on_transition: Callable[[str, str, dt.datetime, dt.datetime], Any] | None = None,
        on_poll: Callable[[str], Any] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._aw_host = aw_host
        self._title_salt = title_salt
        self._poll_seconds = poll_seconds
        self._transition_threshold = transition_minutes * 60
        self._on_transition = on_transition
        self._on_poll = on_poll
        self._event_bus = event_bus

        self._current_app: str | None = None
        self._current_app_since: dt.datetime | None = None
        self._candidate_app: str | None = None
        self._candidate_duration: int = 0

        self._bucket_id: str | None = None
        self._aw_warned = False
        self._stop = threading.Event()

    def _discover_bucket(self) -> str:
        from taskclf.adapters.activitywatch.client import find_window_bucket_id

        return find_window_bucket_id(self._aw_host)

    def _poll_dominant_app(self) -> str | None:
        """Fetch recent AW events and return the most common app_id."""
        from taskclf.adapters.activitywatch.client import fetch_aw_events

        if self._bucket_id is None:
            try:
                self._bucket_id = self._discover_bucket()
                if self._aw_warned:
                    print(f"Connected to ActivityWatch at {self._aw_host}")
                    self._aw_warned = False
            except Exception:
                if not self._aw_warned:
                    print(
                        f"Waiting for ActivityWatch at {self._aw_host} "
                        f"(retrying every {self._poll_seconds}s)..."
                    )
                    self._aw_warned = True
                return None

        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        start = now - dt.timedelta(seconds=self._poll_seconds)
        try:
            events = fetch_aw_events(
                self._aw_host, self._bucket_id, start, now,
                title_salt=self._title_salt,
            )
        except Exception:
            logger.debug("Failed to fetch AW events", exc_info=True)
            return None

        if not events:
            return None

        counts = Counter(ev.app_id for ev in events)
        return counts.most_common(1)[0][0]

    def check_transition(self, dominant_app: str) -> None:
        """Update internal state and fire transition callback if warranted.

        Exposed as a public method so that transition logic can be unit-tested
        without requiring a live ActivityWatch server.
        """
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

        if self._current_app is None:
            self._current_app = dominant_app
            self._current_app_since = now
            return

        if dominant_app != self._current_app:
            if self._candidate_app == dominant_app:
                self._candidate_duration += self._poll_seconds
                if self._candidate_duration >= self._transition_threshold:
                    block_start = self._current_app_since or now
                    block_end = now - dt.timedelta(seconds=self._candidate_duration)
                    prev = self._current_app

                    self._current_app = dominant_app
                    self._current_app_since = block_end
                    self._candidate_app = None
                    self._candidate_duration = 0

                    if self._on_transition is not None:
                        self._on_transition(prev, dominant_app, block_start, block_end)
            else:
                self._candidate_app = dominant_app
                self._candidate_duration = self._poll_seconds
        else:
            self._candidate_app = None
            self._candidate_duration = 0

    def _publish_status(self, dominant_app: str) -> None:
        if self._event_bus is not None:
            self._event_bus.publish_threadsafe({
                "type": "status",
                "state": "collecting",
                "current_app": dominant_app,
            })

    def run(self) -> None:
        """Blocking poll loop. Call from a daemon thread."""
        while not self._stop.is_set():
            dominant = self._poll_dominant_app()
            if dominant is not None:
                if self._on_poll is not None:
                    self._on_poll(dominant)
                self._publish_status(dominant)
                self.check_transition(dominant)
            self._stop.wait(timeout=self._poll_seconds)

    def stop(self) -> None:
        """Signal the poll loop to stop."""
        self._stop.set()

    @property
    def current_app(self) -> str | None:
        return self._current_app


# ---------------------------------------------------------------------------
# Model predictor wrapper (optional)
# ---------------------------------------------------------------------------


class _LabelSuggester:
    """Wraps the online predictor for single-bucket label suggestions."""

    def __init__(self, model_dir: Path) -> None:
        from taskclf.core.model_io import load_model_bundle
        from taskclf.infer.online import OnlinePredictor

        model, metadata, cat_encoders = load_model_bundle(model_dir)
        self._predictor = OnlinePredictor(
            model, metadata, cat_encoders=cat_encoders,
        )
        self._aw_host: str = DEFAULT_AW_HOST
        self._title_salt: str = DEFAULT_TITLE_SALT

    def suggest(
        self, start: dt.datetime, end: dt.datetime,
    ) -> tuple[str, float] | None:
        """Predict a label for the given time window. Returns (label, confidence) or None."""
        from taskclf.adapters.activitywatch.client import fetch_aw_events, find_window_bucket_id
        from taskclf.features.build import build_features_from_aw_events

        try:
            bucket_id = find_window_bucket_id(self._aw_host)
            events = fetch_aw_events(
                self._aw_host, bucket_id, start, end,
                title_salt=self._title_salt,
            )
            if not events:
                return None

            rows = build_features_from_aw_events(events)
            if not rows:
                return None

            prediction = self._predictor.predict_bucket(rows[-1])
            return (prediction.core_label_name, prediction.confidence)
        except Exception:
            logger.warning("Could not generate label suggestion", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Tray icon and menu
# ---------------------------------------------------------------------------


def _make_icon_image(color: str = "#4CAF50", size: int = 64) -> Image.Image:
    """Generate a simple colored circle icon for the tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=color,
    )
    return img


class TrayLabeler:
    """System tray icon with labeling menus and notification support.

    Args:
        data_dir: Path to the processed data directory (for label storage).
        model_dir: Optional path to a model bundle for label suggestions.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        poll_seconds: Seconds between AW polls.
        transition_minutes: Minutes for transition detection threshold.
        event_bus: Optional shared event bus for broadcasting events.
    """

    def __init__(
        self,
        *,
        data_dir: Path = Path(DEFAULT_DATA_DIR),
        model_dir: Path | None = None,
        aw_host: str = DEFAULT_AW_HOST,
        title_salt: str = DEFAULT_TITLE_SALT,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
        transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
        event_bus: EventBus | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._labels_path = data_dir / "labels_v1" / "labels.parquet"
        self._current_app: str = "unknown"
        self._suggested_label: str | None = None
        self._suggested_confidence: float | None = None
        self._event_bus = event_bus

        self._suggester: _LabelSuggester | None = None
        if model_dir is not None:
            try:
                self._suggester = _LabelSuggester(model_dir)
                self._suggester._aw_host = aw_host
                self._suggester._title_salt = title_salt
                logger.info("Model loaded from %s", model_dir)
            except Exception:
                logger.warning("Could not load model from %s", model_dir, exc_info=True)

        self._monitor = ActivityMonitor(
            aw_host=aw_host,
            title_salt=title_salt,
            poll_seconds=poll_seconds,
            transition_minutes=transition_minutes,
            on_transition=self._handle_transition,
            on_poll=self._handle_poll,
            event_bus=event_bus,
        )
        self._icon: Any = None

    def _handle_poll(self, dominant_app: str) -> None:
        self._current_app = dominant_app
        if self._icon is not None:
            self._icon.update_menu()

    def _handle_transition(
        self,
        prev_app: str,
        new_app: str,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        suggestion = None
        if self._suggester is not None:
            suggestion = self._suggester.suggest(block_start, block_end)

        if suggestion is not None:
            self._suggested_label, self._suggested_confidence = suggestion
        else:
            self._suggested_label = None
            self._suggested_confidence = None

        if self._icon is not None:
            self._icon.update_menu()

        self._send_notification(prev_app, new_app, block_start, block_end)

        if self._event_bus is not None:
            if self._suggested_label is not None and self._suggested_confidence is not None:
                self._event_bus.publish_threadsafe({
                    "type": "suggest_label",
                    "reason": "app_switch",
                    "old_label": prev_app,
                    "suggested": self._suggested_label,
                    "confidence": self._suggested_confidence,
                    "block_start": block_start.isoformat(),
                    "block_end": block_end.isoformat(),
                })
            else:
                self._event_bus.publish_threadsafe({
                    "type": "prediction",
                    "label": "unknown",
                    "confidence": 0.0,
                    "ts": block_end.isoformat(),
                    "mapped_label": "unknown",
                    "current_app": new_app,
                })

    def _send_notification(
        self,
        prev_app: str,
        new_app: str,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        title = "taskclf — Activity changed"
        duration_min = max(1, int((block_end - block_start).total_seconds() / 60))
        message = f"{prev_app} → {new_app}\nLabel the last {duration_min} min?"
        if self._suggested_label is not None and self._suggested_confidence is not None:
            message += f"\nSuggested: {self._suggested_label} ({self._suggested_confidence:.0%})"

        try:
            from plyer import notification

            notification.notify(
                title=title,
                message=message,
                app_name="taskclf",
                timeout=10,
            )
        except Exception:
            logger.warning("Could not send notification", exc_info=True)

    def _label_action(self, label: str, minutes: int) -> Callable[..., None]:
        """Return a callback that creates a label span for the last N minutes."""
        def _do_label(*_args: Any) -> None:
            end_ts = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            start_ts = end_ts - dt.timedelta(minutes=minutes)

            span = LabelSpan(
                start_ts=start_ts,
                end_ts=end_ts,
                label=label,
                provenance="manual",
                user_id="default-user",
            )
            try:
                append_label_span(span, self._labels_path)
                logger.info("Labeled: %s [%s → %s]", label, start_ts, end_ts)
                try:
                    from plyer import notification

                    notification.notify(
                        title="taskclf — Label saved",
                        message=f"{label} [{start_ts:%H:%M} → {end_ts:%H:%M} UTC]",
                        app_name="taskclf",
                        timeout=5,
                    )
                except Exception:
                    pass
            except ValueError as exc:
                logger.error("Label failed: %s", exc)

        return _do_label

    def _build_menu(self) -> "pystray.Menu":
        import pystray

        items: list[pystray.MenuItem] = []

        items.append(pystray.MenuItem(
            f"Current: {self._current_app}",
            action=None,
            enabled=False,
        ))

        if self._suggested_label is not None and self._suggested_confidence is not None:
            items.append(pystray.MenuItem(
                f"Suggested: {self._suggested_label} ({self._suggested_confidence:.0%})",
                action=None,
                enabled=False,
            ))

        items.append(pystray.Menu.SEPARATOR)

        for minutes in (5, 10, 15, 30):
            label_items = []
            for lbl in _ALL_LABELS:
                label_items.append(pystray.MenuItem(
                    lbl,
                    self._label_action(lbl, minutes),
                ))
            items.append(pystray.MenuItem(
                f"Label Last {minutes} min",
                pystray.Menu(*label_items),
            ))

        items.append(pystray.Menu.SEPARATOR)

        items.append(pystray.MenuItem(
            "Show/Hide Window",
            self._toggle_window,
        ))
        items.append(pystray.MenuItem("Quit", self._quit))

        return pystray.Menu(*items)

    def _toggle_window(self, *_args: Any) -> None:
        import urllib.request

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8741/api/window/toggle",
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            self._notify("UI server not running. Start with: taskclf ui")

    def _quit(self, *_args: Any) -> None:
        self._monitor.stop()
        if self._icon is not None:
            self._icon.stop()

    def _notify(self, message: str) -> None:
        try:
            from plyer import notification

            notification.notify(
                title="taskclf",
                message=message,
                app_name="taskclf",
                timeout=5,
            )
        except Exception:
            logger.warning("Could not send notification", exc_info=True)

    def run(self) -> None:
        """Start the tray icon and background monitor. Blocks until quit."""
        import pystray

        monitor_thread = threading.Thread(
            target=self._monitor.run, daemon=True,
        )
        monitor_thread.start()

        icon_image = _make_icon_image()
        self._icon = pystray.Icon(
            "taskclf",
            icon_image,
            "taskclf",
            menu=self._build_menu(),
        )

        mode = "with model suggestions" if self._suggester else "label-only (no model)"
        print(f"taskclf tray started ({mode})")
        print("Right-click the tray icon to label. Press Quit to exit.")

        self._icon.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_tray(
    *,
    model_dir: Path | None = None,
    aw_host: str = DEFAULT_AW_HOST,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    title_salt: str = DEFAULT_TITLE_SALT,
    data_dir: Path = Path(DEFAULT_DATA_DIR),
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
    event_bus: EventBus | None = None,
) -> None:
    """Launch the system tray labeling app.

    Args:
        model_dir: Optional path to a trained model bundle.  When
            provided, the tray suggests labels on activity transitions.
            When omitted, all 8 core labels are presented equally.
        aw_host: ActivityWatch server URL.
        poll_seconds: Seconds between AW polling cycles.
        title_salt: Salt for hashing window titles.
        data_dir: Processed data directory (labels stored here).
        transition_minutes: Minutes a new dominant app must persist
            before a transition notification fires.
        event_bus: Optional shared event bus for broadcasting events
            to connected WebSocket clients.
    """
    tray = TrayLabeler(
        data_dir=data_dir,
        model_dir=model_dir,
        aw_host=aw_host,
        title_salt=title_salt,
        poll_seconds=poll_seconds,
        transition_minutes=transition_minutes,
        event_bus=event_bus,
    )
    tray.run()
