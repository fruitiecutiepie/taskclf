"""System tray labeling app: persistent background labeler with activity transition detection.

Launch with::

    taskclf tray
    taskclf tray --model-dir models/run_20260226

Runs a pystray icon in the system tray that:

- Polls ActivityWatch for the current foreground app
- Detects activity transitions (dominant app changes persisting >= N minutes)
- Sends desktop notifications prompting the user to label completed blocks
- Left-click opens the web dashboard; labeling is done through the web UI
- Publishes prediction/suggestion events to a shared EventBus for the web UI
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray

from taskclf.core.config import UserConfig
from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_AW_TIMEOUT_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_IDLE_TRANSITION_MINUTES,
    DEFAULT_INFERENCE_POLICY_FILE,
    DEFAULT_POLL_SECONDS,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)
from taskclf.ui.events import EventBus
from taskclf.ui.runtime import (
    ActivityMonitor,
    _LabelSuggester,
    _LOCKSCREEN_APP_IDS,
)

logger = logging.getLogger(__name__)

_VITE_DEV_PORT = 5173


def _send_desktop_notification(title: str, message: str, timeout: int = 10) -> None:
    """Best-effort passive desktop notification (secondary fallback).

    The primary notification mechanism is the Web Notifications API
    in the frontend (cross-platform, works on mobile).  This function
    serves as a backup for when no browser client is connected.

    The osascript call runs in a daemon thread so a slow or hung
    macOS notification subsystem never blocks the caller.
    """
    if platform.system() == "Darwin":
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        safe_message = (
            message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        )
        script = f'display notification "{safe_message}" with title "{safe_title}"'

        def _fire() -> None:
            try:
                proc = subprocess.Popen(
                    ["osascript", "-e", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    logger.debug("osascript notification timed out after %ds", timeout)
            except Exception:
                logger.debug("osascript notification failed", exc_info=True)

        threading.Thread(target=_fire, daemon=True).start()
        return

    logger.info("[%s] %s", title, message)


def _display_clock_time_local(ts: dt.datetime) -> str:
    """Format a transition-boundary timestamp in the user's local timezone."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone().strftime("%H:%M")


def _display_time_range_exact_local(
    start: dt.datetime,
    end: dt.datetime,
) -> str:
    """Format an exact local transition range, adding dates if it crosses midnight."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)

    start_local = start.astimezone()
    end_local = end.astimezone()
    if start_local.date() == end_local.date():
        return (
            f"{start_local.strftime('%H:%M:%S')}\u2013{end_local.strftime('%H:%M:%S')}"
        )
    return (
        f"{start_local.strftime('%b %d %H:%M:%S')}"
        f"\u2013{end_local.strftime('%b %d %H:%M:%S')}"
    )


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


@dataclass(kw_only=True, eq=False)
class TrayLabeler:
    """System tray icon with labeling menus and notification support.

    Args:
        data_dir: Path to the processed data directory (for label storage).
        model_dir: Optional path to a model bundle for label suggestions.
        models_dir: Optional path to the directory containing all model
            bundles.  When set, the tray builds a dynamic "Prediction Model"
            submenu listing available bundles for hot-swapping.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        poll_seconds: Seconds between AW polls.
        transition_minutes: Minutes for transition detection threshold.
        event_bus: Optional shared event bus for broadcasting events.
        ui_port: Port for the embedded UI server.
        open_browser: When ``True``, browser-mode startup opens the UI in
            the default browser immediately.  When ``False``, the server
            starts without launching a browser tab.
    """

    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR))
    model_dir: Path | None = None
    models_dir: Path | None = None
    aw_host: str = DEFAULT_AW_HOST
    title_salt: str = DEFAULT_TITLE_SALT
    poll_seconds: int = DEFAULT_POLL_SECONDS
    aw_timeout_seconds: int = DEFAULT_AW_TIMEOUT_SECONDS
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES
    event_bus: EventBus | None = None
    ui_port: int = 8741
    dev: bool = False
    browser: bool = False
    no_tray: bool = False
    open_browser: bool = True
    username: str | None = None
    notifications_enabled: bool = True
    privacy_notifications: bool = True
    retrain_config: Path | None = None
    gap_fill_escalation_minutes: int = 480
    _data_dir: Path = field(init=False)
    _model_dir: Path | None = field(init=False, default=None)
    _models_dir: Path | None = field(init=False, default=None)
    _retrain_config: Path | None = field(init=False, default=None)
    _labels_path: Path = field(init=False)
    _config: UserConfig = field(init=False)
    _notifications_enabled: bool = field(init=False, default=True)
    _privacy_notifications: bool = field(init=False, default=True)
    _current_app: str = field(init=False, default="unknown")
    _suggested_label: str | None = field(init=False, default=None)
    _suggested_confidence: float | None = field(init=False, default=None)
    _ui_port: int = field(init=False, default=8741)
    _ui_server_running: bool = field(init=False, default=False)
    _ui_proc: Any = field(init=False, default=None)
    _vite_proc: Any = field(init=False, default=None)
    _aw_host: str = field(init=False, default=DEFAULT_AW_HOST)
    _title_salt: str = field(init=False, default=DEFAULT_TITLE_SALT)
    _dev: bool = field(init=False, default=False)
    _browser: bool = field(init=False, default=False)
    _no_tray: bool = field(init=False, default=False)
    _open_browser: bool = field(init=False, default=True)
    _transition_count: int = field(init=False, default=0)
    _last_transition: dict[str, Any] | None = field(init=False, default=None)
    _labels_saved_count: int = field(init=False, default=0)
    _model_schema_hash: str | None = field(init=False, default=None)
    _event_bus: EventBus = field(init=False)
    _suggester: _LabelSuggester | None = field(init=False, default=None)
    _initial_model_load_started: bool = field(init=False, default=False)
    _monitor: ActivityMonitor = field(init=False)
    _icon: Any = field(init=False, default=None)
    _unlabeled_minutes: float = field(init=False, default=0.0)
    _last_label_end_cache: dt.datetime | None = field(init=False, default=None)
    _last_label_cache_count: int = field(init=False, default=-1)
    _escalated: bool = field(init=False, default=False)
    _gap_fill_escalation_minutes: int = field(init=False, default=480)

    def __post_init__(self) -> None:
        self._data_dir = self.data_dir
        self._model_dir = self.model_dir
        self._models_dir = self.models_dir
        self._retrain_config = self.retrain_config
        self._labels_path = self.data_dir / "labels_v1" / "labels.parquet"
        self._config = UserConfig(self.data_dir)
        if self.username is not None:
            self._config.username = self.username

        saved = self._config.as_dict()
        notifications_enabled = self._resolve(
            saved,
            "notifications_enabled",
            self.notifications_enabled,
            True,
        )
        privacy_notifications = self._resolve(
            saved,
            "privacy_notifications",
            self.privacy_notifications,
            True,
        )
        poll_seconds = self._resolve(
            saved,
            "poll_seconds",
            self.poll_seconds,
            DEFAULT_POLL_SECONDS,
        )
        aw_timeout_seconds = self._resolve(
            saved,
            "aw_timeout_seconds",
            self.aw_timeout_seconds,
            DEFAULT_AW_TIMEOUT_SECONDS,
        )
        transition_minutes = self._resolve(
            saved,
            "transition_minutes",
            self.transition_minutes,
            DEFAULT_TRANSITION_MINUTES,
        )
        aw_host = self._resolve(saved, "aw_host", self.aw_host, DEFAULT_AW_HOST)
        title_salt = self._resolve(
            saved, "title_salt", self.title_salt, DEFAULT_TITLE_SALT
        )
        ui_port = self._resolve(saved, "ui_port", self.ui_port, 8741)

        # Resolved values are applied at runtime only; do not rewrite config.toml on
        # every startup (starter template is written once when the file is missing).

        self._notifications_enabled = notifications_enabled
        self._privacy_notifications = privacy_notifications
        self._current_app: str = "unknown"
        self._suggested_label: str | None = None
        self._suggested_confidence: float | None = None
        self._ui_port = ui_port
        self._ui_server_running = False
        self._ui_proc: Any = None
        self._vite_proc: Any = None
        self._aw_host = aw_host
        self._title_salt = title_salt
        self._dev = self.dev
        self._browser = self.browser
        self._no_tray = self.no_tray
        self._open_browser = self.open_browser

        # Electron-spawned sidecar: never use pystray (can return immediately
        # without a GUI context). CLI users can still use --browser --no-open-browser
        # with a tray icon unless TASKCLF_ELECTRON_SHELL=1 is set by Electron.
        if (
            os.environ.get("TASKCLF_ELECTRON_SHELL") == "1"
            and self._browser
            and not self._open_browser
        ):
            self._no_tray = True

        self._transition_count: int = 0
        self._last_transition: dict[str, Any] | None = None
        self._labels_saved_count: int = 0
        self._model_schema_hash: str | None = None

        self._gap_fill_escalation_minutes = self._resolve(
            saved,
            "gap_fill_escalation_minutes",
            self.gap_fill_escalation_minutes,
            480,
        )

        self._event_bus = self.event_bus if self.event_bus is not None else EventBus()

        self._suggester: _LabelSuggester | None = None
        self._initial_model_load_started = False

        idle_transition_minutes = self._resolve(
            saved,
            "idle_transition_minutes",
            DEFAULT_IDLE_TRANSITION_MINUTES,
            DEFAULT_IDLE_TRANSITION_MINUTES,
        )

        self._monitor = ActivityMonitor(
            aw_host=aw_host,
            title_salt=title_salt,
            poll_seconds=poll_seconds,
            aw_timeout_seconds=aw_timeout_seconds,
            transition_minutes=transition_minutes,
            idle_transition_minutes=idle_transition_minutes,
            on_transition=self._handle_transition,
            on_poll=self._handle_poll,
            on_initial_app=self._handle_initial_app,
            event_bus=self._event_bus,
        )

    @staticmethod
    def _resolve(saved: dict[str, Any], key: str, cli_val: Any, default: Any) -> Any:
        """Return *cli_val* when it was explicitly set, else the persisted value."""
        if cli_val != default:
            return cli_val
        return saved.get(key, default)

    def _get_last_label_end(self) -> dt.datetime | None:
        """Return the latest label ``end_ts``, using a cache keyed on save count."""
        if self._last_label_cache_count == self._labels_saved_count:
            return self._last_label_end_cache

        self._last_label_cache_count = self._labels_saved_count

        if not self._labels_path.exists():
            self._last_label_end_cache = None
            return None

        try:
            from taskclf.labels.store import read_label_spans

            spans = read_label_spans(self._labels_path)
            if not spans:
                self._last_label_end_cache = None
                return None
            latest = max(s.end_ts for s in spans)
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=dt.timezone.utc)
            self._last_label_end_cache = latest
            return latest
        except Exception:
            logger.debug("Could not read labels for gap-fill", exc_info=True)
            self._last_label_end_cache = None
            return None

    @staticmethod
    def _format_duration(minutes: float) -> str:
        """Format a duration in minutes to a human-readable string like '2h 30m'."""
        total = int(minutes)
        if total < 1:
            return "0m"
        hours, mins = divmod(total, 60)
        if hours and mins:
            return f"{hours}h {mins}m"
        if hours:
            return f"{hours}h"
        return f"{mins}m"

    def _publish_unlabeled_time(self) -> None:
        """Compute unlabeled time and publish an ``unlabeled_time`` event."""
        if self._event_bus is None:
            return

        last_end = self._get_last_label_end()
        now = dt.datetime.now(dt.timezone.utc)

        if last_end is None:
            self._unlabeled_minutes = 0.0
            return

        delta = (now - last_end).total_seconds() / 60.0
        self._unlabeled_minutes = max(0.0, delta)

        if self._unlabeled_minutes <= 0:
            return

        from taskclf.ui.copy import gap_fill_prompt

        duration_str = self._format_duration(self._unlabeled_minutes)
        self._event_bus.publish_threadsafe(
            {
                "type": "unlabeled_time",
                "unlabeled_minutes": round(self._unlabeled_minutes, 1),
                "text": gap_fill_prompt(duration_str),
                "last_label_end": last_end.isoformat(),
                "ts": now.isoformat(),
            }
        )

        self._check_escalation()

    def _check_escalation(self) -> None:
        """Publish ``gap_fill_escalated`` and update icon when threshold is exceeded."""
        should_escalate = self._unlabeled_minutes >= self._gap_fill_escalation_minutes
        if should_escalate and not self._escalated:
            self._escalated = True
            if self._event_bus is not None:
                self._event_bus.publish_threadsafe(
                    {
                        "type": "gap_fill_escalated",
                        "unlabeled_minutes": round(self._unlabeled_minutes, 1),
                        "threshold_minutes": self._gap_fill_escalation_minutes,
                    }
                )
            if self._icon is not None:
                self._icon.icon = _make_icon_image(color="#FF9800")
        elif not should_escalate and self._escalated:
            self._escalated = False
            if self._icon is not None:
                self._icon.icon = _make_icon_image()

    def _publish_gap_fill_prompt(self, trigger: str) -> None:
        """Publish a ``gap_fill_prompt`` event if unlabeled time exists.

        Args:
            trigger: One of ``"idle_return"``, ``"session_start"``,
                or ``"post_acceptance"``.
        """
        if self._event_bus is None:
            return

        last_end = self._get_last_label_end()
        now = dt.datetime.now(dt.timezone.utc)
        if last_end is None:
            return

        minutes = max(0.0, (now - last_end).total_seconds() / 60.0)
        if minutes <= 0:
            return

        self._unlabeled_minutes = minutes

        from taskclf.ui.copy import gap_fill_prompt

        duration_str = self._format_duration(minutes)
        self._event_bus.publish_threadsafe(
            {
                "type": "gap_fill_prompt",
                "trigger": trigger,
                "unlabeled_minutes": round(minutes, 1),
                "text": gap_fill_prompt(duration_str),
                "last_label_end": last_end.isoformat(),
                "ts": now.isoformat(),
            }
        )

    def _on_suggestion_accepted(self) -> None:
        """Called when the user accepts a transition suggestion.

        Publishes a ``gap_fill_prompt`` event if adjacent unlabeled time
        exists, piggybacking on the user's labeling attention.
        """
        self._last_label_cache_count = -1
        self._publish_gap_fill_prompt("post_acceptance")

    def _handle_initial_app(self, app: str, ts: dt.datetime) -> None:
        """Publish an initial_app event so the UI can prompt for the pre-start period."""
        if self._event_bus is not None:
            self._event_bus.publish_threadsafe(
                {
                    "type": "initial_app",
                    "app": app,
                    "ts": ts.isoformat(),
                }
            )
        self._publish_gap_fill_prompt("session_start")

    def _on_label_saved(self) -> None:
        """Increment the saved-label counter (called by the embedded server)."""
        self._labels_saved_count += 1

    def _on_model_trained(self, model_dir_str: str) -> None:
        """Auto-reload the model when training completes via the web UI."""
        model_path = Path(model_dir_str)
        if not model_path.is_dir():
            return

        if self._models_dir is not None:
            try:
                new_suggester = _LabelSuggester.from_policy(self._models_dir)
                new_suggester._aw_host = self._aw_host
                new_suggester._title_salt = self._title_salt
                new_suggester._user_id = self._config.user_id
                self._suggester = new_suggester
                self._model_dir = model_path
                self._model_schema_hash = new_suggester._predictor.metadata.schema_hash
                logger.info("Auto-loaded via inference policy after training")
                return
            except Exception:
                logger.debug(
                    "Policy load failed after training; using bundle directly",
                    exc_info=True,
                )

        try:
            new_suggester = _LabelSuggester(model_path)
            new_suggester._aw_host = self._aw_host
            new_suggester._title_salt = self._title_salt
            new_suggester._user_id = self._config.user_id
            self._suggester = new_suggester
            self._model_dir = model_path
            self._model_schema_hash = new_suggester._predictor.metadata.schema_hash
            logger.info("Auto-loaded newly trained model from %s", model_path)
        except Exception:
            logger.warning(
                "Could not auto-load trained model from %s", model_path, exc_info=True
            )

    def _tray_state_event(self) -> dict[str, Any]:
        """Build the latest tray state payload for WebSocket and snapshot clients."""
        return {
            "type": "tray_state",
            "model_loaded": self._suggester is not None,
            "model_dir": str(self._model_dir) if self._model_dir else None,
            "model_schema_hash": self._model_schema_hash,
            "suggested_label": self._suggested_label,
            "suggested_confidence": self._suggested_confidence,
            "transition_count": self._transition_count,
            "last_transition": self._last_transition,
            "labels_saved_count": self._labels_saved_count,
            "data_dir": str(self._data_dir),
            "ui_port": self._ui_port,
            "dev_mode": self._dev,
            "paused": self._monitor.is_paused,
        }

    def _publish_tray_state(self) -> None:
        """Publish the current tray state when the shared EventBus is ready."""
        if self._event_bus is None:
            return
        self._event_bus.publish_threadsafe(self._tray_state_event())

    def _model_configured(self) -> bool:
        """Return True when startup should try to load a suggester."""
        return self._models_dir is not None or self._model_dir is not None

    def _load_initial_suggester(self) -> None:
        """Load the optional suggester after UI startup to reduce cold-start latency."""
        if self._models_dir is not None:
            try:
                self._suggester = _LabelSuggester.from_policy(self._models_dir)
                self._suggester._aw_host = self._aw_host
                self._suggester._title_salt = self._title_salt
                self._suggester._user_id = self._config.user_id
                self._model_schema_hash = (
                    self._suggester._predictor.metadata.schema_hash
                )
                logger.info("Model loaded via inference policy")
            except Exception:
                logger.debug(
                    "No inference policy; trying model_dir fallback", exc_info=True
                )
                self._suggester = None

        if self._suggester is None and self._model_dir is not None:
            try:
                self._suggester = _LabelSuggester(self._model_dir)
                self._suggester._aw_host = self._aw_host
                self._suggester._title_salt = self._title_salt
                self._suggester._user_id = self._config.user_id
                self._model_schema_hash = (
                    self._suggester._predictor.metadata.schema_hash
                )
                logger.info("Model loaded from %s", self._model_dir)
            except Exception:
                logger.warning(
                    "Could not load model from %s", self._model_dir, exc_info=True
                )

        if self._event_bus.wait_ready(timeout=30):
            self._publish_tray_state()

    def _start_initial_model_load(self) -> None:
        """Start lazy model loading once the UI server launch path has been kicked off."""
        if self._initial_model_load_started or not self._model_configured():
            return
        self._initial_model_load_started = True
        threading.Thread(
            target=self._load_initial_suggester,
            daemon=True,
            name="taskclf-model-load",
        ).start()

    def _toggle_pause(self) -> bool:
        """Toggle pause state on the monitor. Returns new paused state."""
        if self._monitor.is_paused:
            self._monitor.resume()
        else:
            self._monitor.pause()
        return self._monitor.is_paused

    def _handle_poll(self, dominant_app: str) -> None:
        self._current_app = dominant_app
        self._publish_tray_state()
        self._publish_live_status()
        self._publish_unlabeled_time()

    def _publish_live_status(self) -> None:
        """Predict the current bucket and publish a ``live_status`` event.

        This is a passive, glanceable status separate from transition
        suggestions.  It uses only the latest single bucket (SEM-002).
        """
        if self._suggester is None or self._event_bus is None:
            return

        now = dt.datetime.now(dt.timezone.utc)
        bucket_start = now.replace(second=0, microsecond=0)
        bucket_end = now

        result = self._suggester.suggest(bucket_start, bucket_end)
        if result is None:
            return

        label, _confidence = result

        from taskclf.ui.copy import live_status_text

        self._event_bus.publish_threadsafe(
            {
                "type": "live_status",
                "label": label,
                "text": live_status_text(label),
                "ts": now.isoformat(),
            }
        )

    def _is_breakidle_block(self, prev_app: str) -> bool:
        """Return True when the completed block should be auto-labeled BreakIdle.

        A block qualifies when:
        - ``prev_app`` is a known lockscreen/screensaver app ID, OR
        - the model suggested ``BreakIdle`` for this block.

        ``prev_app`` is a normalized reverse-domain app ID (e.g.
        ``com.apple.loginwindow``) as returned by
        :func:`~taskclf.adapters.activitywatch.mapping.normalize_app`.
        """
        if prev_app in _LOCKSCREEN_APP_IDS:
            return True
        if self._suggested_label == "BreakIdle":
            return True
        return False

    def _auto_save_breakidle(
        self,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        """Write a BreakIdle label span directly without user confirmation."""
        from taskclf.core.types import LabelSpan
        from taskclf.labels.store import overwrite_label_span

        uid = self._config.user_id
        span = LabelSpan(
            start_ts=block_start,
            end_ts=block_end,
            label="BreakIdle",
            provenance="auto_idle",
            user_id=uid,
            confidence=1.0,
        )
        try:
            overwrite_label_span(span, self._labels_path)
            self._on_label_saved()
            logger.info(
                "Auto-saved BreakIdle label: %s → %s",
                block_start.isoformat(),
                block_end.isoformat(),
            )
        except Exception:
            logger.warning("Failed to auto-save BreakIdle label", exc_info=True)

        if self._event_bus is not None:
            self._event_bus.publish_threadsafe(
                {
                    "type": "label_created",
                    "label": "BreakIdle",
                    "confidence": 1.0,
                    "ts": block_end.isoformat(),
                    "start_ts": block_start.isoformat(),
                    "extend_forward": False,
                }
            )
            self._event_bus.publish_threadsafe(
                {"type": "suggestion_cleared", "reason": "auto_saved_breakidle"}
            )
            self._event_bus.publish_threadsafe(
                {
                    "type": "labels_changed",
                    "reason": "auto_saved_breakidle",
                    "ts": block_end.isoformat(),
                }
            )

    def _handle_transition(
        self,
        prev_app: str,
        new_app: str,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        self._transition_count += 1
        self._last_transition = {
            "prev_app": prev_app,
            "new_app": new_app,
            "block_start": block_start.isoformat(),
            "block_end": block_end.isoformat(),
            "fired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }

        suggestion = None
        if self._suggester is not None:
            suggestion = self._suggester.suggest(block_start, block_end)

        if suggestion is not None:
            self._suggested_label, self._suggested_confidence = suggestion
        else:
            self._suggested_label = None
            self._suggested_confidence = None

        is_lockscreen = prev_app in _LOCKSCREEN_APP_IDS
        is_breakidle = self._is_breakidle_block(prev_app)
        logger.debug(
            "DEBUG transition: prev_app=%r -> new_app=%r, "
            "is_lockscreen=%s, suggested_label=%r, is_breakidle=%s",
            prev_app,
            new_app,
            is_lockscreen,
            self._suggested_label,
            is_breakidle,
        )
        if is_breakidle:
            self._auto_save_breakidle(block_start, block_end)
            idle_duration_min = (block_end - block_start).total_seconds() / 60.0
            if is_lockscreen and idle_duration_min > 5:
                self._last_label_cache_count = -1
                self._publish_gap_fill_prompt("idle_return")
            return

        if self._event_bus is None or not self._event_bus.has_subscribers:
            self._send_notification(prev_app, new_app, block_start, block_end)

        if self._event_bus is not None:
            from taskclf.ui.copy import transition_suggestion_text

            start_str = _display_clock_time_local(block_start)
            end_str = _display_clock_time_local(block_end)
            suggestion_text = (
                transition_suggestion_text(self._suggested_label, start_str, end_str)
                if self._suggested_label is not None
                else None
            )
            self._event_bus.publish_threadsafe(
                {
                    "type": "prompt_label",
                    "prev_app": prev_app,
                    "new_app": new_app,
                    "block_start": block_start.isoformat(),
                    "block_end": block_end.isoformat(),
                    "duration_min": max(
                        1, int((block_end - block_start).total_seconds() / 60)
                    ),
                    "suggested_label": self._suggested_label,
                    "suggestion_text": suggestion_text,
                }
            )
            if (
                self._suggested_label is not None
                and self._suggested_confidence is not None
            ):
                self._event_bus.publish_threadsafe(
                    {
                        "type": "suggest_label",
                        "reason": "app_switch",
                        "old_label": prev_app,
                        "suggested": self._suggested_label,
                        "confidence": self._suggested_confidence,
                        "block_start": block_start.isoformat(),
                        "block_end": block_end.isoformat(),
                    }
                )
            else:
                self._event_bus.publish_threadsafe(
                    {
                        "type": "no_model_transition",
                        "current_app": new_app,
                        "ts": block_end.isoformat(),
                        "block_start": block_start.isoformat(),
                        "block_end": block_end.isoformat(),
                    }
                )

    def _send_notification(
        self,
        prev_app: str,
        new_app: str,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        if not self._notifications_enabled:
            return

        from taskclf.ui.copy import transition_suggestion_text

        title = "taskclf — Activity changed"
        start_str = _display_clock_time_local(block_start)
        end_str = _display_clock_time_local(block_end)
        range_str = _display_time_range_exact_local(block_start, block_end)

        if self._suggested_label is not None:
            message = (
                f"{transition_suggestion_text(self._suggested_label, start_str, end_str)}"
                f"\n{range_str}"
            )
        elif self._privacy_notifications:
            message = f"Activity changed\n{range_str}"
        else:
            message = f"{prev_app} \u2192 {new_app}\n{range_str}"

        _send_desktop_notification(title, message, timeout=10)

    def _build_menu_items(self) -> tuple["pystray.MenuItem", ...]:
        """Return top-level menu items.

        Used as a callable by ``pystray.Menu`` so the menu is rebuilt
        (including a fresh Prediction Model submenu scan) on every right-click.
        """
        import pystray

        return (
            pystray.MenuItem(
                "Toggle Dashboard",
                self._open_dashboard,
                default=True,
            ),
            pystray.MenuItem(
                lambda _: "Resume" if self._monitor.is_paused else "Pause",
                self._on_pause_menu,
            ),
            pystray.MenuItem("Show Status", self._show_status),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Today's Labels", self._label_stats),
            pystray.MenuItem("Import Labels", self._import_labels),
            pystray.MenuItem("Export Labels", self._export_labels),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Prediction Model", self._build_model_submenu()),
            pystray.MenuItem("Open Data Folder", self._open_data_dir),
            pystray.MenuItem("Edit Config", self._edit_config),
            pystray.MenuItem("Advanced", self._build_advanced_submenu()),
            pystray.MenuItem("Report Issue", self._report_issue),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    def _build_menu(self) -> "pystray.Menu":
        """Build a static snapshot of the menu (used by tests)."""
        import pystray

        return pystray.Menu(*self._build_menu_items())

    def _on_pause_menu(self, *_args: Any) -> None:
        paused = self._toggle_pause()
        state = "paused" if paused else "resumed"
        self._notify(f"Monitoring {state}")

    def _export_labels(self, *_args: Any) -> None:
        from taskclf.labels.store import export_labels_to_csv

        csv_path: Path | None = None
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            chosen = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="labels_export.csv",
                title="Export Labels",
            )
            root.destroy()
            if not chosen:
                return
            csv_path = Path(chosen)
        except Exception:
            logger.debug("tkinter unavailable, using default export path")
            csv_path = self._data_dir / "labels_v1" / "labels_export.csv"

        try:
            export_labels_to_csv(self._labels_path, csv_path)
            self._notify_with_reveal(
                f"Labels exported to {csv_path.name}",
                csv_path,
            )
            logger.info("Labels exported to %s", csv_path)
        except ValueError as exc:
            self._notify(f"Export failed: {exc}")
            logger.warning("Label export failed: %s", exc)

    def _import_labels(self, *_args: Any) -> None:
        from taskclf.labels.store import (
            import_labels_from_csv,
            merge_label_spans,
            read_label_spans,
            write_label_spans,
        )

        csv_path: Path | None = None
        strategy: str | None = None
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox

            root = tk.Tk()
            root.withdraw()
            chosen = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Import Labels",
            )
            if not chosen:
                root.destroy()
                return
            csv_path = Path(chosen)

            answer = messagebox.askyesnocancel(
                "Import Strategy",
                "Merge with existing labels?\n\n"
                "Yes = merge (keep existing, add new)\n"
                "No = overwrite (replace all labels)",
                parent=root,
            )
            root.destroy()
            if answer is None:
                return
            strategy = "merge" if answer else "overwrite"
        except Exception:
            logger.debug("tkinter unavailable for import dialog, trying osascript")
            csv_path, strategy = self._import_labels_osascript()
            if csv_path is None:
                return

        try:
            imported = import_labels_from_csv(csv_path)
        except (ValueError, Exception) as exc:
            self._notify(f"Import failed: {exc}")
            logger.warning("Label import failed: %s", exc)
            return

        try:
            if strategy == "overwrite":
                write_label_spans(imported, self._labels_path)
            else:
                existing: list = []
                if self._labels_path.exists():
                    existing = read_label_spans(self._labels_path)
                merged = merge_label_spans(existing, imported)
                write_label_spans(merged, self._labels_path)

            self._notify(f"Imported {len(imported)} labels from {csv_path.name}")
            logger.info(
                "Imported %d labels from %s (strategy=%s)",
                len(imported),
                csv_path,
                strategy,
            )
        except ValueError as exc:
            self._notify(f"Import failed: {exc}")
            logger.warning("Label import failed: %s", exc)

    def _import_labels_osascript(self) -> tuple[Path | None, str | None]:
        """macOS fallback for import file dialog using osascript."""
        if platform.system() != "Darwin":
            self._notify("Import failed: no file dialog available")
            return None, None
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'POSIX path of (choose file of type {"csv"}'
                    ' with prompt "Import Labels")',
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None, None
            csv_path = Path(result.stdout.strip())

            btn = subprocess.run(
                [
                    "osascript",
                    "-e",
                    "button returned of (display dialog"
                    ' "Merge with existing labels?\\n\\n'
                    "Merge = keep existing, add new\\n"
                    'Overwrite = replace all labels"'
                    ' buttons {"Cancel","Overwrite","Merge"}'
                    ' default button "Merge")',
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if btn.returncode != 0 or not btn.stdout.strip():
                return None, None
            strategy = "merge" if btn.stdout.strip() == "Merge" else "overwrite"
            return csv_path, strategy
        except Exception as exc:
            logger.debug("osascript import dialog failed: %s", exc)
            self._notify("Import failed: no file dialog available")
            return None, None

    def _label_stats(self, *_args: Any) -> None:
        """Show a notification with today's labeling progress."""
        from taskclf.labels.store import read_label_spans

        if not self._labels_path.exists():
            self._notify("No labels yet")
            return

        try:
            spans = read_label_spans(self._labels_path)
        except Exception as exc:
            self._notify(f"Could not read labels: {exc}")
            return

        today = dt.datetime.now(dt.timezone.utc).date()
        today_spans = [s for s in spans if s.start_ts.date() == today]

        if not today_spans:
            self._notify("Today: no labels yet")
            return

        breakdown: dict[str, float] = {}
        for s in today_spans:
            mins = (s.end_ts - s.start_ts).total_seconds() / 60
            breakdown[s.label] = breakdown.get(s.label, 0) + mins

        total_min = sum(breakdown.values())
        hours = int(total_min // 60)
        mins = int(total_min % 60)
        time_str = f"{hours}h {mins}m" if hours else f"{mins}m"

        parts = [
            f"{label} {self._format_duration(m)}"
            for label, m in sorted(
                breakdown.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        summary = f"Today: {len(today_spans)} labels, {time_str} — {', '.join(parts)}"
        self._notify(summary)

    def _open_data_dir(self, *_args: Any) -> None:
        """Open the data directory in the OS file manager."""
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", str(self._data_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self._data_dir)])
        except Exception:
            logger.debug("Could not open data directory", exc_info=True)
            self._notify(f"Data dir: {self._data_dir}")

    def _edit_config(self, *_args: Any) -> None:
        """Open ``config.toml`` in the default text editor."""
        config_path = self._config._path
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", "-t", str(config_path)])
            else:
                subprocess.Popen(["xdg-open", str(config_path)])
        except Exception:
            logger.debug("Could not open config file", exc_info=True)
            self._notify(f"Config: {config_path}")

    def _candidate_calibrator_store_dirs(self, base: Path) -> list[Path]:
        """Return candidate calibrator-store directories under ``artifacts/``."""
        artifacts_dir = base / "artifacts"
        if not artifacts_dir.is_dir():
            return []

        default_store = artifacts_dir / "calibrator_store"
        candidates: list[Path] = []
        seen: set[Path] = set()

        for store_dir in [
            default_store,
            *(p.parent for p in artifacts_dir.rglob("store.json")),
        ]:
            if store_dir in seen:
                continue
            seen.add(store_dir)
            if not (store_dir / "store.json").is_file():
                continue
            if not (store_dir / "global.json").is_file():
                continue
            candidates.append(store_dir)
        return candidates

    def _find_matching_calibrator_store(
        self,
        *,
        models_dir: Path,
        model_bundle: Path,
        model_schema_hash: str,
    ) -> tuple[str | None, str | None]:
        """Return ``(relative_store_dir, method)`` for a matching store.

        Only stores with explicit model binding metadata are auto-selected,
        which avoids guessing across unrelated calibration outputs.
        """
        base = models_dir.parent
        matches: list[tuple[int, int, int, str, Path, str | None]] = []

        for store_dir in self._candidate_calibrator_store_dirs(base):
            try:
                store_meta = json.loads((store_dir / "store.json").read_text())
            except json.JSONDecodeError, OSError:
                logger.debug(
                    "Could not inspect calibrator store metadata at %s",
                    store_dir,
                    exc_info=True,
                )
                continue

            store_bundle_id = store_meta.get("model_bundle_id")
            store_schema_hash = store_meta.get("model_schema_hash")
            if store_bundle_id is None and store_schema_hash is None:
                continue
            if (
                store_bundle_id is not None
                and str(store_bundle_id) != model_bundle.name
            ):
                continue
            if (
                store_schema_hash is not None
                and str(store_schema_hash) != model_schema_hash
            ):
                continue

            method_raw = store_meta.get("method")
            method = method_raw if isinstance(method_raw, str) else None
            created_at_raw = store_meta.get("created_at")
            created_at = created_at_raw if isinstance(created_at_raw, str) else ""
            matches.append(
                (
                    1 if store_bundle_id == model_bundle.name else 0,
                    1 if store_schema_hash == model_schema_hash else 0,
                    1 if store_dir.name == "calibrator_store" else 0,
                    created_at,
                    store_dir,
                    method,
                )
            )

        if not matches:
            return None, None

        _, _, _, _, best_dir, best_method = max(matches, key=lambda item: item[:4])
        return str(best_dir.relative_to(base)), best_method

    def _ensure_inference_policy_file_for_editing(
        self,
    ) -> tuple[Path | None, str | None]:
        """Return ``(policy_path, notice)`` for editing.

        When ``inference_policy.json`` is missing, creates it only when a
        real model bundle can be resolved and seeded.

        Returns:
            ``(None, None)`` when ``models_dir`` is not configured.
            ``(path, None)`` when a policy file is ready to edit.
            ``(None, notice)`` when no resolved model is available; *notice*
            explains in-app steps first, with an optional CLI hint.
        """
        if self._models_dir is None:
            return None, None

        models_dir = self._models_dir
        policy_path = models_dir / DEFAULT_INFERENCE_POLICY_FILE
        if policy_path.is_file():
            return policy_path, None

        from taskclf.core.inference_policy import (
            build_inference_policy,
            PolicyValidationError,
            save_inference_policy,
            validate_policy,
        )
        from taskclf.infer.resolve import ModelResolutionError, resolve_model_dir

        models_dir.mkdir(parents=True, exist_ok=True)

        model_bundle: Path | None = None
        if (
            self._model_dir is not None
            and (self._model_dir / "metadata.json").is_file()
        ):
            model_bundle = self._model_dir
        else:
            try:
                model_bundle = resolve_model_dir(None, models_dir)
            except ModelResolutionError:
                model_bundle = None

        if model_bundle is not None:
            meta_path = model_bundle / "metadata.json"
            try:
                meta = json.loads(meta_path.read_text())
                model_schema_hash = str(meta["schema_hash"])
                model_label_set = list(meta["label_set"])
            except KeyError, TypeError, ValueError, json.JSONDecodeError, OSError:
                logger.debug(
                    "Could not seed inference policy from %s; use CLI instead",
                    model_bundle,
                    exc_info=True,
                )
            else:
                raw_threshold = meta.get("reject_threshold")
                try:
                    reject_threshold = (
                        float(raw_threshold)
                        if raw_threshold is not None
                        else DEFAULT_REJECT_THRESHOLD
                    )
                except TypeError, ValueError:
                    reject_threshold = DEFAULT_REJECT_THRESHOLD

                cal_store_rel, cal_method = self._find_matching_calibrator_store(
                    models_dir=models_dir,
                    model_bundle=model_bundle,
                    model_schema_hash=model_schema_hash,
                )
                policy = build_inference_policy(
                    model_dir=os.path.relpath(model_bundle, models_dir.parent),
                    model_schema_hash=model_schema_hash,
                    model_label_set=model_label_set,
                    reject_threshold=reject_threshold,
                    calibrator_store_dir=cal_store_rel,
                    calibration_method=cal_method,
                    source="tray-edit",
                )
                if cal_store_rel is not None:
                    try:
                        validate_policy(policy, models_dir)
                    except PolicyValidationError:
                        logger.debug(
                            "Ignoring detected calibrator store %s for starter policy",
                            cal_store_rel,
                            exc_info=True,
                        )
                        policy = policy.model_copy(
                            update={
                                "calibrator_store_dir": None,
                                "calibration_method": None,
                            }
                        )
                written = save_inference_policy(policy, models_dir)
                return written, None

        return (
            None,
            "No model available to seed inference_policy.json. "
            "Use Prediction Model or Open Data Folder (models/ next to your data folder). "
            "If you have the CLI: taskclf policy create --model-dir models/<run_id>",
        )

    def _edit_inference_policy(self, *_args: Any) -> None:
        """Open ``inference_policy.json`` in the default text editor.

        Creates the file when missing only if a resolved model can seed it.
        Otherwise, notifies the user with in-app guidance and an optional CLI hint.
        """
        if self._models_dir is None:
            self._notify("No models directory configured")
            return

        policy_path, notice = self._ensure_inference_policy_file_for_editing()
        if notice is not None:
            self._notify(notice)
        if policy_path is None:
            return

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", "-t", str(policy_path)])
            else:
                subprocess.Popen(["xdg-open", str(policy_path)])
        except Exception:
            logger.debug("Could not open inference policy file", exc_info=True)
            self._notify(f"Inference policy: {policy_path}")

    def _reload_model(self, *_args: Any) -> None:
        """Re-read the model bundle from disk without restarting."""
        if self._models_dir is not None:
            try:
                new_suggester = _LabelSuggester.from_policy(self._models_dir)
                new_suggester._aw_host = self._aw_host
                new_suggester._title_salt = self._title_salt
                new_suggester._user_id = self._config.user_id
                self._suggester = new_suggester
                self._model_schema_hash = new_suggester._predictor.metadata.schema_hash
                self._notify("Config reloaded via inference policy")
                logger.info("Config reloaded via inference policy")
                return
            except Exception:
                logger.debug("Policy reload failed; trying model_dir", exc_info=True)

        if self._model_dir is None:
            self._notify("No model directory configured")
            return
        try:
            new_suggester = _LabelSuggester(self._model_dir)
            new_suggester._aw_host = self._aw_host
            new_suggester._title_salt = self._title_salt
            new_suggester._user_id = self._config.user_id
            self._suggester = new_suggester
            self._model_schema_hash = new_suggester._predictor.metadata.schema_hash
            self._notify(f"Model reloaded from {self._model_dir.name}")
            logger.info("Model reloaded from %s", self._model_dir)
        except Exception as exc:
            self._notify(f"Reload failed: {exc}")
            logger.warning("Model reload failed: %s", exc, exc_info=True)

    def _check_retrain(self, *_args: Any) -> None:
        """Check whether retraining is due and show a notification."""
        if self._models_dir is None:
            self._notify("No models directory configured")
            return

        try:
            import json

            from taskclf.train.retrain import (
                RetrainConfig,
                check_retrain_due,
                find_latest_model,
                load_retrain_config,
            )

            config = (
                load_retrain_config(self._retrain_config)
                if self._retrain_config is not None and self._retrain_config.is_file()
                else RetrainConfig()
            )

            latest = find_latest_model(self._models_dir)
            due = check_retrain_due(
                self._models_dir,
                config.global_retrain_cadence_days,
            )

            if latest is not None:
                raw = json.loads((latest / "metadata.json").read_text())
                created = raw.get("created_at", "unknown")
                if due:
                    self._notify(
                        f"Retrain recommended "
                        f"(cadence: {config.global_retrain_cadence_days}d, "
                        f"last: {latest.name} created {created})"
                    )
                else:
                    self._notify(f"Model is current ({latest.name}, created {created})")
            else:
                self._notify("Retrain recommended: no models found")
        except Exception as exc:
            self._notify(f"Check failed: {exc}")
            logger.warning("Retrain check failed: %s", exc, exc_info=True)

    def _build_model_submenu(self) -> "pystray.Menu":
        """Build a dynamic submenu listing available model bundles."""
        import pystray

        from taskclf.model_registry import list_bundles

        items: list[pystray.MenuItem] = []

        bundles = list_bundles(self._models_dir) if self._models_dir is not None else []
        valid_bundles = [b for b in bundles if b.valid]

        if valid_bundles:
            for bundle in valid_bundles:
                model_path = bundle.path

                def make_switch_cb(p: Path) -> Callable[..., None]:
                    def cb(*_a: Any) -> None:
                        self._switch_model(p)

                    return cb

                def make_checked(p: Path) -> Callable[..., bool]:
                    return lambda _item: (
                        self._model_dir is not None
                        and self._model_dir.resolve() == p.resolve()
                    )

                items.append(
                    pystray.MenuItem(
                        bundle.model_id,
                        make_switch_cb(model_path),
                        checked=make_checked(model_path),
                    )
                )

            items.append(
                pystray.MenuItem(
                    "No Model",
                    self._unload_model,
                    checked=lambda _item: self._model_dir is None,
                )
            )
            items.append(pystray.Menu.SEPARATOR)
            items.append(
                pystray.MenuItem(
                    "Refresh Model",
                    self._reload_model,
                    enabled=lambda _: self._model_dir is not None,
                )
            )
            items.append(
                pystray.MenuItem(
                    "Retrain Status",
                    self._check_retrain,
                    enabled=lambda _: self._models_dir is not None,
                )
            )
        else:
            items.append(
                pystray.MenuItem(
                    "No Models Found",
                    None,
                    enabled=False,
                )
            )
            items.append(pystray.Menu.SEPARATOR)
            items.append(
                pystray.MenuItem(
                    "Refresh Model",
                    self._reload_model,
                    enabled=lambda _: self._model_dir is not None,
                )
            )
            items.append(
                pystray.MenuItem(
                    "Retrain Status",
                    self._check_retrain,
                    enabled=lambda _: self._models_dir is not None,
                )
            )

        return pystray.Menu(*items)

    def _build_advanced_submenu(self) -> "pystray.Menu":
        """Power-user actions (inference policy, etc.)."""
        import pystray

        return pystray.Menu(
            pystray.MenuItem(
                "Edit Inference Policy",
                self._edit_inference_policy,
                enabled=lambda _: self._models_dir is not None,
            ),
        )

    def _switch_model(self, model_path: Path) -> None:
        """Hot-swap the active model to a different bundle."""
        if (
            self._model_dir is not None
            and self._model_dir.resolve() == model_path.resolve()
        ):
            return

        try:
            new_suggester = _LabelSuggester(model_path)
            new_suggester._aw_host = self._aw_host
            new_suggester._title_salt = self._title_salt
            new_suggester._user_id = self._config.user_id
            self._suggester = new_suggester
            self._model_dir = model_path
            self._model_schema_hash = new_suggester._predictor.metadata.schema_hash
            self._notify(f"Switched to model {model_path.name}")
            logger.info("Switched to model %s", model_path)
        except Exception as exc:
            self._notify(f"Switch failed: {exc}")
            logger.warning(
                "Model switch to %s failed: %s", model_path, exc, exc_info=True
            )

    def _unload_model(self, *_args: Any) -> None:
        """Unload the current model entirely."""
        self._suggester = None
        self._model_dir = None
        self._model_schema_hash = None
        self._suggested_label = None
        self._suggested_confidence = None
        self._notify("Model unloaded")
        logger.info("Model unloaded")

    def _show_status(self, *_args: Any) -> None:
        """Show a notification with connection and session status."""
        aw_status = (
            "connected" if self._monitor._bucket_id is not None else "disconnected"
        )
        paused = " (paused)" if self._monitor.is_paused else ""
        model_name = self._model_dir.name if self._model_dir else "none"

        parts = [
            f"AW: {aw_status}{paused}",
            f"Polls: {self._monitor._poll_count}",
            f"Transitions: {self._transition_count}",
            f"Labels: {self._labels_saved_count}",
            f"Model: {model_name}",
        ]
        self._notify(" | ".join(parts))

    def _open_dashboard(self, *_args: Any) -> None:
        if self._browser:
            import webbrowser

            ui_port = (
                _VITE_DEV_PORT
                if (
                    self._dev
                    and self._vite_proc is not None
                    and self._vite_proc.poll() is None
                )
                else self._ui_port
            )
            webbrowser.open(f"http://127.0.0.1:{ui_port}")
            return

        if self._ui_proc is not None and self._ui_proc.poll() is None:
            logger.debug("Sending toggle to UI process (pid=%s)", self._ui_proc.pid)
            try:
                self._ui_proc.stdin.write(b"toggle\n")
                self._ui_proc.stdin.flush()
            except BrokenPipeError, OSError:
                logger.debug("Could not send toggle to UI process", exc_info=True)
            return

        logger.debug("No running UI process — spawning new window")
        self._spawn_window()

    def _quit(self, *_args: Any) -> None:
        self._monitor.stop()
        self._cleanup_ui()
        if self._icon is not None:
            self._icon.stop()

    def _notify(self, message: str) -> None:
        _send_desktop_notification("taskclf", message, timeout=5)

    def _notify_with_reveal(self, message: str, path: Path) -> None:
        """Show a notification with an option to reveal *path* in the file manager.

        On macOS an AppleScript dialog with an "Show in Finder" button is
        displayed (auto-dismisses after 10 s).  On other platforms the
        containing folder is opened automatically alongside the notification.
        """
        folder = path.parent if path.is_file() else path
        if platform.system() == "Darwin":
            safe_msg = (
                message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            )
            safe_folder = str(folder).replace("\\", "\\\\").replace('"', '\\"')
            script = (
                f'set theResult to display dialog "{safe_msg}" '
                f'buttons {{"OK", "Show in Finder"}} default button "OK" '
                f"giving up after 10\n"
                f'if button returned of theResult is "Show in Finder" then\n'
                f'    do shell script "open \\"{safe_folder}\\""\n'
                f"end if"
            )
            try:
                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                return
            except Exception:
                logger.debug("osascript dialog failed, falling back", exc_info=True)

        self._notify(message)
        self._reveal_in_file_manager(folder)

    def _reveal_in_file_manager(self, path: Path) -> None:
        """Open *path* in the platform file manager."""
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            logger.debug("Could not open folder", exc_info=True)

    _MAX_ISSUE_URL_LEN = 8000

    def _build_report_issue_url(self) -> str:
        """Build a GitHub new-issue URL pre-filled with diagnostics and logs.

        Automatically runs the equivalent of ``taskclf diagnostics`` and
        reads the sanitized log tail from the user's data directory so
        the bug report template fields are pre-populated.
        """
        from urllib.parse import urlencode

        from taskclf.core.crash import _read_log_tail
        from taskclf.core.diagnostics import (
            collect_diagnostics,
            format_diagnostics_text,
        )
        from taskclf.core.paths import taskclf_home

        home = taskclf_home()
        models_dir = str(self._models_dir) if self._models_dir else str(home / "models")

        try:
            info = collect_diagnostics(
                aw_host=self._config.as_dict().get("aw_host", DEFAULT_AW_HOST),
                data_dir=str(self._data_dir),
                models_dir=models_dir,
                include_logs=False,
            )
            diagnostics_text = format_diagnostics_text(info)
        except Exception:
            logger.debug("Failed to collect diagnostics", exc_info=True)
            diagnostics_text = "<unable to collect diagnostics>"

        try:
            log_tail = _read_log_tail(home / "logs", 30)
            logs_text = "\n".join(log_tail) if log_tail else ""
        except Exception:
            logger.debug("Failed to read log tail", exc_info=True)
            logs_text = ""

        params: dict[str, str] = {
            "template": "bug_report.yml",
            "title": "[Bug]: ",
            "diagnostics": diagnostics_text,
        }
        if logs_text:
            params["logs"] = logs_text

        base = "https://github.com/fruitiecutiepie/taskclf/issues/new"
        url = f"{base}?{urlencode(params)}"

        if len(url) > self._MAX_ISSUE_URL_LEN:
            params.pop("logs", None)
            url = f"{base}?{urlencode(params)}"

        return url

    def _report_issue(self, *_args: Any) -> None:
        """Open the GitHub issue tracker in the default browser."""
        import webbrowser

        url = self._build_report_issue_url()
        webbrowser.open(url)

    def _start_server(self) -> int:
        """Start FastAPI + uvicorn in-process, sharing the tray's EventBus.

        Both ``--browser`` and native-window modes call this so that tray
        events (status, tray_state, suggest_label, prediction) are always
        visible to WebSocket clients.

        Returns:
            The effective UI port (may differ from ``self._ui_port`` when
            ``--dev`` starts a Vite dev server on ``_VITE_DEV_PORT``).
        """
        import os

        import uvicorn

        from taskclf.ui.server import create_app

        tray_actions: dict[str, Callable[..., Any]] = {
            "open_dashboard": self._open_dashboard,
            "pause_toggle": self._on_pause_menu,
            "label_stats": self._label_stats,
            "import_labels": self._import_labels,
            "export_labels": self._export_labels,
            "switch_model": self._switch_model,
            "unload_model": self._unload_model,
            "reload_model": self._reload_model,
            "check_retrain": self._check_retrain,
            "show_status": self._show_status,
            "open_data_dir": self._open_data_dir,
            "edit_config": self._edit_config,
            "edit_inference_policy": self._edit_inference_policy,
            "report_issue": self._report_issue,
            "quit": self._quit,
        }

        def get_tray_state() -> dict[str, Any]:
            return {
                "paused": self._monitor.is_paused,
                "model_dir": str(self._model_dir.resolve())
                if self._model_dir
                else None,
                "models_dir": str(self._models_dir.resolve())
                if self._models_dir
                else None,
            }

        fastapi_app = create_app(
            data_dir=self._data_dir,
            models_dir=self._models_dir,
            aw_host=self._aw_host,
            title_salt=self._title_salt,
            event_bus=self._event_bus,
            on_label_saved=self._on_label_saved,
            on_model_trained=self._on_model_trained,
            on_suggestion_accepted=self._on_suggestion_accepted,
            pause_toggle=self._toggle_pause,
            is_paused=lambda: self._monitor.is_paused,
            tray_actions=tray_actions,
            get_tray_state=get_tray_state,
            get_activity_provider_status=lambda: self._monitor.activity_provider_status,
        )

        uvicorn_config = uvicorn.Config(
            fastapi_app,
            host="127.0.0.1",
            port=self._ui_port,
            log_level="warning",
            ws_ping_interval=30,
            ws_ping_timeout=30,
        )
        server = uvicorn.Server(uvicorn_config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        self._ui_server_running = True

        print(f"taskclf API on http://127.0.0.1:{self._ui_port}", flush=True)

        ui_port = self._ui_port

        if self._dev:
            import taskclf.ui.server as _ui_srv

            frontend_dir = Path(_ui_srv.__file__).resolve().parent / "frontend"
            if not frontend_dir.is_dir():
                print(
                    "Warning: frontend source not found (--dev requires a repo checkout)"
                )
                return ui_port

            if not (frontend_dir / "node_modules").is_dir():
                print("Installing frontend dependencies…")
                subprocess.run(["pnpm", "install"], cwd=frontend_dir, check=True)

            vite_env = {
                **os.environ,
                "TASKCLF_PORT": str(self._ui_port),
                "TASKCLF_DEV_PORT": str(_VITE_DEV_PORT),
            }
            self._vite_proc = subprocess.Popen(
                ["pnpm", "run", "dev"],
                cwd=frontend_dir,
                env=vite_env,
            )
            ui_port = _VITE_DEV_PORT
            print(f"Vite dev server → http://127.0.0.1:{ui_port} (hot reload)")

            for _attempt in range(30):
                try:
                    import urllib.request

                    urllib.request.urlopen(f"http://127.0.0.1:{ui_port}", timeout=1)
                    break
                except Exception:
                    if self._vite_proc.poll() is not None:
                        print("Warning: Vite dev server exited unexpectedly")
                        return ui_port
                    time.sleep(0.5)
            else:
                print("Warning: Vite dev server not responding, opening anyway")

        return ui_port

    def _start_ui_subprocess(self) -> None:
        """Run FastAPI in-process and spawn a pywebview window subprocess.

        The server runs in-process so the tray's ``EventBus`` is shared
        with WebSocket clients.  Only the native window shell runs in a
        child process (no duplicate ``ActivityMonitor`` or ``EventBus``).
        """
        self._start_server()
        self._spawn_window()

    def _spawn_window(self) -> None:
        """Spawn a pywebview-only subprocess pointing at the in-process server."""
        import sys

        from taskclf.ui import window_run as _window_module

        try:
            cmd = [
                sys.executable,
                "-m",
                _window_module.__name__,
                "--port",
                str(self._ui_port),
            ]
            self._ui_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            mode = " (dev)" if self._dev else ""
            print(
                f"UI window launched{mode} (pid={self._ui_proc.pid}, port={self._ui_port})"
            )
        except Exception:
            logger.warning("Could not start UI window subprocess", exc_info=True)
            print(
                f"Warning: UI window failed to start. Dashboard at http://127.0.0.1:{self._ui_port}"
            )

    def _start_ui_embedded(self) -> None:
        """Run FastAPI in-process and optionally open the dashboard in a browser."""
        ui_port = self._start_server()
        mode = " (dev)" if self._dev else ""
        if self._open_browser:
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{ui_port}")
            print(f"UI opened in browser{mode} (port={ui_port})")
            return

        print(f"UI server ready{mode} (port={ui_port})")

    def _cleanup_ui(self) -> None:
        """Terminate UI and Vite subprocesses if still running."""
        if self._ui_proc is not None and self._ui_proc.poll() is None:
            self._ui_proc.terminate()
            try:
                self._ui_proc.wait(timeout=5)
            except Exception:
                logger.debug(
                    "UI process did not exit gracefully, killing", exc_info=True
                )
                self._ui_proc.kill()
        if self._vite_proc is not None and self._vite_proc.poll() is None:
            self._vite_proc.terminate()
            try:
                self._vite_proc.wait(timeout=5)
            except Exception:
                logger.debug(
                    "Vite process did not exit gracefully, killing", exc_info=True
                )
                self._vite_proc.kill()

    def run(self) -> None:
        """Start the tray icon and background monitor. Blocks until quit."""
        try:
            self._run_inner()
        except SystemExit, KeyboardInterrupt:
            raise
        except Exception as exc:
            from taskclf.core.crash import write_crash_report

            try:
                path = write_crash_report(exc)
                _send_desktop_notification(
                    "taskclf crashed",
                    f"Details saved to {path}",
                    timeout=10,
                )
            except Exception:
                logger.debug("Could not write crash report", exc_info=True)
            raise

    def _run_inner(self) -> None:
        """Actual run logic, separated so ``run()`` can wrap it."""
        import atexit

        from taskclf.core.logging import setup_file_logging

        setup_file_logging()

        if self._browser:
            self._start_ui_embedded()
        else:
            self._start_ui_subprocess()
        atexit.register(self._cleanup_ui)
        self._start_initial_model_load()

        monitor_thread = threading.Thread(
            target=self._monitor.run,
            daemon=True,
        )
        monitor_thread.start()

        if self._suggester is not None:
            mode = "with model suggestions"
        elif self._model_configured():
            mode = "loading model suggestions"
        else:
            mode = "label-only (no model)"

        if self._no_tray:
            # Duplicate to stderr so headless / frozen sidecars still show lines if
            # stdout is not attached (Electron spawn, some PyInstaller configs).
            for line in (
                f"taskclf running ({mode}), no tray icon.",
                f"UI available at http://127.0.0.1:{self._ui_port}",
                "Press Ctrl+C to quit.",
            ):
                print(line)
                try:
                    print(line, file=sys.stderr, flush=True)
                except Exception:
                    pass
            # threading.Event.wait() can return spuriously on some platforms; keep
            # the Electron sidecar alive until interrupt or process exit.
            try:
                while True:
                    time.sleep(86400.0)
            except KeyboardInterrupt:
                pass
            finally:
                self._monitor.stop()
                self._cleanup_ui()
            return

        import pystray

        icon_image = _make_icon_image()
        self._icon = pystray.Icon(
            "taskclf",
            icon_image,
            "taskclf",
            menu=pystray.Menu(self._build_menu_items),
        )

        print(f"taskclf tray started ({mode})")
        print(
            "Click the tray icon to open the dashboard. Press Ctrl+C or Quit to exit."
        )

        try:
            self._icon.run()
        finally:
            self._cleanup_ui()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_tray(
    *,
    model_dir: Path | None = None,
    models_dir: Path | None = None,
    aw_host: str = DEFAULT_AW_HOST,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    aw_timeout_seconds: int = DEFAULT_AW_TIMEOUT_SECONDS,
    title_salt: str = DEFAULT_TITLE_SALT,
    data_dir: Path = Path(DEFAULT_DATA_DIR),
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
    event_bus: EventBus | None = None,
    ui_port: int = 8741,
    dev: bool = False,
    browser: bool = False,
    no_tray: bool = False,
    open_browser: bool = True,
    username: str | None = None,
    notifications_enabled: bool = True,
    privacy_notifications: bool = True,
    retrain_config: Path | None = None,
) -> None:
    """Launch the system tray labeling app.

    Always starts the FastAPI server in-process so the tray's
    ``EventBus`` is shared with WebSocket clients.  In browser mode
    the dashboard opens in the default browser; otherwise a lightweight
    pywebview subprocess provides the native floating window.

    Args:
        model_dir: Optional path to a trained model bundle.  When
            provided, the tray suggests labels on activity transitions.
        models_dir: Optional path to the directory containing all model
            bundles.  Enables the "Prediction Model" submenu for hot-swapping.
        aw_host: ActivityWatch server URL.
        poll_seconds: Seconds between AW polling cycles.
        aw_timeout_seconds: Seconds to wait for AW API responses.
        title_salt: Salt for hashing window titles.
        data_dir: Processed data directory (labels stored here).
        transition_minutes: Minutes a new dominant app must persist
            before a transition notification fires.
        event_bus: Optional shared event bus for broadcasting events
            to connected WebSocket clients.
        ui_port: Port for the embedded web UI server.
        dev: When ``True``, the spawned UI subprocess starts a Vite
            dev server for frontend hot reload.
        browser: When ``True``, the spawned UI subprocess opens in the
            default browser instead of a native window.
        no_tray: When ``True``, skip the native tray icon entirely.
            The main thread blocks until interrupted.  Useful with
            ``--browser`` for a fully browser-based workflow.
        open_browser: When ``True``, browser mode launches the default
            browser automatically.  Set to ``False`` when another host
            shell (for example Electron) will render the web UI.
        username: Display name to persist in ``config.json``.  Does not
            affect label identity (labels use the stable auto-generated
            UUID ``user_id``).
        notifications_enabled: When ``False``, desktop notifications
            are suppressed entirely.
        privacy_notifications: When ``True`` (the default), app names
            are redacted from desktop notifications to protect privacy.
            Set to ``False`` to show raw app identifiers.
        retrain_config: Optional path to a retrain YAML config.
            Enables the "Retrain Status" item in the Prediction Model submenu.
    """
    tray = TrayLabeler(
        data_dir=data_dir,
        model_dir=model_dir,
        models_dir=models_dir,
        aw_host=aw_host,
        title_salt=title_salt,
        poll_seconds=poll_seconds,
        aw_timeout_seconds=aw_timeout_seconds,
        transition_minutes=transition_minutes,
        event_bus=event_bus,
        ui_port=ui_port,
        dev=dev,
        browser=browser,
        no_tray=no_tray,
        open_browser=open_browser,
        username=username,
        notifications_enabled=notifications_enabled,
        privacy_notifications=privacy_notifications,
        retrain_config=retrain_config,
    )
    tray.run()
