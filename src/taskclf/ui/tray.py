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
import logging
import platform
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray

from taskclf.core.config import UserConfig
from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)
from taskclf.ui.events import EventBus

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
        safe_message = message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        script = (
            f'display notification "{safe_message}" '
            f'with title "{safe_title}"'
        )

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
        on_initial_app: Callable[[str, dt.datetime], Any] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._aw_host = aw_host
        self._title_salt = title_salt
        self._poll_seconds = poll_seconds
        self._transition_threshold = transition_minutes * 60
        self._on_transition = on_transition
        self._on_poll = on_poll
        self._on_initial_app = on_initial_app
        self._event_bus = event_bus

        self._current_app: str | None = None
        self._current_app_since: dt.datetime | None = None
        self._candidate_app: str | None = None
        self._candidate_duration: int = 0
        self._candidate_first_seen: dt.datetime | None = None
        self._last_check_time: dt.datetime | None = None

        self._bucket_id: str | None = None
        self._aw_warned = False
        self._stop = threading.Event()
        self._paused = threading.Event()

        self._poll_count: int = 0
        self._last_event_count: int = 0
        self._last_app_counts: dict[str, int] = {}
        self._last_poll_ts: dt.datetime | None = None
        self._started_at: dt.datetime | None = None

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
                self._last_event_count = 0
                self._last_app_counts = {}
                return None

        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(seconds=self._poll_seconds)
        try:
            events = fetch_aw_events(
                self._aw_host, self._bucket_id, start, now,
                title_salt=self._title_salt,
            )
        except Exception:
            logger.debug("Failed to fetch AW events", exc_info=True)
            self._last_event_count = 0
            self._last_app_counts = {}
            return None

        if not events:
            self._last_event_count = 0
            self._last_app_counts = {}
            return None

        counts = Counter(ev.app_id for ev in events)
        self._last_event_count = len(events)
        self._last_app_counts = dict(counts.most_common(5))
        return counts.most_common(1)[0][0]

    def check_transition(
        self, dominant_app: str, *, _now: dt.datetime | None = None,
    ) -> None:
        """Update internal state and fire transition callback if warranted.

        Exposed as a public method so that transition logic can be unit-tested
        without requiring a live ActivityWatch server.

        Args:
            dominant_app: The current dominant foreground application.
            _now: Override for the current time (testing only).
        """
        now = _now or dt.datetime.now(dt.timezone.utc)
        elapsed = (
            int((now - self._last_check_time).total_seconds())
            if self._last_check_time is not None
            else self._poll_seconds
        )
        self._last_check_time = now

        if self._current_app is None:
            self._current_app = dominant_app
            self._current_app_since = now
            if self._on_initial_app is not None:
                self._on_initial_app(dominant_app, now)
            return

        if dominant_app != self._current_app:
            if self._candidate_app == dominant_app:
                self._candidate_duration += elapsed
                if self._candidate_duration >= self._transition_threshold:
                    block_start = self._current_app_since or now
                    block_end = self._candidate_first_seen or now
                    prev = self._current_app

                    self._current_app = dominant_app
                    self._current_app_since = block_end
                    self._candidate_app = None
                    self._candidate_duration = 0
                    self._candidate_first_seen = None

                    if self._on_transition is not None:
                        self._on_transition(prev, dominant_app, block_start, block_end)
            else:
                self._candidate_app = dominant_app
                self._candidate_duration = elapsed
                self._candidate_first_seen = now
        else:
            self._candidate_app = None
            self._candidate_duration = 0
            self._candidate_first_seen = None

    def _publish_status(
        self, dominant_app: str, *, state: str = "collecting",
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        self._last_poll_ts = now
        self._poll_count += 1

        if self._event_bus is not None:
            uptime_s = (
                int((now - self._started_at).total_seconds())
                if self._started_at else 0
            )
            self._event_bus.publish_threadsafe({
                "type": "status",
                "state": state,
                "current_app": dominant_app,
                "current_app_since": (
                    self._current_app_since.isoformat()
                    if self._current_app_since else None
                ),
                "candidate_app": self._candidate_app,
                "candidate_duration_s": self._candidate_duration,
                "transition_threshold_s": self._transition_threshold,
                "poll_seconds": self._poll_seconds,
                "poll_count": self._poll_count,
                "last_poll_ts": now.isoformat(),
                "uptime_s": uptime_s,
                "aw_connected": self._bucket_id is not None,
                "aw_bucket_id": self._bucket_id,
                "aw_host": self._aw_host,
                "last_event_count": self._last_event_count,
                "last_app_counts": self._last_app_counts,
            })

    def run(self) -> None:
        """Blocking poll loop. Call from a daemon thread."""
        if self._event_bus is not None:
            if not self._event_bus.wait_ready(timeout=30):
                logger.warning("EventBus loop not bound after 30s, starting anyway")
        self._started_at = dt.datetime.now(dt.timezone.utc)
        while not self._stop.is_set():
            if self._paused.is_set():
                app = self._current_app or "unknown"
                self._publish_status(app, state="paused")
            else:
                dominant = self._poll_dominant_app()
                app = dominant or self._current_app or "unknown"
                if self._on_poll is not None:
                    self._on_poll(app)
                self._publish_status(app)
                if dominant is not None:
                    self.check_transition(dominant)
            self._stop.wait(timeout=self._poll_seconds)

    def stop(self) -> None:
        """Signal the poll loop to stop."""
        self._stop.set()

    def pause(self) -> None:
        """Pause monitoring without clearing session state."""
        self._paused.set()

    def resume(self) -> None:
        """Resume monitoring after a pause."""
        self._last_check_time = None
        self._paused.clear()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

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
        models_dir: Optional path to the directory containing all model
            bundles.  When set, the tray builds a dynamic "Model" submenu
            listing available bundles for hot-swapping.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        poll_seconds: Seconds between AW polls.
        transition_minutes: Minutes for transition detection threshold.
        event_bus: Optional shared event bus for broadcasting events.
        ui_port: Port for the embedded UI server.
    """

    def __init__(
        self,
        *,
        data_dir: Path = Path(DEFAULT_DATA_DIR),
        model_dir: Path | None = None,
        models_dir: Path | None = None,
        aw_host: str = DEFAULT_AW_HOST,
        title_salt: str = DEFAULT_TITLE_SALT,
        poll_seconds: int = DEFAULT_POLL_SECONDS,
        transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
        event_bus: EventBus | None = None,
        ui_port: int = 8741,
        dev: bool = False,
        browser: bool = False,
        no_tray: bool = False,
        username: str | None = None,
        notifications_enabled: bool = True,
        privacy_notifications: bool = True,
        retrain_config: Path | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._model_dir = model_dir
        self._models_dir = models_dir
        self._retrain_config = retrain_config
        self._labels_path = data_dir / "labels_v1" / "labels.parquet"
        self._config = UserConfig(data_dir)
        if username is not None:
            self._config.username = username

        saved = self._config.as_dict()
        notifications_enabled = self._resolve(
            saved, "notifications_enabled", notifications_enabled, True,
        )
        privacy_notifications = self._resolve(
            saved, "privacy_notifications", privacy_notifications, True,
        )
        poll_seconds = self._resolve(
            saved, "poll_seconds", poll_seconds, DEFAULT_POLL_SECONDS,
        )
        transition_minutes = self._resolve(
            saved, "transition_minutes", transition_minutes, DEFAULT_TRANSITION_MINUTES,
        )
        aw_host = self._resolve(saved, "aw_host", aw_host, DEFAULT_AW_HOST)
        title_salt = self._resolve(saved, "title_salt", title_salt, DEFAULT_TITLE_SALT)
        ui_port = self._resolve(saved, "ui_port", ui_port, 8741)

        self._config.update({
            "notifications_enabled": notifications_enabled,
            "privacy_notifications": privacy_notifications,
            "poll_seconds": poll_seconds,
            "transition_minutes": transition_minutes,
            "aw_host": aw_host,
            "title_salt": title_salt,
            "ui_port": ui_port,
        })

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
        self._dev = dev
        self._browser = browser
        self._no_tray = no_tray

        self._transition_count: int = 0
        self._last_transition: dict[str, Any] | None = None
        self._labels_saved_count: int = 0
        self._model_schema_hash: str | None = None

        self._event_bus = event_bus if event_bus is not None else EventBus()

        self._suggester: _LabelSuggester | None = None
        if model_dir is not None:
            try:
                self._suggester = _LabelSuggester(model_dir)
                self._suggester._aw_host = aw_host
                self._suggester._title_salt = title_salt
                self._model_schema_hash = self._suggester._predictor._metadata.schema_hash
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
            on_initial_app=self._handle_initial_app,
            event_bus=self._event_bus,
        )
        self._icon: Any = None

    @staticmethod
    def _resolve(saved: dict[str, Any], key: str, cli_val: Any, default: Any) -> Any:
        """Return *cli_val* when it was explicitly set, else the persisted value."""
        if cli_val != default:
            return cli_val
        return saved.get(key, default)

    def _handle_initial_app(self, app: str, ts: dt.datetime) -> None:
        """Publish an initial_app event so the UI can prompt for the pre-start period."""
        if self._event_bus is not None:
            self._event_bus.publish_threadsafe({
                "type": "initial_app",
                "app": app,
                "ts": ts.isoformat(),
            })

    def _on_label_saved(self) -> None:
        """Increment the saved-label counter (called by the embedded server)."""
        self._labels_saved_count += 1

    def _on_model_trained(self, model_dir_str: str) -> None:
        """Auto-reload the model when training completes via the web UI."""
        model_path = Path(model_dir_str)
        if not model_path.is_dir():
            return
        try:
            new_suggester = _LabelSuggester(model_path)
            new_suggester._aw_host = self._aw_host
            new_suggester._title_salt = self._title_salt
            self._suggester = new_suggester
            self._model_dir = model_path
            self._model_schema_hash = new_suggester._predictor._metadata.schema_hash
            logger.info("Auto-loaded newly trained model from %s", model_path)
        except Exception:
            logger.warning("Could not auto-load trained model from %s", model_path, exc_info=True)

    def _toggle_pause(self) -> bool:
        """Toggle pause state on the monitor. Returns new paused state."""
        if self._monitor.is_paused:
            self._monitor.resume()
        else:
            self._monitor.pause()
        return self._monitor.is_paused

    def _handle_poll(self, dominant_app: str) -> None:
        self._current_app = dominant_app
        if self._event_bus is not None:
            self._event_bus.publish_threadsafe({
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
            })

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

        if self._event_bus is None or not self._event_bus.has_subscribers:
            self._send_notification(prev_app, new_app, block_start, block_end)

        if self._event_bus is not None:
            self._event_bus.publish_threadsafe({
                "type": "prompt_label",
                "prev_app": prev_app,
                "new_app": new_app,
                "block_start": block_start.isoformat(),
                "block_end": block_end.isoformat(),
                "duration_min": max(1, int((block_end - block_start).total_seconds() / 60)),
                "suggested_label": self._suggested_label,
                "suggested_confidence": self._suggested_confidence,
            })
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
                    "type": "no_model_transition",
                    "current_app": new_app,
                    "ts": block_end.isoformat(),
                    "block_start": block_start.isoformat(),
                    "block_end": block_end.isoformat(),
                })

    def _send_notification(
        self,
        prev_app: str,
        new_app: str,
        block_start: dt.datetime,
        block_end: dt.datetime,
    ) -> None:
        if not self._notifications_enabled:
            return

        title = "taskclf — Activity changed"
        duration_min = max(1, int((block_end - block_start).total_seconds() / 60))

        if self._privacy_notifications:
            message = f"Activity changed\nLabel the last {duration_min} min?"
        else:
            message = f"{prev_app} → {new_app}\nLabel the last {duration_min} min?"

        if self._suggested_label is not None and self._suggested_confidence is not None:
            message += f"\nSuggested: {self._suggested_label} ({self._suggested_confidence:.0%})"

        _send_desktop_notification(title, message, timeout=10)

    def _build_menu_items(self) -> tuple["pystray.MenuItem", ...]:
        """Return top-level menu items.

        Used as a callable by ``pystray.Menu`` so the menu is rebuilt
        (including a fresh Model submenu scan) on every right-click.
        """
        import pystray

        return (
            pystray.MenuItem(
                "Open Dashboard", self._open_dashboard, default=True,
            ),
            pystray.MenuItem(
                lambda _: "Resume" if self._monitor.is_paused else "Pause",
                self._on_pause_menu,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Label Stats", self._label_stats),
            pystray.MenuItem("Import Labels", self._import_labels),
            pystray.MenuItem("Export Labels", self._export_labels),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model", self._build_model_submenu()),
            pystray.MenuItem("Status", self._show_status),
            pystray.MenuItem("Open Data Folder", self._open_data_dir),
            pystray.MenuItem("Edit Config", self._edit_config),
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
                f"Labels exported to {csv_path.name}", csv_path,
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
            logger.info("Imported %d labels from %s (strategy=%s)", len(imported), csv_path, strategy)
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
                    "osascript", "-e",
                    'POSIX path of (choose file of type {"csv"}'
                    ' with prompt "Import Labels")',
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None, None
            csv_path = Path(result.stdout.strip())

            btn = subprocess.run(
                [
                    "osascript", "-e",
                    'button returned of (display dialog'
                    ' "Merge with existing labels?\\n\\n'
                    'Merge = keep existing, add new\\n'
                    'Overwrite = replace all labels"'
                    ' buttons {"Cancel","Overwrite","Merge"}'
                    ' default button "Merge")',
                ],
                capture_output=True, text=True, timeout=120,
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
        today_spans = [
            s for s in spans
            if s.start_ts.date() == today
            or (hasattr(s.start_ts, 'astimezone') and s.start_ts.date() == today)
        ]

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

        parts = [f"{label} {int(m)}m" for label, m in sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True,
        )]
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
        """Open config.json in the default text editor."""
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

    def _reload_model(self, *_args: Any) -> None:
        """Re-read the model bundle from disk without restarting."""
        if self._model_dir is None:
            self._notify("No model directory configured")
            return
        try:
            new_suggester = _LabelSuggester(self._model_dir)
            new_suggester._aw_host = self._aw_host
            new_suggester._title_salt = self._title_salt
            self._suggester = new_suggester
            self._model_schema_hash = new_suggester._predictor._metadata.schema_hash
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
                self._models_dir, config.global_retrain_cadence_days,
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
                    self._notify(
                        f"Model is current ({latest.name}, created {created})"
                    )
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

        bundles = (
            list_bundles(self._models_dir)
            if self._models_dir is not None
            else []
        )
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

                items.append(pystray.MenuItem(
                    bundle.model_id,
                    make_switch_cb(model_path),
                    checked=make_checked(model_path),
                ))

            items.append(pystray.MenuItem(
                "(No Model)",
                self._unload_model,
                checked=lambda _item: self._model_dir is None,
            ))
            items.append(pystray.Menu.SEPARATOR)
            items.append(pystray.MenuItem(
                "Reload Model",
                self._reload_model,
                enabled=lambda _: self._model_dir is not None,
            ))
            items.append(pystray.MenuItem(
                "Check Retrain",
                self._check_retrain,
                enabled=lambda _: self._models_dir is not None,
            ))
        else:
            items.append(pystray.MenuItem(
                "(no models found)", None, enabled=False,
            ))
            items.append(pystray.Menu.SEPARATOR)
            items.append(pystray.MenuItem(
                "Reload Model",
                self._reload_model,
                enabled=lambda _: self._model_dir is not None,
            ))
            items.append(pystray.MenuItem(
                "Check Retrain",
                self._check_retrain,
                enabled=lambda _: self._models_dir is not None,
            ))

        return pystray.Menu(*items)

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
            self._suggester = new_suggester
            self._model_dir = model_path
            self._model_schema_hash = new_suggester._predictor._metadata.schema_hash
            self._notify(f"Switched to model {model_path.name}")
            logger.info("Switched to model %s", model_path)
        except Exception as exc:
            self._notify(f"Switch failed: {exc}")
            logger.warning("Model switch to %s failed: %s", model_path, exc, exc_info=True)

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
        aw_status = "connected" if self._monitor._bucket_id is not None else "disconnected"
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

            ui_port = _VITE_DEV_PORT if (self._dev and self._vite_proc is not None
                                        and self._vite_proc.poll() is None) else self._ui_port
            webbrowser.open(f"http://127.0.0.1:{ui_port}")
            return

        if self._ui_proc is not None and self._ui_proc.poll() is None:
            logger.debug("Sending toggle to UI process (pid=%s)", self._ui_proc.pid)
            try:
                self._ui_proc.stdin.write(b"toggle\n")
                self._ui_proc.stdin.flush()
            except (BrokenPipeError, OSError):
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
            safe_msg = message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
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
                    capture_output=True, timeout=15, check=False,
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
        models_dir = (
            str(self._models_dir) if self._models_dir else str(home / "models")
        )

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

        fastapi_app = create_app(
            data_dir=self._data_dir,
            models_dir=self._models_dir,
            aw_host=self._aw_host,
            title_salt=self._title_salt,
            event_bus=self._event_bus,
            on_label_saved=self._on_label_saved,
            on_model_trained=self._on_model_trained,
            pause_toggle=self._toggle_pause,
            is_paused=lambda: self._monitor.is_paused,
        )

        uvicorn_config = uvicorn.Config(
            fastapi_app, host="127.0.0.1", port=self._ui_port,
            log_level="warning",
        )
        server = uvicorn.Server(uvicorn_config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        self._ui_server_running = True

        print(f"taskclf API on http://127.0.0.1:{self._ui_port}")

        ui_port = self._ui_port

        if self._dev:
            import taskclf.ui.server as _ui_srv

            frontend_dir = Path(_ui_srv.__file__).resolve().parent / "frontend"
            if not frontend_dir.is_dir():
                print("Warning: frontend source not found (--dev requires a repo checkout)")
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

        try:
            cmd = [
                sys.executable, "-m", "taskclf.ui.window",
                "--port", str(self._ui_port),
            ]
            self._ui_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            mode = " (dev)" if self._dev else ""
            print(f"UI window launched{mode} (pid={self._ui_proc.pid}, port={self._ui_port})")
        except Exception:
            logger.warning("Could not start UI window subprocess", exc_info=True)
            print(f"Warning: UI window failed to start. Dashboard at http://127.0.0.1:{self._ui_port}")

    def _start_ui_embedded(self) -> None:
        """Run FastAPI in-process and open the dashboard in the default browser."""
        import webbrowser

        ui_port = self._start_server()
        webbrowser.open(f"http://127.0.0.1:{ui_port}")
        mode = " (dev)" if self._dev else ""
        print(f"UI opened in browser{mode} (port={ui_port})")

    def _cleanup_ui(self) -> None:
        """Terminate UI and Vite subprocesses if still running."""
        if self._ui_proc is not None and self._ui_proc.poll() is None:
            self._ui_proc.terminate()
            try:
                self._ui_proc.wait(timeout=5)
            except Exception:
                logger.debug("UI process did not exit gracefully, killing", exc_info=True)
                self._ui_proc.kill()
        if self._vite_proc is not None and self._vite_proc.poll() is None:
            self._vite_proc.terminate()
            try:
                self._vite_proc.wait(timeout=5)
            except Exception:
                logger.debug("Vite process did not exit gracefully, killing", exc_info=True)
                self._vite_proc.kill()

    def run(self) -> None:
        """Start the tray icon and background monitor. Blocks until quit."""
        try:
            self._run_inner()
        except (SystemExit, KeyboardInterrupt):
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

        monitor_thread = threading.Thread(
            target=self._monitor.run, daemon=True,
        )
        monitor_thread.start()

        mode = "with model suggestions" if self._suggester else "label-only (no model)"

        if self._no_tray:
            print(f"taskclf running ({mode}), no tray icon.")
            print(f"UI available at http://127.0.0.1:{self._ui_port}")
            print("Press Ctrl+C to quit.")
            stop = threading.Event()
            try:
                stop.wait()
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
        print("Click the tray icon to open the dashboard. Press Ctrl+C or Quit to exit.")

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
    title_salt: str = DEFAULT_TITLE_SALT,
    data_dir: Path = Path(DEFAULT_DATA_DIR),
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES,
    event_bus: EventBus | None = None,
    ui_port: int = 8741,
    dev: bool = False,
    browser: bool = False,
    no_tray: bool = False,
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
            bundles.  Enables the "Model" submenu for hot-swapping.
        aw_host: ActivityWatch server URL.
        poll_seconds: Seconds between AW polling cycles.
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
        username: Display name to persist in ``config.json``.  Does not
            affect label identity (labels use the stable auto-generated
            UUID ``user_id``).
        notifications_enabled: When ``False``, desktop notifications
            are suppressed entirely.
        privacy_notifications: When ``True`` (the default), app names
            are redacted from desktop notifications to protect privacy.
            Set to ``False`` to show raw app identifiers.
        retrain_config: Optional path to a retrain YAML config.
            Enables the "Check Retrain" item in the Model submenu.
    """
    tray = TrayLabeler(
        data_dir=data_dir,
        model_dir=model_dir,
        models_dir=models_dir,
        aw_host=aw_host,
        title_salt=title_salt,
        poll_seconds=poll_seconds,
        transition_minutes=transition_minutes,
        event_bus=event_bus,
        ui_port=ui_port,
        dev=dev,
        browser=browser,
        no_tray=no_tray,
        username=username,
        notifications_enabled=notifications_enabled,
        privacy_notifications=privacy_notifications,
        retrain_config=retrain_config,
    )
    tray.run()
