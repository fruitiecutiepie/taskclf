"""WindowAPI and WindowChild for pywebview-based floating UI.

``WindowAPI`` is exposed to the SolidJS frontend via
``window.pywebview.api``.  ``WindowChild`` encapsulates the
visibility / pin / timer state machine shared by the label-grid
and state-panel child windows.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_COMPACT_SIZE = (150, 30)
_LABEL_SIZE = (280, 330)
_PANEL_SIZE = (280, 520)
_CHILD_HIDE_DELAY_S = 0.3
_DRAG_TOLERANCE = 10
_TRANSITION_NOTIFICATION_TITLE = "taskclf — Activity changed"
_AGENT_DEBUG_LOG_PATH = (
    "/Users/audreysantoso/github/fruitiecutiepie/taskclf/.cursor/debug-f37ed4.log"
)


# region agent log
def _agent_debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "sessionId": "f37ed4",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
        pass


# endregion


def _iso_dt_parse(value: Any) -> dt.datetime | None:
    """Parse an ISO datetime string, accepting ``Z`` suffixes."""
    if not isinstance(value, str) or not value:
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


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


def _transition_notification_body(prompt: Any) -> str:
    """Build a privacy-safe transition notification body from a prompt payload."""
    if not isinstance(prompt, dict):
        return "Activity changed"

    summary = prompt.get("suggestion_text")
    if not isinstance(summary, str) or not summary.strip():
        summary = "Activity changed"

    start = _iso_dt_parse(prompt.get("block_start"))
    end = _iso_dt_parse(prompt.get("block_end"))
    if start is None or end is None:
        return summary
    return f"{summary}\n{_display_time_range_exact_local(start, end)}"


def _send_desktop_notification(title: str, message: str, timeout: int = 10) -> None:
    """Best-effort native desktop notification for the pywebview shell."""
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


@dataclass(eq=False)
class WindowChild:
    """Visibility, pinning, and delayed-hide for an anchored child window."""

    name: str
    position_fn: Callable[[WindowChild], None]
    window: Any = field(init=False, default=None)
    visible: bool = field(init=False, default=False)
    pinned: bool = field(init=False, default=False)
    hide_timer: threading.Timer | None = field(init=False, default=None)
    expected_pos: tuple[int, int] | None = field(init=False, default=None)

    def visibility_on(self, main: Any) -> None:
        """Show on hover (non-pinned)."""
        if self.window is None or main is None:
            return
        self.timer_cancel()
        if not self.visible:
            self.visible = True
            self.position_sync()
            try:
                self.window.show()
            except Exception:
                logger.debug("Could not show %s", self.name, exc_info=True)

    def visibility_off_deferred(self) -> None:
        """Schedule hide unless pinned."""
        if self.pinned:
            return
        self.timer_cancel()
        timer = threading.Timer(_CHILD_HIDE_DELAY_S, self.visibility_off)
        timer.daemon = True
        timer.start()
        self.hide_timer = timer

    def visibility_off(self) -> None:
        """Immediate hide — clears visible, pinned, and expected_pos."""
        self.hide_timer = None
        if self.window is not None:
            try:
                self.window.hide()
            except Exception:
                logger.debug("Could not hide %s", self.name, exc_info=True)
        self.visible = False
        self.pinned = False
        self.expected_pos = None

    def pin_toggle(self, main: Any) -> None:
        """Toggle pinned state."""
        if self.window is None or main is None:
            return
        if self.visible and self.pinned:
            self.pinned = False
            self.visibility_off()
        elif self.visible and not self.pinned:
            self.pinned = True
        else:
            self.timer_cancel()
            self.pinned = True
            self.visible = True
            self.position_sync()
            try:
                self.window.show()
            except Exception:
                logger.debug("Could not show %s", self.name, exc_info=True)

    def timer_cancel(self) -> None:
        """Cancel any pending hide timer."""
        if self.hide_timer is not None:
            self.hide_timer.cancel()
            self.hide_timer = None

    def position_sync(self) -> None:
        """Reposition via the injected layout callback."""
        self.position_fn(self)

    def drag_detected(self) -> bool:
        """True if the user has dragged this window away from expected position."""
        if not self.visible or self.window is None or self.expected_pos is None:
            return False
        try:
            cx, cy = self.window.x, self.window.y
            ex, ey = self.expected_pos
            return abs(cx - ex) > _DRAG_TOLERANCE or abs(cy - ey) > _DRAG_TOLERANCE
        except Exception:
            return False


@dataclass(eq=False)
class WindowAPI:
    """Python methods exposed to JS as ``window.pywebview.api.<method>()``.

    Each method returns a value that pywebview serializes as a JSON
    Promise to the frontend.  The ``Host`` adapter in ``host.ts``
    calls these; components never reference ``window.pywebview``
    directly.
    """

    _window: Any = field(init=False, default=None)
    _visible: bool = field(init=False, default=True)
    _default_x: int | None = field(init=False, default=None)
    _default_y: int | None = field(init=False, default=None)
    _label: WindowChild = field(init=False)
    _panel: WindowChild = field(init=False)

    def __post_init__(self) -> None:
        self._label = WindowChild("label", self._label_position)
        self._panel = WindowChild("panel", self._panel_position)

    def bind(self, window: Any) -> None:
        self._window = window
        try:
            window.events.moved += self._on_main_window_moved
        except Exception:
            logger.debug("Could not bind moved event", exc_info=True)

    def bind_label(self, label: Any) -> None:
        self._label.window = label

    def bind_panel(self, panel: Any) -> None:
        self._panel.window = panel

    def window_hide(self) -> None:
        if self._window is not None:
            self._window.hide()
        self._visible = False

    def window_show(self) -> None:
        if self._window is not None:
            self._window.show()
        self._visible = True

    def window_toggle(self) -> None:
        if self._visible:
            self.window_hide()
        else:
            self.window_show()

    def dashboard_toggle(self) -> None:
        """Toggle all windows. Re-show positions the pill at its default location."""
        logger.debug("dashboard_toggle called — visible=%s", self._visible)
        if self._visible:
            if self._label.visible:
                self._label.visibility_off()
            if self._panel.visible:
                self._panel.visibility_off()
            self.window_hide()
        else:
            if (
                self._window is not None
                and self._default_x is not None
                and self._default_y is not None
            ):
                try:
                    self._window.move(self._default_x, self._default_y)
                except Exception:
                    logger.debug(
                        "Could not reposition window to default", exc_info=True
                    )
            self.window_show()

    # -- Label grid window -----------------------------------------------------

    def label_grid_show(self) -> None:
        """Show label grid on hover (non-pinned)."""
        self._label.visibility_on(self._window)

    def label_grid_hide(self) -> None:
        """Schedule label grid hide unless pinned."""
        self._label.visibility_off_deferred()

    def label_grid_cancel_hide(self) -> None:
        """Cancel any pending label hide (e.g. mouse entered label window)."""
        self._label.timer_cancel()

    def label_grid_toggle(self) -> None:
        """Toggle pinned state of the label grid."""
        self._label.pin_toggle(self._window)

    def show_transition_notification(self, prompt: dict[str, Any]) -> None:
        """Show a native desktop notification for a transition prompt."""
        # region agent log
        _agent_debug_log(
            "pre-fix",
            "H4",
            "src/taskclf/ui/window.py:291",
            "pywebview api show_transition_notification entered",
            {
                "block_start": prompt.get("block_start"),
                "block_end": prompt.get("block_end"),
                "suggested_label": prompt.get("suggested_label"),
                "suggestion_text_present": prompt.get("suggestion_text") is not None,
            },
        )
        # endregion
        _send_desktop_notification(
            _TRANSITION_NOTIFICATION_TITLE,
            _transition_notification_body(prompt),
            timeout=10,
        )

    # -- State panel window ----------------------------------------------------

    def state_panel_show(self) -> None:
        """Show panel on hover (non-pinned)."""
        self._panel.visibility_on(self._window)

    def state_panel_hide(self) -> None:
        """Schedule panel hide unless pinned."""
        self._panel.visibility_off_deferred()

    def state_panel_cancel_hide(self) -> None:
        """Cancel any pending panel hide (e.g. mouse entered panel window)."""
        self._panel.timer_cancel()

    def state_panel_toggle(self) -> None:
        """Toggle pinned state of the panel."""
        self._panel.pin_toggle(self._window)

    def frontend_debug_log(self, message: str) -> None:
        """Accept debug log lines from the frontend webview."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug("[frontend] %s", message)

    def frontend_error_log(self, message: str) -> None:
        """Accept error log lines from the frontend webview."""
        logger.error("[frontend] %s", message)

    # -- Positioning -----------------------------------------------------------

    def _label_position(self, child: WindowChild) -> None:
        """Place label grid below pill, right-aligned."""
        if self._window is None:
            return
        try:
            if child.visible and child.window is not None:
                new_x = self._window.x + _COMPACT_SIZE[0] - _LABEL_SIZE[0]
                new_y = self._window.y + _COMPACT_SIZE[1] + 4
                child.window.move(new_x, new_y)
                child.expected_pos = (new_x, new_y)
        except Exception:
            logger.debug("Could not reposition label window", exc_info=True)

    def _panel_position(self, child: WindowChild) -> None:
        """Place panel below pill (and below label grid if visible)."""
        if self._window is None or child.window is None:
            return
        try:
            right_x = self._window.x + _COMPACT_SIZE[0]
            y = self._window.y + _COMPACT_SIZE[1] + 4
            if self._label.visible:
                y += _LABEL_SIZE[1] + 4
            new_x = right_x - _PANEL_SIZE[0]
            child.window.move(new_x, y)
            child.expected_pos = (new_x, y)
        except Exception:
            logger.debug("Could not position panel window", exc_info=True)

    def _on_main_window_moved(self) -> None:
        """Reposition child windows, unless the user has dragged them away."""
        if not self._label.drag_detected():
            self._label_position(self._label)
        if not self._panel.drag_detected():
            self._panel_position(self._panel)

    @property
    def visible(self) -> bool:
        return self._visible
