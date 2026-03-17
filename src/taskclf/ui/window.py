"""Native floating window via pywebview.

Creates a frameless, always-on-top, draggable window backed by the
platform webview (WebKit on macOS, Edge WebView2 on Windows).  Exposes
a ``WindowAPI`` to the SolidJS frontend via ``window.pywebview.api``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_COMPACT_SIZE = (150, 30)
_LABEL_SIZE = (280, 330)
_PANEL_SIZE = (280, 520)
_CHILD_HIDE_DELAY_S = 0.3
_DRAG_TOLERANCE = 10  # pixels — ignore drift below this when detecting user drags


class WindowAPI:
    """Python methods exposed to JS as ``window.pywebview.api.<method>()``.

    Each method returns a value that pywebview serializes as a JSON
    Promise to the frontend.  The ``Host`` adapter in ``host.ts``
    calls these; components never reference ``window.pywebview``
    directly.
    """

    def __init__(self) -> None:
        self._window: Any = None
        self._label_window: Any = None
        self._panel_window: Any = None
        self._visible = True
        self._label_visible = False
        self._label_pinned = False
        self._panel_visible = False
        self._panel_pinned = False
        self._label_hide_timer: threading.Timer | None = None
        self._panel_hide_timer: threading.Timer | None = None
        self._label_expected_pos: tuple[int, int] | None = None
        self._panel_expected_pos: tuple[int, int] | None = None
        self._default_x: int | None = None
        self._default_y: int | None = None

    def bind(self, window: Any) -> None:
        self._window = window
        try:
            window.events.moved += self._on_main_window_moved
        except Exception:
            logger.debug("Could not bind moved event", exc_info=True)

    def bind_label(self, label: Any) -> None:
        self._label_window = label

    def bind_panel(self, panel: Any) -> None:
        self._panel_window = panel

    def hide_window(self) -> None:
        if self._window is not None:
            self._window.hide()
        self._visible = False

    def show_window(self) -> None:
        if self._window is not None:
            self._window.show()
        self._visible = True

    def toggle_window(self) -> None:
        if self._visible:
            self.hide_window()
        else:
            self.show_window()

    def toggle_dashboard(self) -> None:
        """Toggle all windows. Re-show positions the pill at its default location."""
        logger.debug("toggle_dashboard called — visible=%s", self._visible)
        if self._visible:
            if self._label_visible:
                self._do_hide_label()
            if self._panel_visible:
                self._do_hide_panel()
            self.hide_window()
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
            self.show_window()

    # -- Label grid window -----------------------------------------------------

    def show_label_grid(self) -> None:
        """Show label grid on hover (non-pinned)."""
        if self._label_window is None or self._window is None:
            return
        self._cancel_timer("_label_hide_timer")
        if not self._label_visible:
            self._label_visible = True
            self._reposition_label()
            try:
                self._label_window.show()
            except Exception:
                logger.debug("Could not show label grid", exc_info=True)

    def hide_label_grid(self) -> None:
        """Schedule label grid hide unless pinned."""
        if self._label_pinned:
            return
        self._schedule_hide("_label_hide_timer", self._do_hide_label)

    def cancel_label_hide(self) -> None:
        """Cancel any pending label hide (e.g. mouse entered label window)."""
        self._cancel_timer("_label_hide_timer")

    def toggle_label_grid(self) -> None:
        """Toggle pinned state of the label grid."""
        if self._label_window is None or self._window is None:
            return
        if self._label_visible and self._label_pinned:
            self._label_pinned = False
            self._do_hide_label()
        elif self._label_visible and not self._label_pinned:
            self._label_pinned = True
        else:
            self._cancel_timer("_label_hide_timer")
            self._label_pinned = True
            self._label_visible = True
            self._reposition_label()
            try:
                self._label_window.show()
            except Exception:
                logger.debug("Could not show label grid", exc_info=True)

    def _do_hide_label(self) -> None:
        self._label_hide_timer = None
        if self._label_window is not None:
            try:
                self._label_window.hide()
            except Exception:
                logger.debug("Could not hide label grid", exc_info=True)
        self._label_visible = False
        self._label_pinned = False
        self._label_expected_pos = None

    # -- State panel window ----------------------------------------------------

    def show_state_panel(self) -> None:
        """Show panel on hover (non-pinned)."""
        if self._panel_window is None or self._window is None:
            return
        self._cancel_timer("_panel_hide_timer")
        if not self._panel_visible:
            self._panel_visible = True
            self._position_panel()
            try:
                self._panel_window.show()
            except Exception:
                logger.debug("Could not show panel", exc_info=True)

    def hide_state_panel(self) -> None:
        """Schedule panel hide unless pinned."""
        if self._panel_pinned:
            return
        self._schedule_hide("_panel_hide_timer", self._do_hide_panel)

    def cancel_panel_hide(self) -> None:
        """Cancel any pending panel hide (e.g. mouse entered panel window)."""
        self._cancel_timer("_panel_hide_timer")

    def toggle_state_panel(self) -> None:
        """Toggle pinned state of the panel."""
        if self._panel_window is None or self._window is None:
            return
        if self._panel_visible and self._panel_pinned:
            self._panel_pinned = False
            self._do_hide_panel()
        elif self._panel_visible and not self._panel_pinned:
            self._panel_pinned = True
        else:
            self._cancel_timer("_panel_hide_timer")
            self._panel_pinned = True
            self._panel_visible = True
            self._position_panel()
            try:
                self._panel_window.show()
            except Exception:
                logger.debug("Could not show panel", exc_info=True)

    def _do_hide_panel(self) -> None:
        self._panel_hide_timer = None
        if self._panel_window is not None:
            try:
                self._panel_window.hide()
            except Exception:
                logger.debug("Could not hide panel", exc_info=True)
        self._panel_visible = False
        self._panel_pinned = False
        self._panel_expected_pos = None

    # -- Shared helpers --------------------------------------------------------

    def _schedule_hide(
        self,
        timer_attr: str,
        callback: Callable[[], None],
    ) -> None:
        self._cancel_timer(timer_attr)
        timer = threading.Timer(_CHILD_HIDE_DELAY_S, callback)
        timer.daemon = True
        timer.start()
        setattr(self, timer_attr, timer)

    def _cancel_timer(self, timer_attr: str) -> None:
        timer = getattr(self, timer_attr, None)
        if timer is not None:
            timer.cancel()
            setattr(self, timer_attr, None)

    def _on_main_window_moved(self) -> None:
        """Reposition child windows, unless the user has dragged them away."""
        label_dragged = False
        if (
            self._label_visible
            and self._label_window is not None
            and self._label_expected_pos is not None
        ):
            try:
                cx, cy = self._label_window.x, self._label_window.y
                ex, ey = self._label_expected_pos
                if abs(cx - ex) > _DRAG_TOLERANCE or abs(cy - ey) > _DRAG_TOLERANCE:
                    label_dragged = True
            except Exception:
                pass
        if not label_dragged:
            self._reposition_label()

        if (
            self._panel_visible
            and self._panel_window is not None
            and self._panel_expected_pos is not None
        ):
            try:
                cx, cy = self._panel_window.x, self._panel_window.y
                ex, ey = self._panel_expected_pos
                if abs(cx - ex) > _DRAG_TOLERANCE or abs(cy - ey) > _DRAG_TOLERANCE:
                    return
            except Exception:
                pass
        self._position_panel()

    def _reposition_label(self) -> None:
        """Place label grid below pill, right-aligned."""
        if self._window is None:
            return
        try:
            if self._label_visible and self._label_window is not None:
                new_x = self._window.x + _COMPACT_SIZE[0] - _LABEL_SIZE[0]
                new_y = self._window.y + _COMPACT_SIZE[1] + 4
                self._label_window.move(new_x, new_y)
                self._label_expected_pos = (new_x, new_y)
        except Exception:
            logger.debug("Could not reposition label window", exc_info=True)

    def _position_panel(self) -> None:
        """Place panel below pill (and below label grid if visible)."""
        if self._window is None or self._panel_window is None:
            return
        try:
            right_x = self._window.x + _COMPACT_SIZE[0]
            y = self._window.y + _COMPACT_SIZE[1] + 4
            if self._label_visible:
                y += _LABEL_SIZE[1] + 4
            new_x = right_x - _PANEL_SIZE[0]
            self._panel_window.move(new_x, y)
            self._panel_expected_pos = (new_x, y)
        except Exception:
            logger.debug("Could not position panel window", exc_info=True)

    @property
    def visible(self) -> bool:
        return self._visible


def run_window(
    port: int = 8741,
    on_ready: Callable[..., Any] | None = None,
    window_api: WindowAPI | None = None,
) -> None:
    """Create and run the pywebview floating window (blocks on main thread).

    Args:
        port: Port of the FastAPI server to load in the webview.
        on_ready: Optional callback invoked after the GUI loop starts.
            Receives the window object as its first argument.
        window_api: Shared ``WindowAPI`` instance.  When ``None`` a new
            one is created.
    """
    import os
    import sys

    import webview

    api = window_api or WindowAPI()

    screens = webview.screens
    primary = screens[0] if screens else None
    x = (primary.width - _COMPACT_SIZE[0] - 16) if primary else None
    y = 16 if primary else None

    api._default_x = x
    api._default_y = y

    window = webview.create_window(
        "taskclf",
        url=f"http://127.0.0.1:{port}",
        width=_COMPACT_SIZE[0],
        height=_COMPACT_SIZE[1],
        x=x,
        y=y,
        frameless=True,
        on_top=True,
        easy_drag=False,
        resizable=False,
        transparent=True,
        js_api=api,
    )
    api.bind(window)

    label_x = (x + _COMPACT_SIZE[0] - _LABEL_SIZE[0]) if x is not None else 0
    label_y = (y + _COMPACT_SIZE[1] + 4) if y is not None else 50
    label = webview.create_window(
        "taskclf-label",
        url=f"http://127.0.0.1:{port}?view=label",
        width=_LABEL_SIZE[0],
        height=_LABEL_SIZE[1],
        x=label_x,
        y=label_y,
        frameless=True,
        on_top=True,
        easy_drag=False,
        resizable=False,
        transparent=True,
        js_api=api,
        hidden=True,
    )
    api.bind_label(label)

    panel_x = (x + _COMPACT_SIZE[0] - _PANEL_SIZE[0]) if x is not None else 0
    panel_y = (y + _COMPACT_SIZE[1] + 4) if y is not None else 50
    panel = webview.create_window(
        "taskclf-panel",
        url=f"http://127.0.0.1:{port}?view=panel",
        width=_PANEL_SIZE[0],
        height=_PANEL_SIZE[1],
        x=panel_x,
        y=panel_y,
        frameless=True,
        on_top=True,
        easy_drag=False,
        resizable=False,
        transparent=True,
        js_api=api,
        hidden=True,
    )
    api.bind_panel(panel)

    def _stdin_reader() -> None:
        """Read commands from stdin (sent by the tray process)."""
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    logger.debug("stdin EOF — reader exiting")
                    break
                cmd = line.strip()
                logger.debug("stdin command received: %r", cmd)
                if cmd == "toggle":
                    api.toggle_dashboard()
        except Exception:
            logger.debug("stdin reader error", exc_info=True)

    stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
    stdin_thread.start()

    saved_stderr_fd: int | None = None
    if sys.platform == "darwin":
        try:
            saved_stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            logger.debug("Could not redirect stderr for pywebview", exc_info=True)
            saved_stderr_fd = None

    def _startup(win: Any) -> None:
        nonlocal saved_stderr_fd
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            except OSError:
                logger.debug(
                    "Could not restore stderr after pywebview init", exc_info=True
                )
            saved_stderr_fd = None
        if on_ready is not None:
            on_ready(win)

    webview.start(func=_startup, args=[window])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch pywebview window for taskclf")
    parser.add_argument(
        "--port", type=int, default=8741, help="Port of the FastAPI server"
    )
    args = parser.parse_args()
    run_window(port=args.port)
