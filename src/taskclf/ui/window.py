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

_COMPACT_SIZE = (280, 44)
_EXPANDED_SIZE = (280, 280)
_PANEL_SIZE = (280, 750)
_PANEL_HIDE_DELAY_S = 0.3


class WindowAPI:
    """Python methods exposed to JS as ``window.pywebview.api.<method>()``.

    Each method returns a value that pywebview serializes as a JSON
    Promise to the frontend.  The ``Host`` adapter in ``host.ts``
    calls these; components never reference ``window.pywebview``
    directly.
    """

    def __init__(self) -> None:
        self._window: Any = None
        self._panel_window: Any = None
        self._visible = True
        self._panel_visible = False
        self._panel_hide_timer: threading.Timer | None = None

    def bind(self, window: Any) -> None:
        self._window = window
        try:
            window.events.moved += self._on_main_window_moved
        except Exception:
            logger.debug("Could not bind moved event", exc_info=True)

    def bind_panel(self, panel: Any) -> None:
        self._panel_window = panel

    def set_compact(self) -> None:
        if self._window is not None:
            self._window.resize(*_COMPACT_SIZE)

    def set_expanded(self) -> None:
        if self._window is not None:
            self._window.resize(*_EXPANDED_SIZE)

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

    # -- State panel window ----------------------------------------------------

    def toggle_state_panel(self) -> None:
        """Toggle the panel window's visibility."""
        if self._panel_window is None or self._window is None:
            return
        if self._panel_visible:
            self._schedule_panel_hide()
        else:
            self._cancel_panel_timer()
            self._position_panel()
            try:
                self._panel_window.show()
            except Exception:
                logger.debug("Could not show panel", exc_info=True)
            self._panel_visible = True

    def _schedule_panel_hide(self) -> None:
        self._cancel_panel_timer()
        self._panel_hide_timer = threading.Timer(
            _PANEL_HIDE_DELAY_S, self._do_hide_panel,
        )
        self._panel_hide_timer.daemon = True
        self._panel_hide_timer.start()

    def _cancel_panel_timer(self) -> None:
        if self._panel_hide_timer is not None:
            self._panel_hide_timer.cancel()
            self._panel_hide_timer = None

    def _do_hide_panel(self) -> None:
        self._panel_hide_timer = None
        if self._panel_window is not None:
            try:
                self._panel_window.hide()
            except Exception:
                logger.debug("Could not hide panel", exc_info=True)
        self._panel_visible = False

    def _position_panel(self) -> None:
        """Place the panel window directly below the main window."""
        if self._panel_window is None or self._window is None:
            return
        try:
            self._panel_window.move(
                self._window.x,
                self._window.y + self._window.height + 4,
            )
        except Exception:
            logger.debug("Could not reposition panel", exc_info=True)

    def _on_main_window_moved(self) -> None:
        """Reposition the panel when the main window is dragged."""
        if self._panel_visible:
            self._position_panel()

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
    import webview

    api = window_api or WindowAPI()

    screens = webview.screens
    primary = screens[0] if screens else None
    x = (primary.width - _COMPACT_SIZE[0] - 16) if primary else None
    y = 16 if primary else None

    window = webview.create_window(
        "taskclf",
        url=f"http://127.0.0.1:{port}",
        width=_COMPACT_SIZE[0],
        height=_COMPACT_SIZE[1],
        x=x,
        y=y,
        frameless=True,
        on_top=True,
        easy_drag=True,
        resizable=False,
        js_api=api,
        background_color="#0f1117",
    )
    api.bind(window)

    panel_x = x if x is not None else 0
    panel_y = (y + _COMPACT_SIZE[1] + 4) if y is not None else 60
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
        js_api=api,
        background_color="#0d0d0d",
        hidden=True,
    )
    api.bind_panel(panel)

    def _startup(win: Any) -> None:
        if on_ready is not None:
            on_ready(win)

    webview.start(func=_startup, args=[window])
