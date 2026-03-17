"""Webview lifecycle bootstrap for the taskclf floating window.

Creates all three pywebview windows (compact pill, label grid, state
panel) and starts the GUI event loop.  Blocks on the main thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from taskclf.ui.window import (
    _COMPACT_SIZE,
    _LABEL_SIZE,
    _PANEL_SIZE,
    WindowAPI,
)

logger = logging.getLogger(__name__)


def window_run(
    port: int = 8741,
    on_ready: Callable[..., Any] | None = None,
    window_api: WindowAPI | None = None,
) -> None:
    """Create and start the pywebview floating window (blocks on main thread).

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
    window_run(port=args.port)
