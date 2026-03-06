"""Platform-aware base directory resolution for taskclf.

Provides :func:`taskclf_home` which resolves the root data directory
using the following precedence:

1. ``TASKCLF_HOME`` environment variable (if set)
2. ``~/Library/Application Support/taskclf/`` on macOS
3. ``$XDG_DATA_HOME/taskclf/`` (or ``~/.local/share/taskclf/``) on Linux
4. ``%LOCALAPPDATA%/taskclf/`` on Windows

All ``DEFAULT_*`` path constants in :mod:`taskclf.core.defaults` are
derived from the value returned by this function.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "TASKCLF_HOME"

_SUBDIRS: tuple[str, ...] = (
    "data/raw/aw",
    "data/processed",
    "models",
    "artifacts/telemetry",
    "configs",
)


def taskclf_home() -> Path:
    """Return the resolved base directory for all taskclf data.

    The result is always an absolute :class:`~pathlib.Path`.
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "taskclf"

    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "taskclf"
        return Path.home() / "AppData" / "Local" / "taskclf"

    # Linux / BSDs / other POSIX
    xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(xdg) / "taskclf"


def ensure_taskclf_dirs() -> Path:
    """Create the standard subdirectory tree under :func:`taskclf_home`.

    Safe to call repeatedly — existing directories are left untouched.
    Returns the home path.
    """
    home = taskclf_home()
    for sub in _SUBDIRS:
        d = home / sub
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", d)
    return home
