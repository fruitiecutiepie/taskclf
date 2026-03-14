"""Top-level crash handler that writes a report file on unhandled exceptions.

Crash reports include environment info and sanitized log tail so the user
can attach them to GitHub issue reports without manual data gathering.
"""

from __future__ import annotations

import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

ISSUE_URL: Final[str] = (
    "https://github.com/fruitiecutiepie/taskclf/issues/new?template=bug_report.yml"
)

_LOG_FILENAME: Final[str] = "taskclf.log"


def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("taskclf")
    except Exception:
        return "unknown"


def _read_log_tail(log_dir: Path, n: int) -> list[str]:
    """Return the last *n* lines from the log file, sanitized."""
    from taskclf.core.logging import redact_message

    log_file = log_dir / _LOG_FILENAME
    if not log_file.is_file():
        return []
    try:
        raw_lines = log_file.read_text("utf-8", errors="replace").splitlines()
        tail = raw_lines[-n:] if len(raw_lines) > n else raw_lines
        return [redact_message(line) for line in tail]
    except Exception:
        return []


def write_crash_report(
    exc: BaseException,
    *,
    log_dir: Path | None = None,
    log_tail_lines: int = 20,
) -> Path:
    """Write a crash report to ``<log_dir>/crash_<YYYYMMDD_HHMMSS>.txt``.

    Args:
        exc: The unhandled exception.
        log_dir: Directory for the crash file.  Defaults to
            ``<TASKCLF_HOME>/logs``.
        log_tail_lines: Number of log-file lines to append.

    Returns:
        Absolute path to the crash report file.
    """
    if log_dir is None:
        from taskclf.core.paths import taskclf_home

        log_dir = taskclf_home() / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp_slug = now.strftime("%Y%m%d_%H%M%S")
    crash_file = log_dir / f"crash_{timestamp_slug}.txt"

    # Avoid overwriting if two crashes land in the same second
    counter = 1
    while crash_file.exists():
        crash_file = log_dir / f"crash_{timestamp_slug}_{counter}.txt"
        counter += 1

    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    version = _get_version()
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    log_tail = _read_log_tail(log_dir, log_tail_lines)

    sections = [
        "taskclf crash report",
        "=" * 40,
        f"Timestamp : {now.isoformat()}",
        f"Version   : taskclf {version}",
        f"Python    : {sys.version}",
        f"OS        : {os_info}",
        "",
        "Exception",
        "-" * 40,
        f"Type    : {type(exc).__qualname__}",
        f"Message : {exc}",
        "",
        "Traceback",
        "-" * 40,
        tb_text.rstrip(),
    ]

    if log_tail:
        sections += [
            "",
            f"Last {len(log_tail)} log lines",
            "-" * 40,
            *log_tail,
        ]

    sections += [
        "",
        "=" * 40,
        f"Please report this issue at:\n  {ISSUE_URL}",
    ]

    crash_file.write_text("\n".join(sections) + "\n", "utf-8")
    return crash_file
