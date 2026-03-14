"""Tests for the top-level crash handler.

Covers TC-CRASH-001 through TC-CRASH-008: crash report creation, contents,
log tail inclusion, privacy, and collision avoidance.
"""

from __future__ import annotations

import platform
import re
import sys
from pathlib import Path


from taskclf.core.crash import ISSUE_URL, write_crash_report


def _make_exception() -> ValueError:
    """Raise and return an exception so it has a real traceback."""
    try:
        raise ValueError("something went wrong")
    except ValueError as exc:
        return exc


class TestWriteCrashReport:
    """TC-CRASH: write_crash_report produces a well-formed crash file."""

    def test_creates_file_in_log_dir(self, tmp_path: Path) -> None:
        """TC-CRASH-001: crash file is created in the specified directory."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)

        assert path.exists()
        assert path.parent == tmp_path
        assert path.suffix == ".txt"

    def test_filename_format(self, tmp_path: Path) -> None:
        """TC-CRASH-002: filename follows crash_YYYYMMDD_HHMMSS pattern."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)

        assert re.match(r"crash_\d{8}_\d{6}\.txt$", path.name)

    def test_contains_timestamp(self, tmp_path: Path) -> None:
        """TC-CRASH-003: report contains ISO-format timestamp."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert "Timestamp" in contents
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", contents)

    def test_contains_exception_details(self, tmp_path: Path) -> None:
        """TC-CRASH-004: report contains exception type, message, and traceback."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert "ValueError" in contents
        assert "something went wrong" in contents
        assert "Traceback" in contents
        assert "_make_exception" in contents

    def test_contains_version_info(self, tmp_path: Path) -> None:
        """TC-CRASH-005: report contains taskclf version, Python version, and OS."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert "Version" in contents
        assert sys.version in contents
        assert platform.system() in contents
        assert platform.release() in contents
        assert platform.machine() in contents

    def test_contains_issue_url(self, tmp_path: Path) -> None:
        """TC-CRASH-005b: report contains the issue URL."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert ISSUE_URL in contents

    def test_includes_log_tail_when_log_exists(self, tmp_path: Path) -> None:
        """TC-CRASH-006: when a log file exists, its tail is appended."""
        log_file = tmp_path / "taskclf.log"
        log_lines = [
            f"2026-03-01 10:00:{i:02d} INFO test — line {i}" for i in range(30)
        ]
        log_file.write_text("\n".join(log_lines) + "\n", "utf-8")

        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path, log_tail_lines=5)
        contents = path.read_text("utf-8")

        assert "Last 5 log lines" in contents
        assert "line 25" in contents
        assert "line 29" in contents
        assert "line 10" not in contents

    def test_missing_log_file_handled_gracefully(self, tmp_path: Path) -> None:
        """TC-CRASH-007: missing log file does not prevent crash report."""
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert "Last" not in contents or "log lines" not in contents
        assert "ValueError" in contents

    def test_empty_log_file_handled_gracefully(self, tmp_path: Path) -> None:
        """TC-CRASH-007b: empty log file does not prevent crash report."""
        log_file = tmp_path / "taskclf.log"
        log_file.write_text("", "utf-8")

        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path)
        contents = path.read_text("utf-8")

        assert "ValueError" in contents

    def test_log_tail_is_sanitized(self, tmp_path: Path) -> None:
        """TC-CRASH-008: sensitive fields in log tail are redacted."""
        log_file = tmp_path / "taskclf.log"
        log_file.write_text(
            'window_title="My Secret Document"\n'
            'raw_keystrokes="password123"\n'
            "safe line here\n",
            "utf-8",
        )

        exc = _make_exception()
        path = write_crash_report(exc, log_dir=tmp_path, log_tail_lines=10)
        contents = path.read_text("utf-8")

        assert "My Secret Document" not in contents
        assert "password123" not in contents
        assert "[REDACTED]" in contents
        assert "safe line here" in contents

    def test_no_overwrite_on_collision(self, tmp_path: Path) -> None:
        """TC-CRASH-009: second crash in same second gets a suffixed filename."""
        exc = _make_exception()
        path1 = write_crash_report(exc, log_dir=tmp_path)
        path1_contents = path1.read_text("utf-8")

        path2 = write_crash_report(exc, log_dir=tmp_path)

        assert path1.exists()
        assert path2.exists()
        assert path1 != path2
        assert path1.read_text("utf-8") == path1_contents

    def test_creates_log_dir_if_missing(self, tmp_path: Path) -> None:
        """TC-CRASH-010: log_dir is created if it does not exist."""
        nested = tmp_path / "a" / "b" / "c"
        exc = _make_exception()
        path = write_crash_report(exc, log_dir=nested)

        assert nested.is_dir()
        assert path.exists()
