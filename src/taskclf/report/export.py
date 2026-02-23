"""Report export utilities: JSON and CSV output."""

from __future__ import annotations

import json
from pathlib import Path

from taskclf.report.daily import DailyReport

_SENSITIVE_KEYS = frozenset({
    "raw_keystrokes",
    "window_title_raw",
    "clipboard_content",
    "clipboard",
})


def export_report_json(report: DailyReport, path: Path) -> Path:
    """Write *report* to a JSON file, rejecting sensitive keys.

    Args:
        report: A ``DailyReport`` instance to serialize.
        path: Destination JSON file path.

    Returns:
        The *path* that was written.

    Raises:
        ValueError: If the serialized output contains any key from
            the sensitive-fields blocklist.
    """
    data = report.model_dump()

    _check_no_sensitive_fields(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _check_no_sensitive_fields(data: dict) -> None:
    """Recursively check *data* for forbidden keys."""
    for key in data:
        if key in _SENSITIVE_KEYS:
            raise ValueError(
                f"Sensitive field {key!r} must not appear in report output"
            )
        if isinstance(data[key], dict):
            _check_no_sensitive_fields(data[key])
