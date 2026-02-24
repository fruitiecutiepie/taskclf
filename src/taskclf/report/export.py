"""Report export utilities: JSON, CSV, and Parquet output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from taskclf.report.daily import DailyReport

_SENSITIVE_KEYS = frozenset({
    "raw_keystrokes",
    "window_title_raw",
    "clipboard_content",
    "clipboard",
})


def _check_no_sensitive_fields(data: dict) -> None:
    """Recursively check *data* for forbidden keys."""
    for key in data:
        if key in _SENSITIVE_KEYS:
            raise ValueError(
                f"Sensitive field {key!r} must not appear in report output"
            )
        if isinstance(data[key], dict):
            _check_no_sensitive_fields(data[key])


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
    data = report.model_dump(exclude_none=True)
    _check_no_sensitive_fields(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _breakdown_to_rows(
    report: DailyReport,
) -> list[dict[str, object]]:
    """Flatten core and mapped breakdowns into tabular rows."""
    rows: list[dict[str, object]] = []
    for label, minutes in sorted(report.core_breakdown.items()):
        rows.append({
            "date": report.date,
            "label_type": "core",
            "label": label,
            "minutes": round(minutes, 2),
        })
    if report.mapped_breakdown is not None:
        for label, minutes in sorted(report.mapped_breakdown.items()):
            rows.append({
                "date": report.date,
                "label_type": "mapped",
                "label": label,
                "minutes": round(minutes, 2),
            })
    return rows


def export_report_csv(report: DailyReport, path: Path) -> Path:
    """Write *report* as a flat CSV with one row per label.

    Columns: ``date``, ``label_type`` (``core`` or ``mapped``),
    ``label``, ``minutes``.

    Args:
        report: A ``DailyReport`` instance.
        path: Destination CSV file path.

    Returns:
        The *path* that was written.

    Raises:
        ValueError: If the report contains sensitive fields.
    """
    _check_no_sensitive_fields(report.model_dump(exclude_none=True))

    rows = _breakdown_to_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "label_type", "label", "minutes"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_report_parquet(report: DailyReport, path: Path) -> Path:
    """Write *report* as a Parquet file with one row per label.

    Schema matches :func:`export_report_csv`.

    Args:
        report: A ``DailyReport`` instance.
        path: Destination Parquet file path.

    Returns:
        The *path* that was written.

    Raises:
        ValueError: If the report contains sensitive fields.
    """
    _check_no_sensitive_fields(report.model_dump(exclude_none=True))

    rows = _breakdown_to_rows(report)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
