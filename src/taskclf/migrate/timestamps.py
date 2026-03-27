"""Audit and rewrite on-disk artifacts for naive-to-aware-UTC migration.

Provides functions to:

- **audit** parquet and JSON files for naive (tz-unaware) datetime columns
- **rewrite** those files so every datetime value is timezone-aware UTC

All rewrite operations are idempotent: running them on already-migrated
files is a no-op that returns ``already_migrated=True``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from taskclf.core.store import read_parquet, write_parquet
from taskclf.core.time import ts_utc_aware_get

_KNOWN_DATETIME_FIELDS: frozenset[str] = frozenset(
    {
        "start_ts",
        "end_ts",
        "bucket_start_ts",
        "bucket_end_ts",
        "created_at",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnAudit:
    """Audit result for a single datetime column."""

    column: str
    total: int
    naive_count: int
    aware_count: int


@dataclass
class FileAudit:
    """Audit result for a single file."""

    path: Path
    file_type: Literal["parquet", "json"]
    columns: list[ColumnAudit] = field(default_factory=list)

    @property
    def needs_migration(self) -> bool:
        return any(c.naive_count > 0 for c in self.columns)


@dataclass
class RewriteResult:
    """Result of a single file rewrite operation."""

    path: Path
    backup_path: Path | None
    columns_migrated: list[str]
    rows_affected: int
    already_migrated: bool


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names whose dtype is datetime64 (any variant)."""
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]


def _is_naive_series(s: pd.Series) -> bool:
    """True if *s* is a datetime series without timezone info."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        return False
    return s.dt.tz is None


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------


def audit_parquet_timestamps(path: Path) -> FileAudit:
    """Inspect a parquet file for naive datetime columns.

    Args:
        path: Path to an existing ``.parquet`` file.

    Returns:
        A :class:`FileAudit` describing each datetime column and how
        many values are naive vs aware.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = read_parquet(path)
    audit = FileAudit(path=path, file_type="parquet")

    for col in _datetime_columns(df):
        total = len(df[col].dropna())
        if _is_naive_series(df[col]):
            audit.columns.append(
                ColumnAudit(column=col, total=total, naive_count=total, aware_count=0)
            )
        else:
            audit.columns.append(
                ColumnAudit(column=col, total=total, naive_count=0, aware_count=total)
            )

    return audit


def audit_json_timestamps(path: Path) -> FileAudit:
    """Inspect a JSON array file for naive datetime strings.

    Checks only fields in :data:`_KNOWN_DATETIME_FIELDS`.

    Args:
        path: Path to an existing JSON file containing an array of objects.

    Returns:
        A :class:`FileAudit` describing each datetime field.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return FileAudit(path=path, file_type="json")

    present_fields: set[str] = set()
    for record in raw:
        if isinstance(record, dict):
            present_fields.update(record.keys())

    audit = FileAudit(path=path, file_type="json")

    for fld in sorted(present_fields & _KNOWN_DATETIME_FIELDS):
        naive_count = 0
        aware_count = 0
        for record in raw:
            if not isinstance(record, dict):
                continue
            val = record.get(fld)
            if val is None:
                continue
            if _is_naive_iso(str(val)):
                naive_count += 1
            else:
                aware_count += 1
        if naive_count + aware_count > 0:
            audit.columns.append(
                ColumnAudit(
                    column=fld,
                    total=naive_count + aware_count,
                    naive_count=naive_count,
                    aware_count=aware_count,
                )
            )

    return audit


def audit_data_dir(data_dir: Path) -> list[FileAudit]:
    """Walk *data_dir* and audit every parquet and queue JSON file.

    Args:
        data_dir: Root directory to scan (e.g. ``data/processed``).

    Returns:
        List of :class:`FileAudit` results, one per audited file.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    results: list[FileAudit] = []

    for pq in sorted(data_dir.rglob("*.parquet")):
        results.append(audit_parquet_timestamps(pq))

    for js in sorted(data_dir.rglob("queue.json")):
        results.append(audit_json_timestamps(js))

    return results


# ---------------------------------------------------------------------------
# Rewrite functions
# ---------------------------------------------------------------------------


def rewrite_parquet_timestamps(
    path: Path,
    *,
    backup: bool = True,
) -> RewriteResult:
    """Normalize naive datetime columns in a parquet file to aware UTC.

    Idempotent: if all datetime columns are already tz-aware, the file
    is not rewritten and ``already_migrated`` is ``True``.

    Args:
        path: Path to the parquet file.
        backup: If ``True`` (default), copy the original to ``<path>.bak``
            before overwriting.

    Returns:
        A :class:`RewriteResult` describing what was changed.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = read_parquet(path)
    naive_cols = [col for col in _datetime_columns(df) if _is_naive_series(df[col])]

    if not naive_cols:
        return RewriteResult(
            path=path,
            backup_path=None,
            columns_migrated=[],
            rows_affected=0,
            already_migrated=True,
        )

    backup_path = None
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    rows_affected = 0
    for col in naive_cols:
        non_null = df[col].notna().sum()
        rows_affected = max(rows_affected, int(non_null))
        df[col] = df[col].dt.tz_localize("UTC")

    write_parquet(df, path)

    return RewriteResult(
        path=path,
        backup_path=backup_path,
        columns_migrated=naive_cols,
        rows_affected=rows_affected,
        already_migrated=False,
    )


def rewrite_json_timestamps(
    path: Path,
    *,
    backup: bool = True,
) -> RewriteResult:
    """Normalize naive datetime strings in a JSON array file to aware UTC.

    Only fields in :data:`_KNOWN_DATETIME_FIELDS` are touched.
    Idempotent: if all datetime strings already carry timezone offsets,
    the file is not rewritten and ``already_migrated`` is ``True``.

    Args:
        path: Path to the JSON file.
        backup: If ``True`` (default), copy the original to ``<path>.bak``
            before overwriting.

    Returns:
        A :class:`RewriteResult` describing what was changed.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return RewriteResult(
            path=path,
            backup_path=None,
            columns_migrated=[],
            rows_affected=0,
            already_migrated=True,
        )

    columns_migrated: set[str] = set()
    rows_affected = 0
    changed = False

    for record in raw:
        if not isinstance(record, dict):
            continue
        row_changed = False
        for fld in _KNOWN_DATETIME_FIELDS:
            val = record.get(fld)
            if val is None:
                continue
            val_str = str(val)
            if _is_naive_iso(val_str):
                parsed = datetime.fromisoformat(val_str)
                aware = ts_utc_aware_get(parsed)
                record[fld] = aware.isoformat()
                columns_migrated.add(fld)
                row_changed = True
                changed = True
        if row_changed:
            rows_affected += 1

    if not changed:
        return RewriteResult(
            path=path,
            backup_path=None,
            columns_migrated=[],
            rows_affected=0,
            already_migrated=True,
        )

    backup_path = None
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    path.write_text(json.dumps(raw, indent=2, default=str) + "\n")

    return RewriteResult(
        path=path,
        backup_path=backup_path,
        columns_migrated=sorted(columns_migrated),
        rows_affected=rows_affected,
        already_migrated=False,
    )


def rewrite_data_dir(
    data_dir: Path,
    *,
    backup: bool = True,
) -> list[RewriteResult]:
    """Discover and rewrite all parquet and queue JSON files under *data_dir*.

    Args:
        data_dir: Root directory to scan.
        backup: If ``True`` (default), back up each file before rewriting.

    Returns:
        List of :class:`RewriteResult`, one per processed file.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    results: list[RewriteResult] = []

    for pq in sorted(data_dir.rglob("*.parquet")):
        results.append(rewrite_parquet_timestamps(pq, backup=backup))

    for js in sorted(data_dir.rglob("queue.json")):
        results.append(rewrite_json_timestamps(js, backup=backup))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_naive_iso(s: str) -> bool:
    """Return True if *s* is an ISO-8601 datetime string without tz offset."""
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError, TypeError:
        return False
    return parsed.tzinfo is None
