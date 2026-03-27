"""Tests for the timestamp migration tooling (Phase 5).

Covers:
- TC-MIG-AUD-PQ-001..005  (parquet audit)
- TC-MIG-AUD-JS-001..005  (JSON audit)
- TC-MIG-AUD-DIR-001..003 (directory audit)
- TC-MIG-RW-PQ-001..005   (parquet rewrite)
- TC-MIG-RW-JS-001..005   (JSON rewrite)
- TC-MIG-RW-DIR-001..003  (directory rewrite)
- TC-MIG-EDGE-001..003    (edge cases)
- TC-MIG-CLI-001..004     (CLI integration)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.store import read_parquet, write_parquet
from taskclf.migrate.timestamps import (
    audit_data_dir,
    audit_json_timestamps,
    audit_parquet_timestamps,
    rewrite_data_dir,
    rewrite_json_timestamps,
    rewrite_parquet_timestamps,
)

runner = CliRunner()

_UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_naive_parquet(path: Path, n: int = 5) -> None:
    """Write a parquet file with naive datetime columns."""
    base = dt.datetime(2025, 6, 15, 10, 0)
    df = pd.DataFrame(
        {
            "start_ts": [base + dt.timedelta(minutes=i) for i in range(n)],
            "end_ts": [base + dt.timedelta(minutes=i + 1) for i in range(n)],
            "label": ["Build"] * n,
        }
    )
    write_parquet(df, path)


def _write_aware_parquet(path: Path, n: int = 5) -> None:
    """Write a parquet file with aware-UTC datetime columns."""
    base = dt.datetime(2025, 6, 15, 10, 0, tzinfo=_UTC)
    df = pd.DataFrame(
        {
            "start_ts": [base + dt.timedelta(minutes=i) for i in range(n)],
            "end_ts": [base + dt.timedelta(minutes=i + 1) for i in range(n)],
            "label": ["Build"] * n,
        }
    )
    write_parquet(df, path)


def _write_naive_json(path: Path, n: int = 3) -> None:
    """Write a JSON file with naive ISO datetime strings."""
    records = [
        {
            "bucket_start_ts": f"2025-06-15T{10 + i:02d}:00:00",
            "bucket_end_ts": f"2025-06-15T{10 + i:02d}:01:00",
            "created_at": f"2025-06-15T{10 + i:02d}:00:00",
            "user_id": "test-user",
            "status": "pending",
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def _write_aware_json(path: Path, n: int = 3) -> None:
    """Write a JSON file with aware-UTC ISO datetime strings."""
    records = [
        {
            "bucket_start_ts": f"2025-06-15T{10 + i:02d}:00:00+00:00",
            "bucket_end_ts": f"2025-06-15T{10 + i:02d}:01:00+00:00",
            "created_at": f"2025-06-15T{10 + i:02d}:00:00+00:00",
            "user_id": "test-user",
            "status": "pending",
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


# ===========================================================================
# Parquet audit tests
# ===========================================================================


class TestAuditParquetTimestamps:
    def test_all_naive_detected(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-PQ-001: all naive columns are flagged."""
        pq = tmp_path / "labels.parquet"
        _write_naive_parquet(pq)

        result = audit_parquet_timestamps(pq)

        assert result.needs_migration is True
        ts_cols = {c.column for c in result.columns}
        assert "start_ts" in ts_cols
        assert "end_ts" in ts_cols
        for c in result.columns:
            assert c.naive_count > 0
            assert c.aware_count == 0

    def test_all_aware_ok(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-PQ-002: all aware columns are clean."""
        pq = tmp_path / "features.parquet"
        _write_aware_parquet(pq)

        result = audit_parquet_timestamps(pq)

        assert result.needs_migration is False
        for c in result.columns:
            assert c.naive_count == 0
            assert c.aware_count > 0

    def test_non_datetime_columns_ignored(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-PQ-003: string/int columns not included in audit."""
        df = pd.DataFrame({"name": ["a", "b"], "count": [1, 2]})
        pq = tmp_path / "no_ts.parquet"
        write_parquet(df, pq)

        result = audit_parquet_timestamps(pq)

        assert result.columns == []
        assert result.needs_migration is False

    def test_empty_dataframe(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-PQ-004: empty DataFrame with datetime columns."""
        df = pd.DataFrame(
            {"start_ts": pd.Series([], dtype="datetime64[ns]"), "label": []}
        )
        pq = tmp_path / "empty.parquet"
        write_parquet(df, pq)

        result = audit_parquet_timestamps(pq)

        assert len(result.columns) == 1
        assert result.columns[0].total == 0
        assert result.needs_migration is False

    def test_file_not_found(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-PQ-005: non-existent file raises."""
        with pytest.raises(FileNotFoundError):
            audit_parquet_timestamps(tmp_path / "missing.parquet")


# ===========================================================================
# JSON audit tests
# ===========================================================================


class TestAuditJsonTimestamps:
    def test_all_naive_detected(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-JS-001: all naive strings are flagged."""
        js = tmp_path / "queue.json"
        _write_naive_json(js)

        result = audit_json_timestamps(js)

        assert result.needs_migration is True
        col_names = {c.column for c in result.columns}
        assert {"bucket_start_ts", "bucket_end_ts", "created_at"} == col_names
        for c in result.columns:
            assert c.naive_count > 0

    def test_all_aware_ok(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-JS-002: all aware strings are clean."""
        js = tmp_path / "queue.json"
        _write_aware_json(js)

        result = audit_json_timestamps(js)

        assert result.needs_migration is False
        for c in result.columns:
            assert c.aware_count > 0
            assert c.naive_count == 0

    def test_z_suffix_treated_as_aware(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-JS-003: ISO strings ending with Z are aware."""
        records = [
            {
                "start_ts": "2025-06-15T10:00:00Z",
                "end_ts": "2025-06-15T10:05:00Z",
            }
        ]
        js = tmp_path / "queue.json"
        js.write_text(json.dumps(records))

        result = audit_json_timestamps(js)

        assert result.needs_migration is False
        for c in result.columns:
            assert c.naive_count == 0

    def test_mixed_naive_aware(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-JS-004: mixed naive and aware strings both counted."""
        records = [
            {"bucket_start_ts": "2025-06-15T10:00:00"},
            {"bucket_start_ts": "2025-06-15T11:00:00+00:00"},
        ]
        js = tmp_path / "queue.json"
        js.write_text(json.dumps(records))

        result = audit_json_timestamps(js)

        assert result.needs_migration is True
        assert len(result.columns) == 1
        c = result.columns[0]
        assert c.naive_count == 1
        assert c.aware_count == 1

    def test_empty_array(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-JS-005: empty JSON array yields no columns."""
        js = tmp_path / "queue.json"
        js.write_text("[]")

        result = audit_json_timestamps(js)

        assert result.columns == []
        assert result.needs_migration is False


# ===========================================================================
# Directory audit tests
# ===========================================================================


class TestAuditDataDir:
    def test_discovers_parquet_and_json(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-DIR-001: finds parquet and queue.json files."""
        _write_naive_parquet(tmp_path / "labels_v1" / "labels.parquet")
        _write_naive_json(tmp_path / "labels_v1" / "queue.json")

        results = audit_data_dir(tmp_path)

        assert len(results) == 2
        types = {r.file_type for r in results}
        assert types == {"parquet", "json"}

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-DIR-002: non-parquet/non-queue-json files ignored."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        (tmp_path / "other.json").write_text("{}")

        results = audit_data_dir(tmp_path)

        assert results == []

    def test_dir_not_found(self, tmp_path: Path) -> None:
        """TC-MIG-AUD-DIR-003: non-existent directory raises."""
        with pytest.raises(FileNotFoundError):
            audit_data_dir(tmp_path / "no_such_dir")


# ===========================================================================
# Parquet rewrite tests
# ===========================================================================


class TestRewriteParquetTimestamps:
    def test_naive_to_aware(self, tmp_path: Path) -> None:
        """TC-MIG-RW-PQ-001: naive columns are localized to UTC."""
        pq = tmp_path / "labels.parquet"
        _write_naive_parquet(pq, n=5)

        result = rewrite_parquet_timestamps(pq)

        assert result.already_migrated is False
        assert "start_ts" in result.columns_migrated
        assert "end_ts" in result.columns_migrated
        assert result.rows_affected == 5

        df = read_parquet(pq)
        assert df["start_ts"].dt.tz is not None
        assert df["end_ts"].dt.tz is not None

    def test_idempotent_on_aware(self, tmp_path: Path) -> None:
        """TC-MIG-RW-PQ-002: already-aware file is not rewritten."""
        pq = tmp_path / "features.parquet"
        _write_aware_parquet(pq)

        result = rewrite_parquet_timestamps(pq)

        assert result.already_migrated is True
        assert result.columns_migrated == []
        assert result.backup_path is None

    def test_backup_created(self, tmp_path: Path) -> None:
        """TC-MIG-RW-PQ-003: backup file created when backup=True."""
        pq = tmp_path / "labels.parquet"
        _write_naive_parquet(pq)

        result = rewrite_parquet_timestamps(pq, backup=True)

        assert result.backup_path is not None
        assert result.backup_path.exists()
        bak_df = read_parquet(result.backup_path)
        assert bak_df["start_ts"].dt.tz is None

    def test_no_backup_mode(self, tmp_path: Path) -> None:
        """TC-MIG-RW-PQ-004: no backup created when backup=False."""
        pq = tmp_path / "labels.parquet"
        _write_naive_parquet(pq)

        result = rewrite_parquet_timestamps(pq, backup=False)

        assert result.backup_path is None
        assert result.already_migrated is False
        bak = pq.with_suffix(pq.suffix + ".bak")
        assert not bak.exists()

    def test_file_not_found(self, tmp_path: Path) -> None:
        """TC-MIG-RW-PQ-005: non-existent file raises."""
        with pytest.raises(FileNotFoundError):
            rewrite_parquet_timestamps(tmp_path / "missing.parquet")


# ===========================================================================
# JSON rewrite tests
# ===========================================================================


class TestRewriteJsonTimestamps:
    def test_naive_to_aware(self, tmp_path: Path) -> None:
        """TC-MIG-RW-JS-001: naive ISO strings are normalized to aware UTC."""
        js = tmp_path / "queue.json"
        _write_naive_json(js, n=3)

        result = rewrite_json_timestamps(js)

        assert result.already_migrated is False
        assert result.rows_affected == 3
        assert set(result.columns_migrated) == {
            "bucket_start_ts",
            "bucket_end_ts",
            "created_at",
        }

        reloaded = json.loads(js.read_text())
        for record in reloaded:
            assert "+00:00" in record["bucket_start_ts"]
            assert "+00:00" in record["bucket_end_ts"]
            assert "+00:00" in record["created_at"]

    def test_idempotent_on_aware(self, tmp_path: Path) -> None:
        """TC-MIG-RW-JS-002: already-aware file is not rewritten."""
        js = tmp_path / "queue.json"
        _write_aware_json(js)

        result = rewrite_json_timestamps(js)

        assert result.already_migrated is True
        assert result.columns_migrated == []
        assert result.backup_path is None

    def test_backup_created(self, tmp_path: Path) -> None:
        """TC-MIG-RW-JS-003: backup file created when backup=True."""
        js = tmp_path / "queue.json"
        _write_naive_json(js)

        result = rewrite_json_timestamps(js, backup=True)

        assert result.backup_path is not None
        assert result.backup_path.exists()
        bak_data = json.loads(result.backup_path.read_text())
        assert "+00:00" not in bak_data[0]["bucket_start_ts"]

    def test_no_backup_mode(self, tmp_path: Path) -> None:
        """TC-MIG-RW-JS-004: no backup when backup=False."""
        js = tmp_path / "queue.json"
        _write_naive_json(js)

        result = rewrite_json_timestamps(js, backup=False)

        assert result.backup_path is None
        assert result.already_migrated is False
        bak = js.with_suffix(js.suffix + ".bak")
        assert not bak.exists()

    def test_file_not_found(self, tmp_path: Path) -> None:
        """TC-MIG-RW-JS-005: non-existent file raises."""
        with pytest.raises(FileNotFoundError):
            rewrite_json_timestamps(tmp_path / "missing.json")


# ===========================================================================
# Directory rewrite tests
# ===========================================================================


class TestRewriteDataDir:
    def test_rewrites_all_matching_files(self, tmp_path: Path) -> None:
        """TC-MIG-RW-DIR-001: rewrites parquet and queue.json files."""
        _write_naive_parquet(tmp_path / "labels_v1" / "labels.parquet")
        _write_naive_json(tmp_path / "labels_v1" / "queue.json")

        results = rewrite_data_dir(tmp_path, backup=False)

        assert len(results) == 2
        migrated = [r for r in results if not r.already_migrated]
        assert len(migrated) == 2

    def test_skips_already_migrated(self, tmp_path: Path) -> None:
        """TC-MIG-RW-DIR-002: already-aware files are skipped."""
        _write_aware_parquet(
            tmp_path / "features_v1" / "day=2025-06-15" / "features.parquet"
        )
        _write_aware_json(tmp_path / "labels_v1" / "queue.json")

        results = rewrite_data_dir(tmp_path)

        assert len(results) == 2
        assert all(r.already_migrated for r in results)

    def test_dir_not_found(self, tmp_path: Path) -> None:
        """TC-MIG-RW-DIR-003: non-existent directory raises."""
        with pytest.raises(FileNotFoundError):
            rewrite_data_dir(tmp_path / "no_such_dir")


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_json_non_array_returns_empty_audit(self, tmp_path: Path) -> None:
        """TC-MIG-EDGE-001: JSON file containing an object (not array) is handled."""
        js = tmp_path / "queue.json"
        js.write_text('{"key": "value"}')

        result = audit_json_timestamps(js)

        assert result.columns == []
        assert result.needs_migration is False

    def test_json_non_array_rewrite_noop(self, tmp_path: Path) -> None:
        """TC-MIG-EDGE-002: JSON file containing an object is a no-op rewrite."""
        js = tmp_path / "queue.json"
        js.write_text('{"key": "value"}')

        result = rewrite_json_timestamps(js)

        assert result.already_migrated is True

    def test_parquet_no_datetime_cols(self, tmp_path: Path) -> None:
        """TC-MIG-EDGE-003: parquet with no datetime columns is a no-op."""
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        pq = tmp_path / "data.parquet"
        write_parquet(df, pq)

        audit = audit_parquet_timestamps(pq)
        assert audit.columns == []
        assert audit.needs_migration is False

        rewrite = rewrite_parquet_timestamps(pq)
        assert rewrite.already_migrated is True


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestMigrateCli:
    def test_audit_shows_naive_columns(self, tmp_path: Path) -> None:
        """TC-MIG-CLI-001: audit command prints table with naive columns."""
        _write_naive_parquet(tmp_path / "labels.parquet")

        result = runner.invoke(app, ["migrate", "audit", "--data-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "needs migration" in result.output.lower() or "Naive" in result.output

    def test_audit_clean_dir(self, tmp_path: Path) -> None:
        """TC-MIG-CLI-002: audit command reports all clean."""
        _write_aware_parquet(tmp_path / "features.parquet")

        result = runner.invoke(app, ["migrate", "audit", "--data-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert (
            "already timezone-aware" in result.output.lower()
            or "ok" in result.output.lower()
        )

    def test_rewrite_migrates_files(self, tmp_path: Path) -> None:
        """TC-MIG-CLI-003: rewrite command normalizes naive files."""
        _write_naive_parquet(tmp_path / "labels.parquet")
        _write_naive_json(tmp_path / "queue.json")

        result = runner.invoke(
            app,
            ["migrate", "rewrite", "--data-dir", str(tmp_path), "--no-backup"],
        )

        assert result.exit_code == 0
        assert "Migrated: 2" in result.output

        df = read_parquet(tmp_path / "labels.parquet")
        assert df["start_ts"].dt.tz is not None

    def test_rewrite_missing_dir(self) -> None:
        """TC-MIG-CLI-004: rewrite with missing dir exits non-zero."""
        result = runner.invoke(
            app,
            ["migrate", "rewrite", "--data-dir", "/tmp/nonexistent_taskclf_dir"],
        )

        assert result.exit_code != 0
