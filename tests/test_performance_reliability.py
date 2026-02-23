"""Performance and reliability tests for taskclf.

Covers TC-PERF-001..002 and TC-REL-001..003.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from taskclf.core.store import read_parquet, write_parquet


# ---------------------------------------------------------------------------
# TC-PERF-001: Feature build processes 1 day under N seconds
# ---------------------------------------------------------------------------


class TestFeatureBuildPerformance:
    """TC-PERF-001: feature build for one day completes within a generous time budget."""

    def test_build_features_for_date_under_5s(self, tmp_path: Path) -> None:
        from taskclf.features.build import build_features_for_date
        import datetime as dt

        date = dt.date(2025, 6, 15)
        start = time.monotonic()
        build_features_for_date(date, tmp_path)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Feature build took {elapsed:.2f}s, expected <5s"

    def test_output_is_valid_parquet(self, tmp_path: Path) -> None:
        from taskclf.features.build import build_features_for_date
        import datetime as dt

        date = dt.date(2025, 6, 15)
        out = build_features_for_date(date, tmp_path)
        df = read_parquet(out)
        assert len(df) > 0
        assert "schema_version" in df.columns


# ---------------------------------------------------------------------------
# TC-PERF-002: DuckDB query performance
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="TODO: remove .skip once DuckDB query path is implemented in core/store.py"
)
def test_tc_perf_002_duckdb_query_performance() -> None:
    """TC-PERF-002: DuckDB queries for last 7 days return under N seconds."""


# ---------------------------------------------------------------------------
# TC-REL-001: Online inference handles missing minute gracefully
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="TODO: remove .skip once infer/online.py real-time predictor loop is implemented"
)
def test_tc_rel_001_online_handles_missing_minute() -> None:
    """TC-REL-001: online inference handles missing minute gracefully (no crash)."""


# ---------------------------------------------------------------------------
# TC-REL-002: Adapter temporary failure retries with backoff
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="TODO: remove .skip once adapters/activitywatch/client.py retry logic is implemented"
)
def test_tc_rel_002_adapter_retry_with_backoff() -> None:
    """TC-REL-002: adapter temporary failure retries with backoff."""


# ---------------------------------------------------------------------------
# TC-REL-003: Writes are atomic (write temp -> rename)
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    """TC-REL-003: write_parquet uses temp-file + os.replace for atomicity."""

    def test_successful_write_produces_readable_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = tmp_path / "output.parquet"

        write_parquet(df, out)

        assert out.exists()
        result = read_parquet(out)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_no_leftover_temp_files(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"x": range(10)})
        out = tmp_path / "clean.parquet"

        write_parquet(df, out)

        temps = list(tmp_path.glob("*.tmp"))
        assert temps == [], f"Leftover temp files: {temps}"

    def test_failed_write_does_not_leave_target(self, tmp_path: Path) -> None:
        """If serialization fails, the target file must not exist."""
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "should_not_exist.parquet"

        with patch("pandas.DataFrame.to_parquet", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                write_parquet(df, out)

        assert not out.exists(), "Target file should not exist after failed write"

    def test_failed_write_cleans_up_temp(self, tmp_path: Path) -> None:
        """A failed write must not leave temp files behind."""
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "fail_clean.parquet"

        with patch("pandas.DataFrame.to_parquet", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                write_parquet(df, out)

        temps = list(tmp_path.glob("*.tmp"))
        assert temps == [], f"Temp files left after failure: {temps}"

    def test_atomic_overwrite_preserves_old_on_failure(self, tmp_path: Path) -> None:
        """If a second write fails, the original file remains intact."""
        df_original = pd.DataFrame({"v": [100]})
        out = tmp_path / "overwrite.parquet"
        write_parquet(df_original, out)

        with patch("pandas.DataFrame.to_parquet", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                write_parquet(pd.DataFrame({"v": [999]}), out)

        result = read_parquet(out)
        assert result["v"].iloc[0] == 100, "Original file should survive a failed overwrite"
