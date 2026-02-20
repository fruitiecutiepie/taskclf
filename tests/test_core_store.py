"""Tests for parquet I/O primitives.

Covers: round-trip read/write, auto-creation of parent directories.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from taskclf.core.store import read_parquet, write_parquet


class TestParquetRoundTrip:
    def test_write_then_read_preserves_data(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = write_parquet(df, tmp_path / "test.parquet")
        loaded = read_parquet(out)
        pd.testing.assert_frame_equal(loaded, df)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "file.parquet"
        df = pd.DataFrame({"col": [42]})
        write_parquet(df, nested)
        assert nested.exists()

    def test_returns_written_path(self, tmp_path: Path) -> None:
        target = tmp_path / "out.parquet"
        df = pd.DataFrame({"x": [1]})
        result = write_parquet(df, target)
        assert result == target

    def test_read_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            read_parquet(tmp_path / "does_not_exist.parquet")
