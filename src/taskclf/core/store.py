"""Parquet I/O primitives for persisting DataFrames."""

# TODO: duckdb IO

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write *df* to a parquet file at *path* atomically.

    Writes to a temporary file in the same directory first, then
    atomically replaces the target via :func:`os.replace`.  This
    prevents readers from ever seeing a partially-written file.

    Args:
        df: DataFrame to persist.
        path: Destination file path (e.g. ``data/processed/.../features.parquet``).

    Returns:
        The *path* that was written, for convenient chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    try:
        os.close(fd)
        df.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Args:
        path: Path to an existing ``.parquet`` file.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_parquet(path, engine="pyarrow")
