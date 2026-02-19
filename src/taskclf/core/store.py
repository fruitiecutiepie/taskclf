# TODO: duckdb IO

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write *df* to a parquet file at *path*, creating parent dirs as needed.

    Args:
        df: DataFrame to persist.
        path: Destination file path (e.g. ``data/processed/.../features.parquet``).

    Returns:
        The *path* that was written, for convenient chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Args:
        path: Path to an existing ``.parquet`` file.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_parquet(path, engine="pyarrow")
