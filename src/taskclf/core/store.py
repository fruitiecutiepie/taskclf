# TODO: duckdb IO

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write *df* to a parquet file at *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")
