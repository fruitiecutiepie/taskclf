"""Training dataset builder: join, exclude, split, and write X/y/splits artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS, MIN_BLOCK_DURATION_SECONDS
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import write_parquet
from taskclf.core.types import LabelSpan
from taskclf.train.dataset import assign_labels_to_buckets, split_by_time
from taskclf.train.lgbm import FEATURE_COLUMNS

_ID_COLUMNS = ["user_id", "bucket_start_ts", "session_id"]

_NUMERIC_FEATURES = [
    c for c in FEATURE_COLUMNS
    if c not in {"app_id", "app_category", "is_browser", "is_editor", "is_terminal", "day_of_week"}
]


class DatasetManifest(BaseModel, frozen=True):
    """Summary returned by :func:`build_training_dataset`."""

    x_path: str
    y_path: str
    splits_path: str
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    excluded_rows: int
    holdout_users: list[str]
    class_distribution: dict[str, int]


def _exclude_short_sessions(
    df: pd.DataFrame,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    min_block_seconds: int = MIN_BLOCK_DURATION_SECONDS,
) -> pd.DataFrame:
    """Drop rows from sessions shorter than *min_block_seconds*."""
    min_buckets = max(1, min_block_seconds // bucket_seconds)
    counts = df.groupby("session_id")["session_id"].transform("count")
    return df[counts >= min_buckets].copy()


def _exclude_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every numeric feature is null."""
    numeric_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
    if not numeric_cols:
        return df
    all_null = df[numeric_cols].isnull().all(axis=1)
    return df[~all_null].copy()


def build_training_dataset(
    features_df: pd.DataFrame,
    label_spans: Sequence[LabelSpan],
    *,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    holdout_user_fraction: float = 0.0,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> DatasetManifest:
    """Join features with labels, apply exclusions, split, and write artifacts.

    Outputs:
        ``output_dir/X.parquet`` -- feature matrix with ID columns and
        ``schema_version``.
        ``output_dir/y.parquet`` -- labels keyed by ``user_id`` and
        ``bucket_start_ts``.
        ``output_dir/splits.json`` -- train/val/test index lists and
        metadata.

    Args:
        features_df: Feature DataFrame conforming to ``FeatureSchemaV1``.
        label_spans: Label spans to project onto feature windows.
        output_dir: Directory to write artifacts into (created if needed).
        train_ratio: Fraction of each user's data for training.
        val_ratio: Fraction for validation.
        holdout_user_fraction: Fraction of users held out entirely for
            the test set (cold-start evaluation).
        bucket_seconds: Window width in seconds.

    Returns:
        A :class:`DatasetManifest` with paths and summary statistics.
    """
    labeled = assign_labels_to_buckets(features_df, label_spans)
    pre_exclusion = len(labeled)

    labeled = _exclude_short_sessions(labeled, bucket_seconds=bucket_seconds)
    labeled = _exclude_missing_critical(labeled)
    labeled = labeled.sort_values("bucket_start_ts").reset_index(drop=True)

    excluded = pre_exclusion - len(labeled)

    splits = split_by_time(
        labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_user_fraction=holdout_user_fraction,
    )

    x_cols = (
        _ID_COLUMNS
        + ["schema_version"]
        + [c for c in FEATURE_COLUMNS if c in labeled.columns]
    )
    x_df = labeled[x_cols]

    provenance_col = "provenance"
    y_cols = ["user_id", "bucket_start_ts", "label"]
    if provenance_col in labeled.columns:
        y_cols.append(provenance_col)
    y_df = labeled[y_cols]

    output_dir = Path(output_dir)
    x_path = output_dir / "X.parquet"
    y_path = output_dir / "y.parquet"
    splits_path = output_dir / "splits.json"

    write_parquet(x_df, x_path)
    write_parquet(y_df, y_path)

    class_dist = labeled["label"].value_counts().to_dict()

    splits_payload: dict[str, Any] = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "holdout_users": splits["holdout_users"],
        "metadata": {
            "feature_schema_version": FeatureSchemaV1.VERSION,
            "label_schema_version": "labels_v1",
            "total_rows": len(labeled),
            "excluded_rows": excluded,
            "user_count": labeled["user_id"].nunique(),
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "holdout_user_fraction": holdout_user_fraction,
        },
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits_payload, indent=2, default=str))

    return DatasetManifest(
        x_path=str(x_path),
        y_path=str(y_path),
        splits_path=str(splits_path),
        total_rows=len(labeled),
        train_rows=len(splits["train"]),
        val_rows=len(splits["val"]),
        test_rows=len(splits["test"]),
        excluded_rows=excluded,
        holdout_users=splits["holdout_users"],
        class_distribution={str(k): int(v) for k, v in class_dist.items()},
    )
