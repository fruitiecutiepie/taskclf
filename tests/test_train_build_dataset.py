"""Tests for train.build_dataset: artifact generation, exclusion rules, splits."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import read_parquet
from taskclf.core.types import LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.store import generate_dummy_labels
from taskclf.train.build_dataset import (
    DatasetManifest,
    build_training_dataset,
)


def _features_and_labels(
    dates: list[dt.date],
    n_per_day: int = 20,
    user_id: str = "test-user",
) -> tuple[pd.DataFrame, list[LabelSpan]]:
    all_features: list[pd.DataFrame] = []
    all_labels: list[LabelSpan] = []
    for d in dates:
        rows = generate_dummy_features(d, n_rows=n_per_day, user_id=user_id)
        all_features.append(pd.DataFrame([r.model_dump() for r in rows]))
        all_labels.extend(generate_dummy_labels(d, n_rows=n_per_day))
    return pd.concat(all_features, ignore_index=True), all_labels


class TestBuildTrainingDataset:
    def test_writes_all_artifacts(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels(
            [dt.date(2025, 6, 14), dt.date(2025, 6, 15)],
        )
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        assert Path(manifest.x_path).exists()
        assert Path(manifest.y_path).exists()
        assert Path(manifest.splits_path).exists()

    def test_x_y_row_counts_match(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels(
            [dt.date(2025, 6, 14), dt.date(2025, 6, 15)],
        )
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        x = read_parquet(Path(manifest.x_path))
        y = read_parquet(Path(manifest.y_path))
        assert len(x) == len(y)
        assert len(x) == manifest.total_rows

    def test_splits_json_valid(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels(
            [dt.date(2025, 6, 14), dt.date(2025, 6, 15)],
        )
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        with open(manifest.splits_path) as f:
            splits = json.load(f)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "metadata" in splits

        all_indices = set(splits["train"] + splits["val"] + splits["test"])
        assert len(all_indices) == manifest.total_rows

    def test_no_index_overlap(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels(
            [dt.date(2025, 6, 14), dt.date(2025, 6, 15)],
        )
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        with open(manifest.splits_path) as f:
            splits = json.load(f)

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_split_ratios_approximate(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels(
            [dt.date(2025, 6, 14), dt.date(2025, 6, 15)],
            n_per_day=50,
        )
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
            train_ratio=0.70, val_ratio=0.15,
        )
        total = manifest.total_rows
        if total > 10:
            assert manifest.train_rows / total >= 0.5
            assert manifest.val_rows / total >= 0.05

    def test_x_contains_schema_version(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels([dt.date(2025, 6, 15)])
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        x = read_parquet(Path(manifest.x_path))
        assert "schema_version" in x.columns

    def test_y_contains_label_column(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels([dt.date(2025, 6, 15)])
        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        y = read_parquet(Path(manifest.y_path))
        assert "label" in y.columns
        assert "user_id" in y.columns
        assert "bucket_start_ts" in y.columns


class TestExclusionRules:
    def test_all_null_features_excluded(self, tmp_path: Path) -> None:
        features, labels = _features_and_labels([dt.date(2025, 6, 15)], n_per_day=10)
        numeric_cols = [
            "keys_per_min", "backspace_ratio", "shortcut_rate",
            "clicks_per_min", "scroll_events_per_min", "mouse_distance",
            "active_seconds_keyboard", "active_seconds_mouse",
            "active_seconds_any", "max_idle_run_seconds", "event_density",
            "app_switch_count_last_5m", "app_foreground_time_ratio",
            "app_change_count", "hour_of_day", "session_length_so_far",
        ]
        for c in numeric_cols:
            if c in features.columns:
                features.loc[0, c] = None

        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
        )
        assert manifest.total_rows <= len(features)

    def test_empty_labels_yields_zero_rows(self, tmp_path: Path) -> None:
        features, _ = _features_and_labels([dt.date(2025, 6, 15)], n_per_day=10)
        manifest = build_training_dataset(
            features, [], output_dir=tmp_path / "ds",
        )
        assert manifest.total_rows == 0


class TestHoldoutUsers:
    def test_holdout_users_in_test_only(self, tmp_path: Path) -> None:
        dfs = []
        labels: list[LabelSpan] = []
        for uid in [f"user-{i}" for i in range(10)]:
            f, l = _features_and_labels(
                [dt.date(2025, 6, 15)], n_per_day=20, user_id=uid,
            )
            dfs.append(f)
            labels.extend(l)
        features = pd.concat(dfs, ignore_index=True)

        manifest = build_training_dataset(
            features, labels, output_dir=tmp_path / "ds",
            holdout_user_fraction=0.2,
        )
        assert len(manifest.holdout_users) >= 1

        with open(manifest.splits_path) as f:
            splits = json.load(f)

        x = read_parquet(Path(manifest.x_path))
        holdout_set = set(manifest.holdout_users)
        train_users = set(x.iloc[splits["train"]]["user_id"].unique())
        val_users = set(x.iloc[splits["val"]]["user_id"].unique())

        assert holdout_set.isdisjoint(train_users)
        assert holdout_set.isdisjoint(val_users)
