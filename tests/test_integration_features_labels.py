"""Integration tests: features -> labels pipeline (no adapter dependency).

Covers TC-INT-010 (feature build produces valid partitioned parquet) and
TC-INT-011 (joining features with labels yields correct labeled row count).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import read_parquet
from taskclf.features.build import build_features_for_date, generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.labels.store import generate_dummy_labels


class TestFeatureBuildProducesParquet:
    """TC-INT-010: pipeline produces features_v1 parquet partition for date."""

    DATE = dt.date(2025, 6, 15)

    def test_parquet_file_exists(self, tmp_path: Path) -> None:
        out = build_features_for_date(self.DATE, tmp_path)
        assert out.exists()
        assert out.suffix == ".parquet"

    def test_parquet_path_follows_partition_convention(self, tmp_path: Path) -> None:
        out = build_features_for_date(self.DATE, tmp_path)
        expected = tmp_path / f"features_v1/date={self.DATE.isoformat()}" / "features.parquet"
        assert out == expected

    def test_schema_validates(self, tmp_path: Path) -> None:
        out = build_features_for_date(self.DATE, tmp_path)
        df = read_parquet(out)
        FeatureSchemaV1.validate_dataframe(df)

    def test_row_count_matches_default(self, tmp_path: Path) -> None:
        out = build_features_for_date(self.DATE, tmp_path)
        df = read_parquet(out)
        expected_rows = len(generate_dummy_features(self.DATE))
        assert len(df) == expected_rows

    def test_round_trip_preserves_schema_metadata(self, tmp_path: Path) -> None:
        out = build_features_for_date(self.DATE, tmp_path)
        df = read_parquet(out)
        assert df["schema_version"].iloc[0] == FeatureSchemaV1.VERSION
        assert df["schema_hash"].iloc[0] == FeatureSchemaV1.SCHEMA_HASH


class TestFeaturesLabelsJoin:
    """TC-INT-011: joining labels yields correct labeled training rows count."""

    DATE = dt.date(2025, 6, 15)
    N_ROWS = 30

    def test_all_rows_labeled_when_spans_align(self) -> None:
        rows = generate_dummy_features(self.DATE, n_rows=self.N_ROWS)
        features_df = pd.DataFrame([r.model_dump() for r in rows])
        labels = generate_dummy_labels(self.DATE, n_rows=self.N_ROWS)

        labeled = project_blocks_to_windows(features_df, labels)
        assert len(labeled) == self.N_ROWS

    def test_labeled_df_has_label_column(self) -> None:
        rows = generate_dummy_features(self.DATE, n_rows=self.N_ROWS)
        features_df = pd.DataFrame([r.model_dump() for r in rows])
        labels = generate_dummy_labels(self.DATE, n_rows=self.N_ROWS)

        labeled = project_blocks_to_windows(features_df, labels)
        assert "label" in labeled.columns
        assert labeled["label"].notna().all()

    def test_labels_are_from_valid_set(self) -> None:
        from taskclf.core.types import LABEL_SET_V1

        rows = generate_dummy_features(self.DATE, n_rows=self.N_ROWS)
        features_df = pd.DataFrame([r.model_dump() for r in rows])
        labels = generate_dummy_labels(self.DATE, n_rows=self.N_ROWS)

        labeled = project_blocks_to_windows(features_df, labels)
        invalid = set(labeled["label"].unique()) - set(LABEL_SET_V1)
        assert not invalid, f"Labels not in LABEL_SET_V1: {invalid}"

    def test_feature_columns_preserved_after_join(self) -> None:
        rows = generate_dummy_features(self.DATE, n_rows=self.N_ROWS)
        features_df = pd.DataFrame([r.model_dump() for r in rows])
        labels = generate_dummy_labels(self.DATE, n_rows=self.N_ROWS)

        labeled = project_blocks_to_windows(features_df, labels)
        for col in FeatureSchemaV1.COLUMNS:
            assert col in labeled.columns, f"Feature column {col!r} lost after join"
