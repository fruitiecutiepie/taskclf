"""Tests for FeatureSchemaV1: hash stability, row validation, DataFrame validation.

Covers: schema invariants S1, S2, S3.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from taskclf.core.schema import (
    FeatureSchemaV1,
    FeatureSchemaV2,
    FeatureSchemaV3,
    _build_schema_hash,
)


class TestSchemaHashStability:
    """S1 + S2: schema hash is non-empty, fixed-length, and deterministic."""

    def test_schema_hash_is_nonempty(self) -> None:
        assert len(FeatureSchemaV1.SCHEMA_HASH) > 0

    def test_schema_hash_length(self) -> None:
        from taskclf.core.hashing import _HASH_TRUNCATION

        assert len(FeatureSchemaV1.SCHEMA_HASH) == _HASH_TRUNCATION

    def test_schema_hash_deterministic(self) -> None:
        h1 = _build_schema_hash(FeatureSchemaV1.COLUMNS)
        h2 = _build_schema_hash(FeatureSchemaV1.COLUMNS)
        assert h1 == h2

    def test_schema_hash_matches_class_constant(self) -> None:
        assert (
            _build_schema_hash(FeatureSchemaV1.COLUMNS) == FeatureSchemaV1.SCHEMA_HASH
        )

    def test_different_columns_yield_different_hash(self) -> None:
        altered = {**FeatureSchemaV1.COLUMNS, "extra_col": int}
        assert _build_schema_hash(altered) != FeatureSchemaV1.SCHEMA_HASH


class TestValidateRow:
    def test_accepts_valid_data(self, valid_feature_row_data: dict[str, Any]) -> None:
        row = FeatureSchemaV1.validate_row(valid_feature_row_data)
        assert row.schema_version == FeatureSchemaV1.VERSION

    def test_rejects_wrong_schema_version(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "schema_version": "v999"}
        with pytest.raises(ValueError, match="schema_version mismatch"):
            FeatureSchemaV1.validate_row(data)

    def test_rejects_wrong_schema_hash(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "schema_hash": "000000000000"}
        with pytest.raises(ValueError, match="schema_hash mismatch"):
            FeatureSchemaV1.validate_row(data)


def _make_valid_df(data: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame from a valid_feature_row_data dict."""
    from taskclf.core.types import FeatureRow

    row = FeatureRow.model_validate(data)
    return pd.DataFrame([row.model_dump()])


class TestValidateDataFrame:
    def test_accepts_valid_dataframe(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        df = _make_valid_df(valid_feature_row_data)
        FeatureSchemaV1.validate_dataframe(df)

    def test_rejects_missing_column(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        df = _make_valid_df(valid_feature_row_data).drop(columns=["app_id"])
        with pytest.raises(ValueError, match="Missing columns"):
            FeatureSchemaV1.validate_dataframe(df)

    def test_rejects_extra_column(self, valid_feature_row_data: dict[str, Any]) -> None:
        df = _make_valid_df(valid_feature_row_data)
        df["bonus_column"] = 42
        with pytest.raises(ValueError, match="Unexpected columns"):
            FeatureSchemaV1.validate_dataframe(df)

    def test_rejects_wrong_dtype(self, valid_feature_row_data: dict[str, Any]) -> None:
        df = _make_valid_df(valid_feature_row_data)
        df["hour_of_day"] = df["hour_of_day"].astype(str)
        with pytest.raises(ValueError, match="dtype mismatches"):
            FeatureSchemaV1.validate_dataframe(df)


# ---------------------------------------------------------------------------
# P6-002 (adapted): FeatureSchemaV2
# ---------------------------------------------------------------------------


class TestFeatureSchemaV2:
    def test_schema_hash_differs_from_v1(self) -> None:
        """P6-002: v2 hash must differ from v1 because user_id is removed."""
        assert FeatureSchemaV2.SCHEMA_HASH != FeatureSchemaV1.SCHEMA_HASH

    def test_user_id_not_in_v2_columns(self) -> None:
        assert "user_id" not in FeatureSchemaV2.COLUMNS

    def test_v2_columns_subset_of_v1(self) -> None:
        assert set(FeatureSchemaV2.COLUMNS) == set(FeatureSchemaV1.COLUMNS) - {
            "user_id"
        }

    def test_v2_version_string(self) -> None:
        assert FeatureSchemaV2.VERSION == "v2"

    def test_v2_hash_deterministic(self) -> None:
        h1 = _build_schema_hash(FeatureSchemaV2.COLUMNS)
        h2 = _build_schema_hash(FeatureSchemaV2.COLUMNS)
        assert h1 == h2 == FeatureSchemaV2.SCHEMA_HASH


class TestFeatureSchemaV3:
    def test_schema_hash_differs_from_v1_and_v2(self) -> None:
        assert FeatureSchemaV3.SCHEMA_HASH != FeatureSchemaV1.SCHEMA_HASH
        assert FeatureSchemaV3.SCHEMA_HASH != FeatureSchemaV2.SCHEMA_HASH

    def test_v3_includes_user_id_and_title_sketch_fields(self) -> None:
        assert "user_id" in FeatureSchemaV3.COLUMNS
        assert "title_token_sketch_000" in FeatureSchemaV3.COLUMNS
        assert "title_char3_sketch_127" in FeatureSchemaV3.COLUMNS

    def test_v3_hash_deterministic(self) -> None:
        h1 = _build_schema_hash(FeatureSchemaV3.COLUMNS)
        h2 = _build_schema_hash(FeatureSchemaV3.COLUMNS)
        assert h1 == h2 == FeatureSchemaV3.SCHEMA_HASH
