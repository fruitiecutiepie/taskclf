"""Tests for train.lgbm: encode_categoricals and prepare_xy.

Covers: TC-TRAIN-LGBM-001 through TC-TRAIN-LGBM-011,
        UNK-001 through UNK-003, PER-001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from taskclf.core.types import LABEL_SET_V1
from taskclf.train.lgbm import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_COLUMNS_V2,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V2,
    encode_categoricals,
    get_categorical_columns,
    get_feature_columns,
    prepare_xy,
)


def _make_feature_df(n_rows: int = 10, *, label: str = "Build") -> pd.DataFrame:
    """Build a minimal DataFrame with all FEATURE_COLUMNS + a label column."""
    data: dict[str, list] = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            data[col] = [f"{col}_val_{i % 3}" for i in range(n_rows)]
        elif col in ("is_browser", "is_editor", "is_terminal"):
            data[col] = [i % 2 == 0 for i in range(n_rows)]
        else:
            data[col] = [float(i) for i in range(n_rows)]
    data["label"] = [label] * n_rows
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TC-TRAIN-LGBM-001 through TC-TRAIN-LGBM-005: encode_categoricals
# ---------------------------------------------------------------------------


class TestEncodeCategoricals:
    def test_fit_new_encoders(self) -> None:
        """TC-TRAIN-LGBM-001: cat_encoders=None fits new encoders for all 4 columns."""
        df = _make_feature_df()
        encoded_df, encoders = encode_categoricals(df)

        assert set(encoders.keys()) == set(CATEGORICAL_COLUMNS)
        for col in CATEGORICAL_COLUMNS:
            assert encoded_df[col].dtype in (np.int64, np.intp, int)
            assert isinstance(encoders[col], LabelEncoder)

    def test_prefitted_encoder_reuse(self) -> None:
        """TC-TRAIN-LGBM-002: pre-fitted encoders produce identical codes."""
        df = _make_feature_df()
        _, encoders = encode_categoricals(df)

        df2 = _make_feature_df()
        encoded_df2, encoders2 = encode_categoricals(df2, cat_encoders=encoders)

        encoded_df1, _ = encode_categoricals(df)
        for col in CATEGORICAL_COLUMNS:
            pd.testing.assert_series_equal(
                encoded_df1[col], encoded_df2[col], check_names=True
            )

    def test_unknown_value_maps_to_unknown_code(self) -> None:
        """TC-TRAIN-LGBM-003: unseen values at inference time encode as __unknown__."""
        df_train = _make_feature_df(20)
        _, encoders = encode_categoricals(df_train, min_category_freq=1)

        df_infer = _make_feature_df(3)
        for col in CATEGORICAL_COLUMNS:
            df_infer[col] = "never_seen_before"

        encoded_df, _ = encode_categoricals(df_infer, cat_encoders=encoders)
        for col in CATEGORICAL_COLUMNS:
            le = encoders[col]
            expected = int(le.transform(["__unknown__"])[0])
            assert (encoded_df[col] == expected).all()

    def test_output_shape_preserved(self) -> None:
        """TC-TRAIN-LGBM-004: encoded DataFrame has same shape as input."""
        df = _make_feature_df(8)
        encoded_df, _ = encode_categoricals(df)
        assert encoded_df.shape == df.shape

    def test_non_categorical_columns_untouched(self) -> None:
        """TC-TRAIN-LGBM-005: numeric feature columns are unchanged."""
        df = _make_feature_df(5)
        encoded_df, _ = encode_categoricals(df)

        non_cat_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
        for col in non_cat_cols:
            pd.testing.assert_series_equal(df[col], encoded_df[col], check_names=True)


# ---------------------------------------------------------------------------
# TC-TRAIN-LGBM-006 through TC-TRAIN-LGBM-011: prepare_xy
# ---------------------------------------------------------------------------


class TestPrepareXY:
    def test_output_shapes(self) -> None:
        """TC-TRAIN-LGBM-006: X and y have correct shapes."""
        df = _make_feature_df(12)
        x, y, _, _ = prepare_xy(df)
        assert x.shape == (12, len(FEATURE_COLUMNS))
        assert y.shape == (12,)

    def test_nan_fill(self) -> None:
        """TC-TRAIN-LGBM-007: NaN in numeric features becomes 0 in X."""
        df = _make_feature_df(5)
        df.loc[0, "keys_per_min"] = float("nan")
        df.loc[1, "mouse_distance"] = float("nan")

        x, _, _, _ = prepare_xy(df)

        keys_idx = FEATURE_COLUMNS.index("keys_per_min")
        mouse_idx = FEATURE_COLUMNS.index("mouse_distance")
        assert x[0, keys_idx] == 0.0
        assert x[1, mouse_idx] == 0.0

    def test_label_encoding_default(self) -> None:
        """TC-TRAIN-LGBM-008: default label_encoder uses sorted(LABEL_SET_V1)."""
        df = _make_feature_df(5)
        _, _, le, _ = prepare_xy(df)
        assert list(le.classes_) == sorted(LABEL_SET_V1)
        assert len(le.classes_) == 8

    def test_prefitted_encoders_returned(self) -> None:
        """TC-TRAIN-LGBM-009: passing pre-fitted encoders returns same objects."""
        df = _make_feature_df(5)
        _, _, le, cat_enc = prepare_xy(df)

        df2 = _make_feature_df(3)
        _, _, le2, cat_enc2 = prepare_xy(df2, label_encoder=le, cat_encoders=cat_enc)

        assert le2 is le
        assert cat_enc2 is cat_enc

    def test_unknown_label_raises(self) -> None:
        """TC-TRAIN-LGBM-010: label not in LABEL_SET_V1 raises ValueError."""
        df = _make_feature_df(3, label="NotARealLabel")
        with pytest.raises(ValueError):
            prepare_xy(df)

    def test_x_dtype(self) -> None:
        """TC-TRAIN-LGBM-011: X is float64."""
        df = _make_feature_df(5)
        x, _, _, _ = prepare_xy(df)
        assert x.dtype == np.float64


# ---------------------------------------------------------------------------
# UNK-001 through UNK-003, PER-001: unknown-category handling
# ---------------------------------------------------------------------------


def _make_feature_df_with_rare(
    n_rows: int = 50,
    rare_count: int = 2,
) -> pd.DataFrame:
    """Build a DataFrame where some categorical values appear rarely."""
    data: dict[str, list] = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            vals = [f"{col}_common_{i % 3}" for i in range(n_rows)]
            for j in range(min(rare_count, n_rows)):
                vals[j] = f"{col}_rare_singleton_{j}"
            data[col] = vals
        elif col in ("is_browser", "is_editor", "is_terminal"):
            data[col] = [i % 2 == 0 for i in range(n_rows)]
        else:
            data[col] = [float(i) for i in range(n_rows)]
    data["label"] = ["Build"] * n_rows
    return pd.DataFrame(data)


class TestUnknownCategoryHandling:
    def test_unk001_rare_categories_become_unknown(self) -> None:
        """UNK-001: categories with count < min_category_freq become __unknown__."""
        df = _make_feature_df_with_rare(50, rare_count=2)
        _, encoders = encode_categoricals(
            df, min_category_freq=5, unknown_mask_rate=0, random_state=42
        )
        for col in CATEGORICAL_COLUMNS:
            le = encoders[col]
            assert "__unknown__" in set(le.classes_)

    def test_unk002_mask_rate_applied(self) -> None:
        """UNK-002: approximately unknown_mask_rate of known rows are masked."""
        n = 200
        df = _make_feature_df(n)
        encoded_no_mask, encs_no_mask = encode_categoricals(
            df.copy(), min_category_freq=1, unknown_mask_rate=0, random_state=0
        )
        encoded_with_mask, encs_mask = encode_categoricals(
            df.copy(), min_category_freq=1, unknown_mask_rate=0.10, random_state=0
        )

        for col in CATEGORICAL_COLUMNS:
            le = encs_mask[col]
            unknown_code = int(le.transform(["__unknown__"])[0])
            masked_count = (encoded_with_mask[col] == unknown_code).sum()
            assert masked_count > 0, f"{col}: expected some masked rows"
            mask_fraction = masked_count / n
            assert 0.02 < mask_fraction < 0.25, (
                f"{col}: mask fraction {mask_fraction:.2f} outside expected range"
            )

    def test_unk003_unknown_in_all_encoders(self) -> None:
        """UNK-003: __unknown__ is in le.classes_ for every categorical encoder."""
        df = _make_feature_df(50)
        _, encoders = encode_categoricals(
            df, min_category_freq=5, unknown_mask_rate=0.05, random_state=42
        )
        for col in CATEGORICAL_COLUMNS:
            le = encoders[col]
            assert "__unknown__" in set(le.classes_), (
                f"{col}: __unknown__ missing from encoder classes"
            )

    def test_per001_user_id_in_feature_columns(self) -> None:
        """PER-001: user_id is in FEATURE_COLUMNS for schema v1."""
        assert "user_id" in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# P2-001: schema-v2 feature columns exclude user_id
# ---------------------------------------------------------------------------


class TestSchemaV2Columns:
    def test_p2001_user_id_not_in_feature_columns_v2(self) -> None:
        """P2-001: user_id is NOT in FEATURE_COLUMNS_V2."""
        assert "user_id" not in FEATURE_COLUMNS_V2
        assert "user_id" not in CATEGORICAL_COLUMNS_V2

    def test_v2_columns_are_subset_of_v1(self) -> None:
        assert set(FEATURE_COLUMNS_V2) == set(FEATURE_COLUMNS) - {"user_id"}
        assert set(CATEGORICAL_COLUMNS_V2) == set(CATEGORICAL_COLUMNS) - {"user_id"}

    def test_get_feature_columns_dispatches(self) -> None:
        assert get_feature_columns("v1") == list(FEATURE_COLUMNS)
        assert get_feature_columns("v2") == list(FEATURE_COLUMNS_V2)

    def test_get_categorical_columns_dispatches(self) -> None:
        assert get_categorical_columns("v1") == list(CATEGORICAL_COLUMNS)
        assert get_categorical_columns("v2") == list(CATEGORICAL_COLUMNS_V2)

    def test_get_feature_columns_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema version"):
            get_feature_columns("v99")

    def test_get_categorical_columns_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema version"):
            get_categorical_columns("v99")
