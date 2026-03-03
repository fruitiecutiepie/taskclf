"""Tests for train.lgbm: encode_categoricals and prepare_xy.

Covers: TC-TRAIN-LGBM-001 through TC-TRAIN-LGBM-011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from taskclf.core.types import LABEL_SET_V1
from taskclf.train.lgbm import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    encode_categoricals,
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

    def test_unknown_value_maps_to_minus_one(self) -> None:
        """TC-TRAIN-LGBM-003: unseen values at inference time encode as -1."""
        df_train = _make_feature_df(5)
        _, encoders = encode_categoricals(df_train)

        df_infer = _make_feature_df(3)
        for col in CATEGORICAL_COLUMNS:
            df_infer[col] = "never_seen_before"

        encoded_df, _ = encode_categoricals(df_infer, cat_encoders=encoders)
        for col in CATEGORICAL_COLUMNS:
            assert (encoded_df[col] == -1).all()

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
