"""Integration tests: train/batch/online preprocessing parity.

Covers:
- TSP-001: FeatureRow with None numerics produces 0.0 in both train and online paths
- TSP-002: batch predict_proba and online predict_bucket produce matching probabilities
- TSP-003: per-column regression guard for numeric imputation parity
- P0-001: batch and online paths produce identical feature vectors
- P0-002: online pipeline stages all execute (encode, predict, calibrate, reject, smooth)
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from taskclf.core.model_io import build_metadata, load_model_bundle, save_model_bundle
from taskclf.core.types import LABEL_SET_V1, FeatureRow, FeatureRowBase, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.infer.batch import predict_proba
from taskclf.infer.online import OnlinePredictor
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import (
    encode_categoricals,
    get_categorical_columns,
    get_feature_columns,
    train_lgbm,
)

TEST_SCHEMA_VERSION = "v3"
TEST_FEATURE_COLUMNS = get_feature_columns(TEST_SCHEMA_VERSION)
TEST_CATEGORICAL_COLUMNS = get_categorical_columns(TEST_SCHEMA_VERSION)
NUMERIC_FEATURE_COLUMNS = [
    c for c in TEST_FEATURE_COLUMNS if c not in TEST_CATEGORICAL_COLUMNS
]

_NULLABLE_NUMERIC_COLUMNS = [
    c for c in NUMERIC_FEATURE_COLUMNS if FeatureRow.model_fields[c].default is None
]


def _row_frame(
    row: FeatureRowBase,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame([{col: getattr(row, col) for col in feature_columns}])


def _build_labeled_df() -> pd.DataFrame:
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows: list[FeatureRowBase] = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=20))

    features_df = pd.DataFrame([r.model_dump() for r in all_rows])

    spans: list[LabelSpan] = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day)
        spans.extend(
            [
                LabelSpan(
                    start_ts=base.replace(hour=9),
                    end_ts=base.replace(hour=12),
                    label="Build",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=12),
                    end_ts=base.replace(hour=14),
                    label="Write",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=14),
                    end_ts=base.replace(hour=16),
                    label="Communicate",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=16),
                    end_ts=base.replace(hour=17),
                    label="BreakIdle",
                    provenance="test",
                ),
            ]
        )

    return project_blocks_to_windows(features_df, spans)


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """Train a small model once and return artifacts for parity tests."""
    labeled = _build_labeled_df()
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df,
        val_df,
        num_boost_round=5,
    )

    base_dir = tmp_path_factory.mktemp("parity_models")
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dt.date(2025, 6, 14),
        train_date_to=dt.date(2025, 6, 15),
        params=params,
        dataset_hash="parity_test_hash",
        data_provenance="synthetic",
    )
    run_dir = save_model_bundle(
        model, metadata, metrics, cm_df, base_dir, cat_encoders=cat_encoders
    )

    return {
        "run_dir": run_dir,
        "val_df": val_df,
        "cat_encoders": cat_encoders,
    }


class TestMissingNumericParity:
    """TSP-001, TSP-003: imputation parity for missing numerics."""

    def test_tsp001_none_numerics_produce_zero_both_paths(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """TSP-001: None numerics yield 0.0 through both prepare_xy and _encode_value."""
        data = {**valid_feature_row_data}
        for col in _NULLABLE_NUMERIC_COLUMNS:
            data[col] = None
        row = FeatureRow(**data)

        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)
        feature_columns = get_feature_columns(metadata.schema_version)

        for col in _NULLABLE_NUMERIC_COLUMNS:
            online_val = pred._encode_value(col, getattr(row, col))
            assert online_val == 0.0, f"{col}: online path produced {online_val}"

        feat_df = _row_frame(
            row,
            feature_columns=feature_columns,
        )
        feat_df, _ = encode_categoricals(
            feat_df,
            cat_encoders,
            schema_version=metadata.schema_version,
        )
        x = feat_df.fillna(0).to_numpy(dtype=np.float64)

        for i, col in enumerate(feature_columns):
            if col in _NULLABLE_NUMERIC_COLUMNS:
                assert x[0, i] == 0.0, f"{col}: train path produced {x[0, i]}"

    @pytest.mark.parametrize("col", _NULLABLE_NUMERIC_COLUMNS)
    def test_tsp003_per_column_imputation_parity(
        self,
        trained_artifacts,
        valid_feature_row_data,
        col: str,
    ) -> None:
        """TSP-003: each nullable numeric column imputes identically in both paths."""
        data = {**valid_feature_row_data}
        data[col] = None
        row = FeatureRow(**data)

        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)
        feature_columns = get_feature_columns(metadata.schema_version)

        online_val = pred._encode_value(col, getattr(row, col))

        feat_df = _row_frame(
            row,
            feature_columns=feature_columns,
        )
        feat_df, _ = encode_categoricals(
            feat_df,
            cat_encoders,
            schema_version=metadata.schema_version,
        )
        batch_vec = feat_df.fillna(0).to_numpy(dtype=np.float64)
        col_idx = feature_columns.index(col)
        batch_val = batch_vec[0, col_idx]

        assert online_val == batch_val, f"{col}: online={online_val}, batch={batch_val}"


class TestPredictionParity:
    """TSP-002: batch and online raw probabilities match."""

    def test_tsp002_batch_online_proba_match(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """TSP-002: predict_proba (batch) and predict_bucket (online) agree."""
        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        row = FeatureRow(**valid_feature_row_data)
        feature_columns = get_feature_columns(metadata.schema_version)
        df = _row_frame(
            row,
            feature_columns=feature_columns,
        )
        batch_proba = predict_proba(
            model,
            df,
            cat_encoders,
            schema_version=metadata.schema_version,
        )

        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
        )
        wp = pred.predict_bucket(row)
        online_proba = np.array(wp.core_probs)

        np.testing.assert_allclose(
            batch_proba[0],
            online_proba,
            atol=1e-5,
            err_msg="Batch and online probabilities diverge",
        )


class TestPipelineStages:
    """P0-001, P0-002: pipeline stage ordering verification."""

    def test_p0001_batch_feature_vector_matches_online(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """P0-001: batch and online paths produce identical feature vectors."""
        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        row = FeatureRow(**valid_feature_row_data)
        pred = OnlinePredictor(model, metadata, cat_encoders=cat_encoders)
        feature_columns = get_feature_columns(metadata.schema_version)

        online_vec = np.array(
            [pred._encode_value(c, getattr(row, c)) for c in feature_columns],
            dtype=np.float64,
        )

        feat_df = _row_frame(
            row,
            feature_columns=feature_columns,
        )
        feat_df, _ = encode_categoricals(
            feat_df,
            cat_encoders,
            schema_version=metadata.schema_version,
        )
        batch_vec = feat_df.fillna(0).to_numpy(dtype=np.float64)[0]

        np.testing.assert_array_equal(
            online_vec,
            batch_vec,
            err_msg="Feature vectors differ between batch and online paths",
        )

    def test_p0002_online_pipeline_runs_all_stages(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """P0-002: online predict_bucket exercises encode, predict, calibrate, reject, smooth."""
        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        row = FeatureRow(**valid_feature_row_data)
        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=3,
            reject_threshold=0.3,
        )

        wp = pred.predict_bucket(row)

        assert len(wp.core_probs) == len(LABEL_SET_V1)
        np.testing.assert_allclose(sum(wp.core_probs), 1.0, atol=1e-5)
        assert 0.0 <= wp.confidence <= 1.0
        assert isinstance(wp.is_rejected, bool)
        assert len(pred._raw_buffer) == 1
        assert len(pred._bucket_ts_buffer) == 1
