"""Integration tests: train -> model bundle -> infer with schema gating.

Covers: TC-INT-020 (train produces bundle), TC-INT-021 (inference on same
schema succeeds), TC-INT-022 (schema alteration causes inference to refuse).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from taskclf.core.model_io import build_metadata, load_model_bundle, save_model_bundle
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.train.dataset import assign_labels_to_buckets, split_by_day
from taskclf.train.lgbm import FEATURE_COLUMNS, train_lgbm


def _build_labeled_df() -> pd.DataFrame:
    """Build a small labeled feature DataFrame spanning two days."""
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=20))

    features_df = pd.DataFrame([r.model_dump() for r in all_rows])

    spans: list[LabelSpan] = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day)
        spans.extend([
            LabelSpan(start_ts=base.replace(hour=9), end_ts=base.replace(hour=12),
                       label="coding", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=12), end_ts=base.replace(hour=14),
                       label="writing_docs", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=14), end_ts=base.replace(hour=16),
                       label="messaging_email", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=16), end_ts=base.replace(hour=17),
                       label="break_idle", provenance="test"),
        ])

    return assign_labels_to_buckets(features_df, spans)


@pytest.fixture(scope="module")
def pipeline_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """Run the full train -> save pipeline once and return all artifacts."""
    labeled = _build_labeled_df()
    train_df, val_df = split_by_day(labeled)

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=5,
    )

    base_dir = tmp_path_factory.mktemp("integration_models")
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dt.date(2025, 6, 14),
        train_date_to=dt.date(2025, 6, 15),
        params=params,
    )
    run_dir = save_model_bundle(model, metadata, metrics, cm_df, base_dir, cat_encoders=cat_encoders)

    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))

    return {
        "model": model,
        "metadata": metadata,
        "metrics": metrics,
        "cm_df": cm_df,
        "run_dir": run_dir,
        "val_df": val_df,
        "label_encoder": le,
    }


class TestTrainProducesBundle:
    """TC-INT-020: training on fixture data produces a valid model bundle."""

    def test_run_dir_contains_required_files(self, pipeline_artifacts) -> None:
        run_dir = pipeline_artifacts["run_dir"]
        for name in ("model.txt", "metadata.json", "metrics.json", "confusion_matrix.csv"):
            assert (run_dir / name).exists(), f"Missing required file: {name}"

    def test_metadata_schema_matches_current(self, pipeline_artifacts) -> None:
        raw = json.loads((pipeline_artifacts["run_dir"] / "metadata.json").read_text())
        assert raw["schema_version"] == FeatureSchemaV1.VERSION
        assert raw["schema_hash"] == FeatureSchemaV1.SCHEMA_HASH

    def test_metadata_label_set_matches_current(self, pipeline_artifacts) -> None:
        raw = json.loads((pipeline_artifacts["run_dir"] / "metadata.json").read_text())
        assert sorted(raw["label_set"]) == sorted(LABEL_SET_V1)

    def test_metrics_contain_expected_keys(self, pipeline_artifacts) -> None:
        raw = json.loads((pipeline_artifacts["run_dir"] / "metrics.json").read_text())
        assert "macro_f1" in raw
        assert "confusion_matrix" in raw
        assert "label_names" in raw

    def test_confusion_matrix_csv_is_square(self, pipeline_artifacts) -> None:
        df = pd.read_csv(pipeline_artifacts["run_dir"] / "confusion_matrix.csv", index_col=0)
        assert df.shape[0] == df.shape[1]


class TestInferOnSameSchema:
    """TC-INT-021: load bundle and predict on same-schema features."""

    def test_load_and_predict_produces_valid_labels(self, pipeline_artifacts) -> None:
        from taskclf.train.lgbm import encode_categoricals

        model, metadata, cat_encoders = load_model_bundle(pipeline_artifacts["run_dir"])
        val_df = pipeline_artifacts["val_df"]
        le = pipeline_artifacts["label_encoder"]

        feat_df, _ = encode_categoricals(val_df[FEATURE_COLUMNS].copy(), cat_encoders)
        x = feat_df.fillna(0).to_numpy(dtype=np.float64)
        proba = model.predict(x)
        pred_indices = proba.argmax(axis=1)
        pred_labels = le.inverse_transform(pred_indices)

        valid_labels = set(LABEL_SET_V1)
        for label in pred_labels:
            assert label in valid_labels, f"Prediction {label!r} not in LABEL_SET_V1"

    def test_prediction_count_matches_input_rows(self, pipeline_artifacts) -> None:
        from taskclf.train.lgbm import encode_categoricals

        model, _, cat_encoders = load_model_bundle(pipeline_artifacts["run_dir"])
        val_df = pipeline_artifacts["val_df"]

        feat_df, _ = encode_categoricals(val_df[FEATURE_COLUMNS].copy(), cat_encoders)
        x = feat_df.fillna(0).to_numpy(dtype=np.float64)
        proba = model.predict(x)
        assert proba.shape[0] == len(val_df)

    def test_probabilities_sum_to_one(self, pipeline_artifacts) -> None:
        from taskclf.train.lgbm import encode_categoricals

        model, _, cat_encoders = load_model_bundle(pipeline_artifacts["run_dir"])
        val_df = pipeline_artifacts["val_df"]

        feat_df, _ = encode_categoricals(val_df[FEATURE_COLUMNS].copy(), cat_encoders)
        x = feat_df.fillna(0).to_numpy(dtype=np.float64)
        proba = model.predict(x)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestSchemaAlterationRefusesInference:
    """TC-INT-022: altered schema hash causes load_model_bundle to refuse."""

    def test_tampered_schema_hash_raises(self, tmp_path, pipeline_artifacts) -> None:
        run_dir = tmp_path / "tampered_schema"
        run_dir.mkdir()
        shutil.copy(pipeline_artifacts["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = pipeline_artifacts["metadata"].model_dump()
        meta_dict["schema_hash"] = "ffffffffffff"
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        with pytest.raises(ValueError, match="Schema hash mismatch"):
            load_model_bundle(run_dir)

    def test_tampered_label_set_raises(self, tmp_path, pipeline_artifacts) -> None:
        run_dir = tmp_path / "tampered_labels"
        run_dir.mkdir()
        shutil.copy(pipeline_artifacts["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = pipeline_artifacts["metadata"].model_dump()
        meta_dict["label_set"] = ["coding", "unknown_task"]
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        with pytest.raises(ValueError, match="Label set mismatch"):
            load_model_bundle(run_dir, validate_schema=False)

    def test_both_validations_pass_on_valid_bundle(self, pipeline_artifacts) -> None:
        model, metadata, _ = load_model_bundle(pipeline_artifacts["run_dir"])
        assert metadata.schema_hash == FeatureSchemaV1.SCHEMA_HASH
        assert sorted(metadata.label_set) == sorted(LABEL_SET_V1)
