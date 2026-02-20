"""Tests for model bundle persistence: save, load, schema gate.

Covers: TC-MODEL-001 (write files), TC-MODEL-002 (missing metadata),
TC-MODEL-003 (schema hash mismatch), TC-MODEL-004 (label set mismatch).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil

import pandas as pd
import pytest
from pydantic import ValidationError

from taskclf.core.model_io import (
    build_metadata,
    load_model_bundle,
    save_model_bundle,
)
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.train.dataset import assign_labels_to_buckets, split_by_day
from taskclf.train.lgbm import train_lgbm


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
def trained_bundle(tmp_path_factory: pytest.TempPathFactory):
    """Train a tiny LightGBM model, save as a bundle, return artifacts."""
    labeled = _build_labeled_df()
    train_df, val_df = split_by_day(labeled)

    model, metrics, cm_df, params = train_lgbm(
        train_df, val_df, num_boost_round=5,
    )

    base_dir = tmp_path_factory.mktemp("models")
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dt.date(2025, 6, 14),
        train_date_to=dt.date(2025, 6, 15),
        params=params,
    )
    run_dir = save_model_bundle(model, metadata, metrics, cm_df, base_dir)

    return {
        "model": model,
        "metadata": metadata,
        "metrics": metrics,
        "cm_df": cm_df,
        "run_dir": run_dir,
    }


class TestSaveModelBundle:
    """TC-MODEL-001: save_model_bundle writes all required bundle files."""

    def test_writes_model_file(self, trained_bundle) -> None:
        run_dir = trained_bundle["run_dir"]
        model_path = run_dir / "model.txt"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_writes_metadata_json_with_required_keys(self, trained_bundle) -> None:
        raw = json.loads((trained_bundle["run_dir"] / "metadata.json").read_text())
        for key in ("schema_version", "schema_hash", "label_set",
                     "train_date_from", "train_date_to", "params", "git_commit"):
            assert key in raw, f"metadata.json missing required key: {key}"

    def test_metadata_schema_hash_matches_current(self, trained_bundle) -> None:
        raw = json.loads((trained_bundle["run_dir"] / "metadata.json").read_text())
        assert raw["schema_hash"] == FeatureSchemaV1.SCHEMA_HASH

    def test_writes_metrics_json(self, trained_bundle) -> None:
        raw = json.loads((trained_bundle["run_dir"] / "metrics.json").read_text())
        assert "macro_f1" in raw
        assert "confusion_matrix" in raw

    def test_writes_confusion_matrix_csv(self, trained_bundle) -> None:
        csv_path = trained_bundle["run_dir"] / "confusion_matrix.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path, index_col=0)
        assert df.shape[0] == df.shape[1]

    def test_run_dir_is_immutable_on_rewrite(self, trained_bundle) -> None:
        """Run dirs must not be overwritten (R2 invariant)."""
        run_dir = trained_bundle["run_dir"]
        assert run_dir.is_dir()


class TestLoadModelBundleMissingMetadata:
    """TC-MODEL-002: load fails when metadata.json is missing schema_hash."""

    def test_missing_schema_hash_raises(self, tmp_path, trained_bundle) -> None:
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        shutil.copy(trained_bundle["run_dir"] / "model.txt", run_dir / "model.txt")

        bad_meta = {
            "schema_version": "v1",
            "label_set": ["coding"],
            "train_date_from": "2025-06-14",
            "train_date_to": "2025-06-15",
            "params": {},
            "git_commit": "test",
            "created_at": "2025-06-15T00:00:00",
        }
        (run_dir / "metadata.json").write_text(json.dumps(bad_meta))

        with pytest.raises(ValidationError):
            load_model_bundle(run_dir)


class TestLoadModelBundleSchemaHashMismatch:
    """TC-MODEL-003: load fails when schema hash doesn't match current schema."""

    def test_mismatch_raises_value_error(self, tmp_path, trained_bundle) -> None:
        run_dir = tmp_path / "mismatched_run"
        run_dir.mkdir()
        shutil.copy(trained_bundle["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = trained_bundle["metadata"].model_dump()
        meta_dict["schema_hash"] = "000000000000"
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        with pytest.raises(ValueError, match="Schema hash mismatch"):
            load_model_bundle(run_dir)

    def test_validate_schema_false_skips_check(self, tmp_path, trained_bundle) -> None:
        run_dir = tmp_path / "skip_validation_run"
        run_dir.mkdir()
        shutil.copy(trained_bundle["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = trained_bundle["metadata"].model_dump()
        meta_dict["schema_hash"] = "000000000000"
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        model, loaded_meta = load_model_bundle(run_dir, validate_schema=False)
        assert loaded_meta.schema_hash == "000000000000"

    def test_valid_hash_loads_successfully(self, trained_bundle) -> None:
        model, metadata = load_model_bundle(trained_bundle["run_dir"])
        assert metadata.schema_hash == FeatureSchemaV1.SCHEMA_HASH


class TestLoadModelBundleLabelSetMismatch:
    """TC-MODEL-004: load fails when label set doesn't match (optional strictness)."""

    def test_label_set_mismatch_raises(self, tmp_path, trained_bundle) -> None:
        run_dir = tmp_path / "bad_labels_run"
        run_dir.mkdir()
        shutil.copy(trained_bundle["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = trained_bundle["metadata"].model_dump()
        meta_dict["label_set"] = ["coding", "sleeping", "gaming"]
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        with pytest.raises(ValueError, match="Label set mismatch"):
            load_model_bundle(run_dir, validate_schema=False)

    def test_validate_labels_false_skips_check(self, tmp_path, trained_bundle) -> None:
        run_dir = tmp_path / "skip_label_validation_run"
        run_dir.mkdir()
        shutil.copy(trained_bundle["run_dir"] / "model.txt", run_dir / "model.txt")

        meta_dict = trained_bundle["metadata"].model_dump()
        meta_dict["label_set"] = ["coding", "sleeping"]
        (run_dir / "metadata.json").write_text(json.dumps(meta_dict))

        model, loaded_meta = load_model_bundle(
            run_dir, validate_schema=False, validate_labels=False,
        )
        assert sorted(loaded_meta.label_set) == ["coding", "sleeping"]

    def test_valid_label_set_loads_successfully(self, trained_bundle) -> None:
        model, metadata = load_model_bundle(trained_bundle["run_dir"])
        from taskclf.core.types import LABEL_SET_V1
        assert sorted(metadata.label_set) == sorted(LABEL_SET_V1)
