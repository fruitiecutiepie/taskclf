"""Tests for the inference-policy artifact.

Covers: round-trip persistence, atomic write, validation against disk
artifacts, builder defaults, load from missing/invalid files, removal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskclf.core.defaults import DEFAULT_REJECT_THRESHOLD
from taskclf.core.inference_policy import (
    InferencePolicy,
    PolicyValidationError,
    build_inference_policy,
    load_inference_policy,
    remove_inference_policy,
    render_default_inference_policy_template_json,
    save_inference_policy,
    validate_policy,
    write_inference_policy_starter_template,
)
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1


def _write_fake_bundle(models_dir: Path) -> Path:
    """Create a minimal model bundle directory for validation tests."""
    run_dir = models_dir / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": FeatureSchemaV1.VERSION,
        "schema_hash": FeatureSchemaV1.SCHEMA_HASH,
        "label_set": sorted(LABEL_SET_V1),
        "train_date_from": "2026-01-01",
        "train_date_to": "2026-01-31",
        "params": {},
        "git_commit": "abc123",
        "dataset_hash": "d4e5f6",
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def _write_fake_calibrator_store(
    base: Path,
    *,
    model_schema_hash: str | None = None,
    model_bundle_id: str | None = None,
) -> Path:
    store_dir = base / "calibrator_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {"method": "temperature", "user_count": 0, "user_ids": []}
    if model_schema_hash is not None:
        meta["model_schema_hash"] = model_schema_hash
    if model_bundle_id is not None:
        meta["model_bundle_id"] = model_bundle_id
    (store_dir / "store.json").write_text(json.dumps(meta))
    (store_dir / "global.json").write_text(json.dumps({"type": "identity"}))
    return store_dir


class TestInferencePolicyRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash="abc123",
            model_label_set=["A", "B"],
            reject_threshold=0.6,
            source="manual",
        )
        save_inference_policy(policy, models_dir)
        loaded = load_inference_policy(models_dir)
        assert loaded is not None
        assert loaded.model_dir == "models/run_001"
        assert loaded.model_schema_hash == "abc123"
        assert loaded.reject_threshold == 0.6
        assert loaded.source == "manual"
        assert sorted(loaded.model_label_set) == ["A", "B"]

    def test_atomic_write_no_tmp_left(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        policy = InferencePolicy(
            model_dir="m/r",
            model_schema_hash="h",
            model_label_set=["X"],
            reject_threshold=0.5,
        )
        save_inference_policy(policy, models_dir)
        assert not (models_dir / ".inference_policy.json.tmp").exists()
        assert (models_dir / "inference_policy.json").exists()

    def test_overwrite(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        p1 = InferencePolicy(
            model_dir="m/old",
            model_schema_hash="h1",
            model_label_set=["A"],
            reject_threshold=0.3,
        )
        save_inference_policy(p1, models_dir)
        p2 = InferencePolicy(
            model_dir="m/new",
            model_schema_hash="h2",
            model_label_set=["B"],
            reject_threshold=0.7,
        )
        save_inference_policy(p2, models_dir)
        loaded = load_inference_policy(models_dir)
        assert loaded is not None
        assert loaded.model_dir == "m/new"
        assert loaded.reject_threshold == 0.7


class TestLoadInferencePolicy:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert load_inference_policy(tmp_path) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "inference_policy.json").write_text("{bad json!!!")
        assert load_inference_policy(tmp_path) is None

    def test_missing_required_field(self, tmp_path: Path) -> None:
        (tmp_path / "inference_policy.json").write_text(json.dumps({"model_dir": "x"}))
        assert load_inference_policy(tmp_path) is None


class TestRemoveInferencePolicy:
    def test_remove_existing(self, tmp_path: Path) -> None:
        (tmp_path / "inference_policy.json").write_text("{}")
        assert remove_inference_policy(tmp_path) is True
        assert not (tmp_path / "inference_policy.json").exists()

    def test_remove_missing(self, tmp_path: Path) -> None:
        assert remove_inference_policy(tmp_path) is False


class TestBuildInferencePolicy:
    def test_default_fields(self) -> None:
        policy = build_inference_policy(
            model_dir="models/run_001",
            model_schema_hash="hash123",
            model_label_set=["C", "A", "B"],
            reject_threshold=0.42,
            source="tune-reject",
        )
        assert policy.policy_version == "v1"
        assert policy.reject_threshold == 0.42
        assert policy.source == "tune-reject"
        assert policy.model_label_set == ["A", "B", "C"]
        assert policy.created_at  # non-empty
        assert policy.calibrator_store_dir is None
        assert policy.calibration_method is None

    def test_with_calibrator(self) -> None:
        policy = build_inference_policy(
            model_dir="models/run_001",
            model_schema_hash="h",
            model_label_set=["X"],
            calibrator_store_dir="artifacts/cal_store",
            calibration_method="isotonic",
            source="calibrate",
        )
        assert policy.calibrator_store_dir == "artifacts/cal_store"
        assert policy.calibration_method == "isotonic"

    def test_default_threshold(self) -> None:
        policy = build_inference_policy(
            model_dir="m/r",
            model_schema_hash="h",
            model_label_set=["A"],
        )
        assert policy.reject_threshold == DEFAULT_REJECT_THRESHOLD


def test_inference_policy_template_file_matches_render() -> None:
    """configs/inference_policy.template.json matches render output."""
    root = Path(__file__).resolve().parents[1]
    on_disk = (root / "configs" / "inference_policy.template.json").read_text()
    assert on_disk == render_default_inference_policy_template_json()


class TestWriteInferencePolicyStarterTemplate:
    def test_writes_valid_policy_and_paths_help(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        path = write_inference_policy_starter_template(models_dir)
        assert path == models_dir / "inference_policy.json"
        loaded = load_inference_policy(models_dir)
        assert loaded is not None
        assert loaded.source == "starter-template"
        raw = json.loads(path.read_text())
        assert raw["_help"]["paths_are_relative_to"] == str(tmp_path)
        assert "canonical_template" in raw["_help"]
        assert not (models_dir / ".inference_policy.starter.tmp").exists()


class TestValidatePolicy:
    def test_valid_policy_no_calibrator(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        run_dir = _write_fake_bundle(models_dir)
        policy = InferencePolicy(
            model_dir=str(run_dir.relative_to(tmp_path)),
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.55,
        )
        validate_policy(policy, models_dir)

    def test_missing_model_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        policy = InferencePolicy(
            model_dir="models/nonexistent",
            model_schema_hash="h",
            model_label_set=["A"],
            reject_threshold=0.5,
        )
        with pytest.raises(PolicyValidationError, match="does not exist"):
            validate_policy(policy, models_dir)

    def test_schema_hash_mismatch(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash="wrong_hash",
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.5,
        )
        with pytest.raises(PolicyValidationError, match="Schema hash mismatch"):
            validate_policy(policy, models_dir)

    def test_label_set_mismatch(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=["WrongLabel"],
            reject_threshold=0.5,
        )
        with pytest.raises(PolicyValidationError, match="Label set mismatch"):
            validate_policy(policy, models_dir)

    def test_missing_calibrator_store(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.5,
            calibrator_store_dir="artifacts/nonexistent",
        )
        with pytest.raises(PolicyValidationError, match="does not exist"):
            validate_policy(policy, models_dir)

    def test_calibrator_store_schema_mismatch(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        _write_fake_calibrator_store(
            tmp_path,
            model_schema_hash="different_hash",
        )
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.5,
            calibrator_store_dir="calibrator_store",
        )
        with pytest.raises(PolicyValidationError, match="different schema"):
            validate_policy(policy, models_dir)

    def test_calibrator_store_no_binding_passes(self, tmp_path: Path) -> None:
        """Old stores without model binding should still pass validation."""
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        _write_fake_calibrator_store(tmp_path)
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.5,
            calibrator_store_dir="calibrator_store",
        )
        validate_policy(policy, models_dir)

    def test_calibrator_store_matching_binding(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_fake_bundle(models_dir)
        _write_fake_calibrator_store(
            tmp_path,
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_bundle_id="run_001",
        )
        policy = InferencePolicy(
            model_dir="models/run_001",
            model_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            model_label_set=sorted(LABEL_SET_V1),
            reject_threshold=0.5,
            calibrator_store_dir="calibrator_store",
        )
        validate_policy(policy, models_dir)
