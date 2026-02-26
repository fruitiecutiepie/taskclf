"""Integration tests for model resolution in inference.

Covers:
- TC-RES-001: No model-dir and active exists → uses active
- TC-RES-002: No active and models exist → picks best
- TC-RES-003: Active points to missing → falls back
- TC-RES-004: Explicit --model-dir overrides everything
- TC-RES-005: No models at all → raises ModelResolutionError
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskclf.core.model_io import ModelMetadata
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.resolve import ActiveModelReloader, ModelResolutionError, resolve_model_dir
from taskclf.model_registry import (
    ActivePointer,
    SelectionPolicy,
    read_active,
    write_active_atomic,
    list_bundles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_METADATA = ModelMetadata(
    schema_version=FeatureSchemaV1.VERSION,
    schema_hash=FeatureSchemaV1.SCHEMA_HASH,
    label_set=sorted(LABEL_SET_V1),
    train_date_from="2026-01-01",
    train_date_to="2026-01-31",
    params={"num_leaves": 31, "learning_rate": 0.05},
    git_commit="abc123",
    dataset_hash="a1b2c3d4e5f6g7h8",
    reject_threshold=0.35,
    data_provenance="real",
    created_at="2026-02-01T00:00:00+00:00",
)

_VALID_METRICS = {
    "macro_f1": 0.82,
    "weighted_f1": 0.85,
    "confusion_matrix": [
        [45, 2, 1, 0, 0, 1, 0, 1],
        [3, 40, 2, 1, 0, 0, 0, 4],
        [1, 3, 38, 2, 1, 0, 0, 5],
        [0, 1, 2, 42, 1, 1, 0, 3],
        [0, 0, 1, 1, 35, 2, 0, 1],
        [1, 0, 0, 1, 2, 40, 1, 5],
        [0, 0, 0, 0, 0, 1, 48, 1],
        [1, 2, 1, 0, 0, 2, 0, 44],
    ],
    "label_names": sorted(LABEL_SET_V1),
}


def _write_bundle(
    bundle_dir: Path,
    metadata: ModelMetadata | dict | None = None,
    metrics: dict | str | None = None,
) -> None:
    """Write metadata.json and/or metrics.json into *bundle_dir*."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if metadata is not None:
        meta_data = (
            metadata.model_dump() if isinstance(metadata, ModelMetadata) else metadata
        )
        (bundle_dir / "metadata.json").write_text(json.dumps(meta_data, indent=2))

    if metrics is not None:
        content = metrics if isinstance(metrics, str) else json.dumps(metrics, indent=2)
        (bundle_dir / "metrics.json").write_text(content)


# ---------------------------------------------------------------------------
# TC-RES-001: No model-dir and active exists → uses active
# ---------------------------------------------------------------------------


class TestActiveExists:
    def test_uses_active_pointer_not_best(self, tmp_path: Path) -> None:
        """When active.json exists and is valid, return its bundle even if
        a higher-scoring bundle is available."""
        models_dir = tmp_path / "models"
        low_metrics = {**_VALID_METRICS, "macro_f1": 0.60}
        high_metrics = {**_VALID_METRICS, "macro_f1": 0.95}
        _write_bundle(models_dir / "low_model", _VALID_METADATA, low_metrics)
        _write_bundle(models_dir / "high_model", _VALID_METADATA, high_metrics)

        low_bundle = [b for b in list_bundles(models_dir) if b.model_id == "low_model"][0]
        write_active_atomic(models_dir, low_bundle, SelectionPolicy())

        resolved = resolve_model_dir(None, models_dir)

        assert resolved == models_dir / "low_model"


# ---------------------------------------------------------------------------
# TC-RES-002: No active and models exist → picks best
# ---------------------------------------------------------------------------


class TestNoActivePicksBest:
    def test_selects_best_by_macro_f1(self, tmp_path: Path) -> None:
        """Without active.json, resolve picks the bundle with highest macro_f1."""
        models_dir = tmp_path / "models"
        low_metrics = {**_VALID_METRICS, "macro_f1": 0.60}
        high_metrics = {**_VALID_METRICS, "macro_f1": 0.95}
        _write_bundle(models_dir / "low_model", _VALID_METADATA, low_metrics)
        _write_bundle(models_dir / "high_model", _VALID_METADATA, high_metrics)

        resolved = resolve_model_dir(None, models_dir)

        assert resolved == models_dir / "high_model"

    def test_self_heals_active_json(self, tmp_path: Path) -> None:
        """After fallback selection, active.json should be written."""
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "only_model", _VALID_METADATA, _VALID_METRICS)

        resolve_model_dir(None, models_dir)

        pointer = read_active(models_dir)
        assert pointer is not None
        assert "only_model" in pointer.model_dir


# ---------------------------------------------------------------------------
# TC-RES-003: Active points to missing → falls back
# ---------------------------------------------------------------------------


class TestActivePointsToMissing:
    def test_fallback_when_active_dir_missing(self, tmp_path: Path) -> None:
        """When active.json points to a directory that no longer exists,
        fall back to selection and repair the pointer."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)

        stale = ActivePointer(
            model_dir="models/deleted_bundle",
            selected_at="2026-01-01T00:00:00+00:00",
            policy_version=1,
        )
        (models_dir / "active.json").write_text(stale.model_dump_json(indent=2))

        _write_bundle(models_dir / "surviving_model", _VALID_METADATA, _VALID_METRICS)

        resolved = resolve_model_dir(None, models_dir)

        assert resolved == models_dir / "surviving_model"

        repaired = read_active(models_dir)
        assert repaired is not None
        assert "surviving_model" in repaired.model_dir


# ---------------------------------------------------------------------------
# TC-RES-004: Explicit --model-dir overrides everything
# ---------------------------------------------------------------------------


class TestExplicitOverride:
    def test_explicit_model_dir_ignores_active(self, tmp_path: Path) -> None:
        """An explicit --model-dir always wins, even if active.json exists."""
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "active_model", _VALID_METADATA, _VALID_METRICS)
        active_bundle = list_bundles(models_dir)[0]
        write_active_atomic(models_dir, active_bundle, SelectionPolicy())

        custom_dir = tmp_path / "custom_model"
        custom_dir.mkdir()

        resolved = resolve_model_dir(str(custom_dir), models_dir)

        assert resolved == custom_dir

    def test_explicit_model_dir_nonexistent_raises(self, tmp_path: Path) -> None:
        """An explicit --model-dir that doesn't exist should raise."""
        with pytest.raises(ModelResolutionError, match="does not exist"):
            resolve_model_dir(str(tmp_path / "nonexistent"), tmp_path / "models")


# ---------------------------------------------------------------------------
# TC-RES-005: No models at all → raises error
# ---------------------------------------------------------------------------


class TestNoModels:
    def test_empty_models_dir_raises(self, tmp_path: Path) -> None:
        """Empty models/ directory with no active.json should raise
        with a descriptive message."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(ModelResolutionError, match="No eligible model found"):
            resolve_model_dir(None, models_dir)

    def test_models_dir_missing_raises(self, tmp_path: Path) -> None:
        """Non-existent models/ directory should raise with a descriptive message."""
        with pytest.raises(ModelResolutionError, match="Models directory does not exist"):
            resolve_model_dir(None, tmp_path / "nonexistent_models")

    def test_only_incompatible_bundles_raises_with_reasons(self, tmp_path: Path) -> None:
        """When all bundles are incompatible, error message includes exclusion reasons."""
        models_dir = tmp_path / "models"
        bad_meta = _VALID_METADATA.model_copy(update={"schema_hash": "wrong_hash"})
        _write_bundle(models_dir / "bad_bundle", bad_meta, _VALID_METRICS)

        with pytest.raises(ModelResolutionError, match="Excluded bundles") as exc_info:
            resolve_model_dir(None, models_dir)
        assert exc_info.value.report is not None
        assert len(exc_info.value.report.excluded) == 1


# ---------------------------------------------------------------------------
# ActiveModelReloader
# ---------------------------------------------------------------------------


class TestActiveModelReloader:
    def test_no_reload_when_mtime_unchanged(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "bundle_a", _VALID_METADATA, _VALID_METRICS)
        active_bundle = list_bundles(models_dir)[0]
        write_active_atomic(models_dir, active_bundle, SelectionPolicy())

        reloader = ActiveModelReloader(models_dir, check_interval_s=0)
        assert reloader.check_reload() is None

    def test_no_reload_before_interval(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "bundle_a", _VALID_METADATA, _VALID_METRICS)

        reloader = ActiveModelReloader(models_dir, check_interval_s=9999)
        assert reloader.check_reload() is None
