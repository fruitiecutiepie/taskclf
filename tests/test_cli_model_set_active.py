"""Tests for ``taskclf model set-active`` CLI command.

Covers:
- TC-MSA-001: Valid bundle sets active pointer
- TC-MSA-002: Missing model_id exits with error
- TC-MSA-003: Invalid bundle exits with error
- TC-MSA-004: Incompatible bundle exits with error
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1
from taskclf.model_registry import read_active

runner = CliRunner()

_VALID_METADATA = {
    "schema_version": FeatureSchemaV1.VERSION,
    "schema_hash": FeatureSchemaV1.SCHEMA_HASH,
    "label_set": sorted(LABEL_SET_V1),
    "train_date_from": "2026-01-01",
    "train_date_to": "2026-01-31",
    "params": {"num_leaves": 31},
    "git_commit": "abc123",
    "dataset_hash": "a1b2c3d4e5f6g7h8",
    "reject_threshold": 0.35,
    "data_provenance": "real",
    "created_at": "2026-02-01T00:00:00+00:00",
}

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
    metadata: dict | None = None,
    metrics: dict | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if metrics is not None:
        (bundle_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# TC-MSA-001: Valid bundle sets active pointer
# ---------------------------------------------------------------------------


class TestSetActiveValid:
    def test_sets_active_pointer(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "run_001", _VALID_METADATA, _VALID_METRICS)

        result = runner.invoke(app, [
            "model", "set-active",
            "--model-id", "run_001",
            "--models-dir", str(models_dir),
        ])

        assert result.exit_code == 0
        assert "run_001" in result.output

        pointer = read_active(models_dir)
        assert pointer is not None
        assert "run_001" in pointer.model_dir

    def test_creates_history_entry(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _write_bundle(models_dir / "run_001", _VALID_METADATA, _VALID_METRICS)

        runner.invoke(app, [
            "model", "set-active",
            "--model-id", "run_001",
            "--models-dir", str(models_dir),
        ])

        history = models_dir / "active_history.jsonl"
        assert history.is_file()
        lines = history.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert "run_001" in entry["new"]["model_dir"]


# ---------------------------------------------------------------------------
# TC-MSA-002: Missing model_id
# ---------------------------------------------------------------------------


class TestSetActiveMissing:
    def test_missing_bundle_exits_with_error(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        result = runner.invoke(app, [
            "model", "set-active",
            "--model-id", "nonexistent",
            "--models-dir", str(models_dir),
        ])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# TC-MSA-003: Invalid bundle
# ---------------------------------------------------------------------------


class TestSetActiveInvalid:
    def test_invalid_bundle_exits_with_error(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        bundle_dir = models_dir / "broken"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "metadata.json").write_text("not valid json {{{")

        result = runner.invoke(app, [
            "model", "set-active",
            "--model-id", "broken",
            "--models-dir", str(models_dir),
        ])

        assert result.exit_code != 0
        assert "invalid" in result.output.lower()


# ---------------------------------------------------------------------------
# TC-MSA-004: Incompatible bundle
# ---------------------------------------------------------------------------


class TestSetActiveIncompatible:
    def test_incompatible_schema_exits_with_error(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        bad_meta = {**_VALID_METADATA, "schema_hash": "wrong_hash"}
        _write_bundle(models_dir / "old_model", bad_meta, _VALID_METRICS)

        result = runner.invoke(app, [
            "model", "set-active",
            "--model-id", "old_model",
            "--models-dir", str(models_dir),
        ])

        assert result.exit_code != 0
        assert "incompatible" in result.output.lower()
