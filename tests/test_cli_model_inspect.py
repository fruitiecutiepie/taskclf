"""Tests for ``taskclf model inspect`` CLI command.

Covers:
- TC-MI-001: bundle-only inspect exits 0 and shows validation metrics section
- TC-MI-002: --json emits parseable JSON with expected keys
- TC-MI-003: replay with --synthetic includes replayed_test_evaluation
- TC-MI-004: only one of --from/--to fails
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from taskclf.cli.main import app

runner = CliRunner()


def _train_model(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    result = runner.invoke(
        app,
        [
            "train",
            "lgbm",
            "--from",
            "2025-06-14",
            "--to",
            "2025-06-15",
            "--synthetic",
            "--models-dir",
            str(models_dir),
            "--num-boost-round",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    return next(models_dir.iterdir())


class TestModelInspect:
    def test_bundle_only_human(self, tmp_path: Path) -> None:
        """TC-MI-001: human output includes bundle and validation sections."""
        model_dir = _train_model(tmp_path)
        result = runner.invoke(
            app,
            [
                "model",
                "inspect",
                "--model-dir",
                str(model_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Bundle-saved validation metrics" in result.output
        assert "Prediction / metrics logic" in result.output
        assert "Tip:" in result.output

    def test_bundle_only_json(self, tmp_path: Path) -> None:
        """TC-MI-002: --json is valid and includes bundle_saved_validation."""
        model_dir = _train_model(tmp_path)
        result = runner.invoke(
            app,
            [
                "model",
                "inspect",
                "--model-dir",
                str(model_dir),
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["bundle_path"]
        assert data["bundle_saved_validation"]["macro_f1"] is not None
        assert "top_confusion_pairs" in data["bundle_saved_validation"]
        assert data["prediction_logic"]["problem_type"] == "multiclass"
        assert data["replayed_test_evaluation"] is None

    def test_replay_synthetic_json(self, tmp_path: Path) -> None:
        """TC-MI-003: synthetic replay populates replayed_test_evaluation."""
        model_dir = _train_model(tmp_path)
        result = runner.invoke(
            app,
            [
                "model",
                "inspect",
                "--model-dir",
                str(model_dir),
                "--from",
                "2025-06-15",
                "--to",
                "2025-06-15",
                "--synthetic",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["replayed_test_evaluation"] is not None
        assert data["replayed_test_evaluation"]["test_row_count"] > 0
        assert data["replay_error"] is None
        rep = data["replayed_test_evaluation"]["report"]
        assert "expected_calibration_error" in rep
        assert "slice_metrics" in rep
        assert "unknown_category_rates" in rep

    def test_from_without_to_errors(self, tmp_path: Path) -> None:
        """TC-MI-004: mismatched date options exit non-zero."""
        model_dir = _train_model(tmp_path)
        result = runner.invoke(
            app,
            [
                "model",
                "inspect",
                "--model-dir",
                str(model_dir),
                "--from",
                "2025-06-15",
            ],
        )
        assert result.exit_code != 0
