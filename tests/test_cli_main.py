"""End-to-end CLI tests for taskclf commands.

Covers TC-E2E-001 through TC-E2E-006.  Tests invoke the Typer CLI via
CliRunner in isolated temp directories and verify exit codes, expected file
outputs, schema consistency, and absence of sensitive data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1

runner = CliRunner()

FORBIDDEN_COLUMNS = {"raw_keystrokes", "window_title_raw", "clipboard_content"}


# ---------------------------------------------------------------------------
# TC-E2E-001: taskclf ingest aw
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once `taskclf ingest aw` CLI command is implemented")
def test_tc_e2e_001_ingest_aw() -> None:
    """TC-E2E-001: `taskclf ingest aw` creates data/raw/ artifacts."""


# ---------------------------------------------------------------------------
# TC-E2E-002: taskclf features build
# ---------------------------------------------------------------------------

class TestFeaturesBuild:
    """TC-E2E-002: `taskclf features build` creates processed parquet."""

    def test_exit_code_zero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "features", "build",
            "--date", "2025-06-15",
            "--data-dir", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output

    def test_parquet_file_created(self, tmp_path: Path) -> None:
        runner.invoke(app, [
            "features", "build",
            "--date", "2025-06-15",
            "--data-dir", str(tmp_path),
        ])
        expected = tmp_path / "features_v1" / "date=2025-06-15" / "features.parquet"
        assert expected.exists()

    def test_schema_columns_present(self, tmp_path: Path) -> None:
        runner.invoke(app, [
            "features", "build",
            "--date", "2025-06-15",
            "--data-dir", str(tmp_path),
        ])
        parquet = tmp_path / "features_v1" / "date=2025-06-15" / "features.parquet"
        df = pd.read_parquet(parquet)

        assert "schema_version" in df.columns
        assert "schema_hash" in df.columns
        assert df["schema_version"].iloc[0] == FeatureSchemaV1.VERSION
        assert df["schema_hash"].iloc[0] == FeatureSchemaV1.SCHEMA_HASH

    def test_no_forbidden_columns(self, tmp_path: Path) -> None:
        runner.invoke(app, [
            "features", "build",
            "--date", "2025-06-15",
            "--data-dir", str(tmp_path),
        ])
        parquet = tmp_path / "features_v1" / "date=2025-06-15" / "features.parquet"
        df = pd.read_parquet(parquet)

        leaked = FORBIDDEN_COLUMNS & set(df.columns)
        assert not leaked, f"Forbidden columns in output: {leaked}"


# ---------------------------------------------------------------------------
# TC-E2E-003: taskclf labels import
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once `taskclf labels import` CLI command is implemented")
def test_tc_e2e_003_labels_import() -> None:
    """TC-E2E-003: `taskclf labels import` creates labels parquet."""


# ---------------------------------------------------------------------------
# TC-E2E-004: taskclf train lgbm
# ---------------------------------------------------------------------------

class TestTrainLgbm:
    """TC-E2E-004: `taskclf train lgbm` creates a valid model bundle."""

    @pytest.fixture()
    def train_result(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        result = runner.invoke(app, [
            "train", "lgbm",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--models-dir", str(models_dir),
            "--num-boost-round", "5",
        ])
        return result, models_dir

    def test_exit_code_zero(self, train_result) -> None:
        result, _ = train_result
        assert result.exit_code == 0, result.output

    def test_run_dir_created_with_required_files(self, train_result) -> None:
        _, models_dir = train_result
        run_dirs = list(models_dir.iterdir())
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        for name in ("model.txt", "metadata.json", "metrics.json", "confusion_matrix.csv"):
            assert (run_dir / name).exists(), f"Missing: {name}"

    def test_metadata_schema_valid(self, train_result) -> None:
        _, models_dir = train_result
        run_dir = next(models_dir.iterdir())
        meta = json.loads((run_dir / "metadata.json").read_text())

        assert meta["schema_version"] == FeatureSchemaV1.VERSION
        assert meta["schema_hash"] == FeatureSchemaV1.SCHEMA_HASH
        assert sorted(meta["label_set"]) == sorted(LABEL_SET_V1)

    def test_metrics_contain_macro_f1(self, train_result) -> None:
        _, models_dir = train_result
        run_dir = next(models_dir.iterdir())
        metrics = json.loads((run_dir / "metrics.json").read_text())

        assert "macro_f1" in metrics
        assert isinstance(metrics["macro_f1"], float)

    def test_no_sensitive_fields_in_metrics(self, train_result) -> None:
        _, models_dir = train_result
        run_dir = next(models_dir.iterdir())
        raw = (run_dir / "metrics.json").read_text()

        for forbidden in ("raw_keystrokes", "window_title_raw", "clipboard"):
            assert forbidden not in raw, f"Sensitive field {forbidden!r} found in metrics"


# ---------------------------------------------------------------------------
# TC-E2E-005: taskclf infer batch
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once `taskclf infer batch` CLI command is implemented")
def test_tc_e2e_005_infer_batch() -> None:
    """TC-E2E-005: `taskclf infer batch` creates predictions and segments."""


# ---------------------------------------------------------------------------
# TC-E2E-006: taskclf report daily
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once `taskclf report daily` CLI command is implemented")
def test_tc_e2e_006_report_daily() -> None:
    """TC-E2E-006: `taskclf report daily` creates report outputs."""
