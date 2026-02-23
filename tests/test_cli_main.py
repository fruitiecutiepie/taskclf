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

class TestIngestAW:
    """TC-E2E-001: `taskclf ingest aw` creates data/raw/ artifacts."""

    @pytest.fixture()
    def aw_export_file(self, tmp_path: Path) -> Path:
        export = {
            "buckets": {
                "aw-watcher-window_testhost": {
                    "id": "aw-watcher-window_testhost",
                    "type": "currentwindow",
                    "client": "aw-watcher-window",
                    "hostname": "testhost",
                    "created": "2026-01-01T00:00:00.000000",
                    "events": [
                        {"timestamp": "2026-02-23T10:00:00Z", "duration": 30.0,
                         "data": {"app": "Firefox", "title": "GitHub"}},
                        {"timestamp": "2026-02-23T10:01:00Z", "duration": 45.0,
                         "data": {"app": "Code", "title": "main.py"}},
                        {"timestamp": "2026-02-24T09:00:00Z", "duration": 20.0,
                         "data": {"app": "Terminal", "title": "bash"}},
                    ],
                },
                "aw-watcher-input_testhost": {
                    "id": "aw-watcher-input_testhost",
                    "type": "os.hid.input",
                    "client": "aw-watcher-input",
                    "hostname": "testhost",
                    "created": "2026-01-01T00:00:00.000000",
                    "events": [
                        {"timestamp": "2026-02-23T10:00:00Z", "duration": 5.0,
                         "data": {"presses": 12, "clicks": 3, "deltaX": 100,
                                  "deltaY": 50, "scrollX": 0, "scrollY": 2}},
                        {"timestamp": "2026-02-23T10:00:05Z", "duration": 5.0,
                         "data": {"presses": 8, "clicks": 1, "deltaX": 80,
                                  "deltaY": 30, "scrollX": 0, "scrollY": 0}},
                    ],
                },
            }
        }
        f = tmp_path / "aw-export.json"
        f.write_text(json.dumps(export))
        return f

    def test_exit_code_zero(self, tmp_path: Path, aw_export_file: Path) -> None:
        out_dir = tmp_path / "raw_aw"
        result = runner.invoke(app, [
            "ingest", "aw",
            "--input", str(aw_export_file),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output

    def test_parquet_files_created_per_day(self, tmp_path: Path, aw_export_file: Path) -> None:
        out_dir = tmp_path / "raw_aw"
        runner.invoke(app, [
            "ingest", "aw",
            "--input", str(aw_export_file),
            "--out-dir", str(out_dir),
        ])
        assert (out_dir / "2026-02-23" / "events.parquet").exists()
        assert (out_dir / "2026-02-24" / "events.parquet").exists()

    def test_no_raw_titles_in_output(self, tmp_path: Path, aw_export_file: Path) -> None:
        out_dir = tmp_path / "raw_aw"
        runner.invoke(app, [
            "ingest", "aw",
            "--input", str(aw_export_file),
            "--out-dir", str(out_dir),
        ])
        df = pd.read_parquet(out_dir / "2026-02-23" / "events.parquet")
        assert "window_title_hash" in df.columns
        for val in df["window_title_hash"]:
            assert "GitHub" not in str(val)
            assert "main.py" not in str(val)

    def test_input_events_ingested(self, tmp_path: Path, aw_export_file: Path) -> None:
        out_dir = tmp_path / "raw_aw"
        result = runner.invoke(app, [
            "ingest", "aw",
            "--input", str(aw_export_file),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        input_parquet = out_dir.parent / "aw-input" / "2026-02-23" / "events.parquet"
        assert input_parquet.exists()
        df = pd.read_parquet(input_parquet)
        assert len(df) == 2
        assert "presses" in df.columns
        assert "clicks" in df.columns

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "ingest", "aw",
            "--input", str(tmp_path / "nonexistent.json"),
            "--out-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 1


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

class TestLabelsImport:
    """TC-E2E-003: `taskclf labels import` creates labels parquet."""

    @pytest.fixture()
    def labels_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(
            "start_ts,end_ts,label,provenance\n"
            "2025-06-15 09:00:00,2025-06-15 10:00:00,coding,manual\n"
            "2025-06-15 10:00:00,2025-06-15 11:00:00,writing_docs,manual\n"
            "2025-06-15 11:00:00,2025-06-15 12:00:00,browsing_research,manual\n"
        )
        return csv_path

    def test_exit_code_zero(self, tmp_path: Path, labels_csv: Path) -> None:
        data_dir = tmp_path / "processed"
        result = runner.invoke(app, [
            "labels", "import",
            "--file", str(labels_csv),
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output

    def test_parquet_file_created(self, tmp_path: Path, labels_csv: Path) -> None:
        data_dir = tmp_path / "processed"
        runner.invoke(app, [
            "labels", "import",
            "--file", str(labels_csv),
            "--data-dir", str(data_dir),
        ])
        expected = data_dir / "labels_v1" / "labels.parquet"
        assert expected.exists()

    def test_round_trip_preserves_spans(self, tmp_path: Path, labels_csv: Path) -> None:
        from taskclf.labels.store import read_label_spans

        data_dir = tmp_path / "processed"
        runner.invoke(app, [
            "labels", "import",
            "--file", str(labels_csv),
            "--data-dir", str(data_dir),
        ])
        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert len(spans) == 3
        assert spans[0].label == "coding"
        assert spans[1].label == "writing_docs"
        assert spans[2].label == "browsing_research"

    def test_no_forbidden_columns_in_parquet(self, tmp_path: Path, labels_csv: Path) -> None:
        data_dir = tmp_path / "processed"
        runner.invoke(app, [
            "labels", "import",
            "--file", str(labels_csv),
            "--data-dir", str(data_dir),
        ])
        df = pd.read_parquet(data_dir / "labels_v1" / "labels.parquet")
        leaked = FORBIDDEN_COLUMNS & set(df.columns)
        assert not leaked, f"Forbidden columns in output: {leaked}"


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

class TestInferBatch:
    """TC-E2E-005: `taskclf infer batch` creates predictions and segments."""

    @pytest.fixture()
    def trained_model_dir(self, tmp_path: Path) -> Path:
        """Train a model first so we have a bundle to infer with."""
        models_dir = tmp_path / "models"
        result = runner.invoke(app, [
            "train", "lgbm",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--models-dir", str(models_dir),
            "--num-boost-round", "5",
        ])
        assert result.exit_code == 0, result.output
        return next(models_dir.iterdir())

    def test_exit_code_zero(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output

    def test_predictions_csv_created(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert (out_dir / "predictions.csv").exists()

    def test_segments_json_created(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert (out_dir / "segments.json").exists()

    def test_predictions_have_expected_columns(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        df = pd.read_csv(out_dir / "predictions.csv")
        assert "bucket_start_ts" in df.columns
        assert "predicted_label" in df.columns

    def test_predicted_labels_are_valid(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        df = pd.read_csv(out_dir / "predictions.csv")
        invalid = set(df["predicted_label"].unique()) - set(LABEL_SET_V1)
        assert not invalid, f"Invalid predicted labels: {invalid}"

    def test_segments_are_ordered_and_nonoverlapping(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        segments = json.loads((out_dir / "segments.json").read_text())
        assert len(segments) > 0
        for i in range(len(segments) - 1):
            assert segments[i]["end_ts"] <= segments[i + 1]["start_ts"]

    def test_no_sensitive_fields_in_outputs(self, tmp_path: Path, trained_model_dir: Path) -> None:
        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        for fname in ("predictions.csv", "segments.json"):
            raw = (out_dir / fname).read_text()
            for forbidden in ("raw_keystrokes", "window_title_raw", "clipboard"):
                assert forbidden not in raw, f"Sensitive field {forbidden!r} in {fname}"


# ---------------------------------------------------------------------------
# TC-E2E-006: taskclf report daily
# ---------------------------------------------------------------------------

class TestReportDaily:
    """TC-E2E-006: `taskclf report daily` creates report outputs."""

    @pytest.fixture()
    def segments_file(self, tmp_path: Path) -> Path:
        """Train + infer to produce a segments.json, then return its path."""
        models_dir = tmp_path / "models"
        runner.invoke(app, [
            "train", "lgbm",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--models-dir", str(models_dir),
            "--num-boost-round", "5",
        ])
        model_dir = next(models_dir.iterdir())

        out_dir = tmp_path / "artifacts"
        runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        return out_dir / "segments.json"

    def test_exit_code_zero(self, tmp_path: Path, segments_file: Path) -> None:
        out_dir = tmp_path / "reports"
        result = runner.invoke(app, [
            "report", "daily",
            "--segments-file", str(segments_file),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output

    def test_report_json_created(self, tmp_path: Path, segments_file: Path) -> None:
        out_dir = tmp_path / "reports"
        runner.invoke(app, [
            "report", "daily",
            "--segments-file", str(segments_file),
            "--out-dir", str(out_dir),
        ])
        report_files = list(out_dir.glob("report_*.json"))
        assert len(report_files) == 1

    def test_report_contains_expected_fields(self, tmp_path: Path, segments_file: Path) -> None:
        out_dir = tmp_path / "reports"
        runner.invoke(app, [
            "report", "daily",
            "--segments-file", str(segments_file),
            "--out-dir", str(out_dir),
        ])
        report_file = next(out_dir.glob("report_*.json"))
        report = json.loads(report_file.read_text())

        assert "date" in report
        assert "total_minutes" in report
        assert "breakdown" in report
        assert "segments_count" in report
        assert isinstance(report["breakdown"], dict)

    def test_breakdown_sums_to_total(self, tmp_path: Path, segments_file: Path) -> None:
        out_dir = tmp_path / "reports"
        runner.invoke(app, [
            "report", "daily",
            "--segments-file", str(segments_file),
            "--out-dir", str(out_dir),
        ])
        report_file = next(out_dir.glob("report_*.json"))
        report = json.loads(report_file.read_text())

        breakdown_sum = sum(report["breakdown"].values())
        assert abs(breakdown_sum - report["total_minutes"]) < 0.01

    def test_no_sensitive_fields_in_report(self, tmp_path: Path, segments_file: Path) -> None:
        out_dir = tmp_path / "reports"
        runner.invoke(app, [
            "report", "daily",
            "--segments-file", str(segments_file),
            "--out-dir", str(out_dir),
        ])
        report_file = next(out_dir.glob("report_*.json"))
        raw = report_file.read_text()

        for forbidden in ("raw_keystrokes", "window_title_raw", "clipboard"):
            assert forbidden not in raw, f"Sensitive field {forbidden!r} in report"
