"""CLI CliRunner tests for commands that had no end-to-end coverage.

Covers:
- A1  labels add-block     (TC-CLI-AB-001 .. TC-CLI-AB-005)
- A2  labels label-now     (TC-CLI-LN-001 .. TC-CLI-LN-005)
- A3  labels show-queue    (TC-CLI-SQ-001 .. TC-CLI-SQ-004)
- A4  labels project       (TC-CLI-LP-001 .. TC-CLI-LP-004)
- A5  train build-dataset  (TC-CLI-BD-001 .. TC-CLI-BD-003)
- A6  train evaluate       (TC-CLI-EV-001 .. TC-CLI-EV-004)
- A7  train tune-reject    (TC-CLI-TR-001 .. TC-CLI-TR-003)
- A8  train calibrate      (TC-CLI-CA-001 .. TC-CLI-CA-003)
- A9  train retrain        (TC-CLI-RT-001 .. TC-CLI-RT-004)
- A10 train check-retrain  (TC-CLI-CR-001 .. TC-CLI-CR-003)
- A11 infer baseline       (TC-CLI-BL-001 .. TC-CLI-BL-003)
- A12 infer compare        (TC-CLI-IC-001 .. TC-CLI-IC-003)
- A14 monitor drift-check  (TC-CLI-DC-001 .. TC-CLI-DC-003)
- A15 monitor telemetry    (TC-CLI-TEL-001 .. TC-CLI-TEL-002)
- A16 monitor show         (TC-CLI-MS-001 .. TC-CLI-MS-004)
- A13 infer online         (TC-CLI-IO-001 .. TC-CLI-IO-002)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.telemetry import NUMERICAL_FEATURES
from taskclf.labels.store import read_label_spans

runner = CliRunner()


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_features(tmp_path: Path, date: str = "2025-06-15") -> Path:
    """Use the CLI to generate a synthetic features parquet and return data_dir."""
    data_dir = tmp_path / "data"
    result = runner.invoke(app, [
        "features", "build",
        "--date", date,
        "--data-dir", str(data_dir),
    ])
    assert result.exit_code == 0, result.output
    return data_dir


def _import_labels(data_dir: Path, tmp_path: Path) -> None:
    """Import a small labels CSV covering 2025-06-15 into *data_dir*."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "start_ts,end_ts,label,provenance\n"
        "2025-06-15 09:00:00,2025-06-15 10:00:00,Build,manual\n"
        "2025-06-15 10:00:00,2025-06-15 11:00:00,Write,manual\n"
    )
    result = runner.invoke(app, [
        "labels", "import",
        "--file", str(csv_path),
        "--data-dir", str(data_dir),
    ])
    assert result.exit_code == 0, result.output


def _populate_queue(data_dir: Path, *, n: int = 3, user_id: str = "u1") -> None:
    """Write *n* pending items into the labeling queue inside *data_dir*."""
    from taskclf.labels.queue import ActiveLabelingQueue

    queue_path = data_dir / "labels_v1" / "queue.json"
    queue = ActiveLabelingQueue(queue_path)
    df = pd.DataFrame([
        {
            "user_id": user_id,
            "bucket_start_ts": dt.datetime(2025, 6, 15, 10, i),
            "bucket_end_ts": dt.datetime(2025, 6, 15, 10, i + 1),
            "confidence": 0.2,
            "predicted_label": "Build",
        }
        for i in range(n)
    ])
    queue.enqueue_low_confidence(df, threshold=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# A1  labels add-block
# ═══════════════════════════════════════════════════════════════════════════


class TestLabelsAddBlock:
    """TC-CLI-AB: `taskclf labels add-block` CLI wiring."""

    def test_basic_block_creation(self, tmp_path: Path) -> None:
        """TC-CLI-AB-001: exit 0 and span persisted in labels.parquet."""
        data_dir = tmp_path / "data"
        result = runner.invoke(app, [
            "labels", "add-block",
            "--start", "2025-06-15T09:00:00",
            "--end", "2025-06-15T10:00:00",
            "--label", "Build",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Added label block" in result.output

        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert len(spans) == 1
        assert spans[0].label == "Build"

    def test_overlap_rejection(self, tmp_path: Path) -> None:
        """TC-CLI-AB-002: exit 1 when block overlaps an existing span."""
        data_dir = tmp_path / "data"
        base_args = [
            "labels", "add-block",
            "--start", "2025-06-15T09:00:00",
            "--end", "2025-06-15T10:00:00",
            "--label", "Build",
            "--data-dir", str(data_dir),
        ]
        runner.invoke(app, base_args)
        result = runner.invoke(app, base_args)
        assert result.exit_code == 1

    def test_invalid_label(self, tmp_path: Path) -> None:
        """TC-CLI-AB-003: exit != 0 for a label not in LABEL_SET_V1."""
        data_dir = tmp_path / "data"
        result = runner.invoke(app, [
            "labels", "add-block",
            "--start", "2025-06-15T09:00:00",
            "--end", "2025-06-15T10:00:00",
            "--label", "InvalidLabel",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code != 0

    def test_confidence_persisted(self, tmp_path: Path) -> None:
        """TC-CLI-AB-004: --confidence value round-trips through parquet."""
        data_dir = tmp_path / "data"
        result = runner.invoke(app, [
            "labels", "add-block",
            "--start", "2025-06-15T09:00:00",
            "--end", "2025-06-15T10:00:00",
            "--label", "Write",
            "--confidence", "0.85",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output

        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert len(spans) == 1
        assert spans[0].confidence == pytest.approx(0.85)

    def test_feature_summary_displayed(self, tmp_path: Path) -> None:
        """TC-CLI-AB-005: when features exist, Block Summary table renders."""
        data_dir = _build_features(tmp_path, date="2025-06-15")
        result = runner.invoke(app, [
            "labels", "add-block",
            "--start", "2025-06-15T09:00:00",
            "--end", "2025-06-15T10:00:00",
            "--label", "Build",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Block Summary" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A2  labels label-now
# ═══════════════════════════════════════════════════════════════════════════


_FROZEN_NOW = dt.datetime(2025, 6, 15, 12, 0, 0)


def _patch_now():
    """Patch ``datetime.datetime.now`` inside the CLI module to return _FROZEN_NOW."""
    real_datetime = dt.datetime

    class _FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return _FROZEN_NOW.replace(tzinfo=tz)
            return _FROZEN_NOW

    return patch("taskclf.cli.main.dt.datetime", _FrozenDatetime)


class TestLabelsLabelNow:
    """TC-CLI-LN: `taskclf labels label-now` CLI wiring."""

    def test_basic_labeling(self, tmp_path: Path) -> None:
        """TC-CLI-LN-001: exit 0 and span persisted in labels.parquet."""
        data_dir = tmp_path / "data"
        with _patch_now():
            result = runner.invoke(app, [
                "labels", "label-now",
                "--label", "Build",
                "--data-dir", str(data_dir),
            ])
        assert result.exit_code == 0, result.output
        assert "Labeled" in result.output
        assert "Build" in result.output

        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert len(spans) == 1
        assert spans[0].label == "Build"

    def test_minutes_respected(self, tmp_path: Path) -> None:
        """TC-CLI-LN-002: end_ts - start_ts == timedelta(minutes=N)."""
        data_dir = tmp_path / "data"
        with _patch_now():
            result = runner.invoke(app, [
                "labels", "label-now",
                "--label", "Write",
                "--minutes", "25",
                "--data-dir", str(data_dir),
            ])
        assert result.exit_code == 0, result.output

        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert len(spans) == 1
        delta = spans[0].end_ts - spans[0].start_ts
        assert delta == dt.timedelta(minutes=25)

    def test_aw_unreachable_graceful(self, tmp_path: Path) -> None:
        """TC-CLI-LN-003: exit 0 with 'not reachable' when AW is down."""
        data_dir = tmp_path / "data"
        with _patch_now():
            result = runner.invoke(app, [
                "labels", "label-now",
                "--label", "Build",
                "--aw-host", "http://192.0.2.1:1",
                "--data-dir", str(data_dir),
            ])
        assert result.exit_code == 0, result.output
        assert "not reachable" in result.output.lower()

    def test_overlap_rejection(self, tmp_path: Path) -> None:
        """TC-CLI-LN-004: exit 1 on second identical invocation (overlap)."""
        data_dir = tmp_path / "data"
        with _patch_now():
            runner.invoke(app, [
                "labels", "label-now",
                "--label", "Build",
                "--data-dir", str(data_dir),
            ])
            result = runner.invoke(app, [
                "labels", "label-now",
                "--label", "Write",
                "--data-dir", str(data_dir),
            ])
        assert result.exit_code == 1

    def test_confidence_defaults_to_one(self, tmp_path: Path) -> None:
        """TC-CLI-LN-005: omitting --confidence stores 1.0."""
        data_dir = tmp_path / "data"
        with _patch_now():
            result = runner.invoke(app, [
                "labels", "label-now",
                "--label", "Debug",
                "--data-dir", str(data_dir),
            ])
        assert result.exit_code == 0, result.output
        assert "confidence=1.0" in result.output

        spans = read_label_spans(data_dir / "labels_v1" / "labels.parquet")
        assert spans[0].confidence == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# A3  labels show-queue
# ═══════════════════════════════════════════════════════════════════════════


class TestLabelsShowQueue:
    """TC-CLI-SQ: `taskclf labels show-queue` CLI wiring."""

    def test_no_queue_file(self, tmp_path: Path) -> None:
        """TC-CLI-SQ-001: exit 0 with 'No labeling queue' when file absent."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = runner.invoke(app, [
            "labels", "show-queue",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "No labeling queue" in result.output

    def test_populated_queue(self, tmp_path: Path) -> None:
        """TC-CLI-SQ-002: populated queue renders a table."""
        data_dir = tmp_path / "data"
        _populate_queue(data_dir, n=3)
        result = runner.invoke(app, [
            "labels", "show-queue",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Pending Labeling Requests" in result.output

    def test_user_id_filter(self, tmp_path: Path) -> None:
        """TC-CLI-SQ-003: --user-id shows only matching user's items."""
        data_dir = tmp_path / "data"
        _populate_queue(data_dir, n=2, user_id="alice")
        _populate_queue(data_dir, n=1, user_id="bob")

        result = runner.invoke(app, [
            "labels", "show-queue",
            "--user-id", "alice",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "alice" in result.output
        assert "bob" not in result.output

    def test_limit_cap(self, tmp_path: Path) -> None:
        """TC-CLI-SQ-004: --limit caps visible items."""
        data_dir = tmp_path / "data"
        _populate_queue(data_dir, n=5)
        result = runner.invoke(app, [
            "labels", "show-queue",
            "--limit", "2",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Pending Labeling Requests (2)" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A4  labels project
# ═══════════════════════════════════════════════════════════════════════════


class TestLabelsProject:
    """TC-CLI-LP: `taskclf labels project` CLI wiring."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """TC-CLI-LP-001: exit 0 and projected_labels.parquet created."""
        data_dir = _build_features(tmp_path, date="2025-06-15")
        _import_labels(data_dir, tmp_path)

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "labels", "project",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "labels_v1" / "projected_labels.parquet").exists()

    def test_no_labels_file(self, tmp_path: Path) -> None:
        """TC-CLI-LP-002: exit 1 when labels.parquet is missing."""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        result = runner.invoke(app, [
            "labels", "project",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 1

    def test_no_features_in_range(self, tmp_path: Path) -> None:
        """TC-CLI-LP-003: exit 1 when no feature files for the date range."""
        data_dir = tmp_path / "data"
        _import_labels(data_dir, tmp_path)

        result = runner.invoke(app, [
            "labels", "project",
            "--from", "2099-01-01",
            "--to", "2099-01-01",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 1

    def test_projected_row_count_in_output(self, tmp_path: Path) -> None:
        """TC-CLI-LP-004: output mentions the projected row count."""
        data_dir = _build_features(tmp_path, date="2025-06-15")
        _import_labels(data_dir, tmp_path)

        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "labels", "project",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Projected" in result.output
        assert "labeled windows" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A5  train build-dataset
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainBuildDataset:
    """TC-CLI-BD: `taskclf train build-dataset` CLI wiring."""

    def test_synthetic_dataset_created(self, tmp_path: Path) -> None:
        """TC-CLI-BD-001: exit 0 and X/y/splits artifacts created."""
        out_dir = tmp_path / "data"
        result = runner.invoke(app, [
            "train", "build-dataset",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output

        ds_dir = out_dir / "training_dataset"
        assert (ds_dir / "X.parquet").exists()
        assert (ds_dir / "y.parquet").exists()
        assert (ds_dir / "splits.json").exists()

    def test_custom_ratios(self, tmp_path: Path) -> None:
        """TC-CLI-BD-002: custom train/val ratios reflected in splits."""
        out_dir = tmp_path / "data"
        result = runner.invoke(app, [
            "train", "build-dataset",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
            "--train-ratio", "0.80",
            "--val-ratio", "0.10",
        ])
        assert result.exit_code == 0, result.output

        splits = json.loads(
            (out_dir / "training_dataset" / "splits.json").read_text()
        )
        assert "train" in splits
        assert "val" in splits

    def test_no_features_non_synthetic(self, tmp_path: Path) -> None:
        """TC-CLI-BD-003: exit 1 when non-synthetic and no features on disk."""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        result = runner.invoke(app, [
            "train", "build-dataset",
            "--from", "2099-01-01",
            "--to", "2099-01-01",
            "--data-dir", str(data_dir),
        ])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# A11  infer baseline
# ═══════════════════════════════════════════════════════════════════════════


class TestInferBaseline:
    """TC-CLI-BL: `taskclf infer baseline` CLI wiring."""

    def test_synthetic_baseline(self, tmp_path: Path) -> None:
        """TC-CLI-BL-001: exit 0, predictions CSV + segments JSON created."""
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "baseline",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "baseline_predictions.csv").exists()
        assert (out_dir / "baseline_segments.json").exists()

    def test_reject_rate_in_output(self, tmp_path: Path) -> None:
        """TC-CLI-BL-002: output contains 'reject rate'."""
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "baseline",
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "reject rate" in result.output.lower()

    def test_no_features_in_range(self, tmp_path: Path) -> None:
        """TC-CLI-BL-003: exit 1 when no features for the date range."""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "baseline",
            "--from", "2099-01-01",
            "--to", "2099-01-01",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# A16  monitor show
# ═══════════════════════════════════════════════════════════════════════════


class TestMonitorShow:
    """TC-CLI-MS: `taskclf monitor show` CLI wiring."""

    def test_empty_store(self, tmp_path: Path) -> None:
        """TC-CLI-MS-001: exit 0, 'No telemetry snapshots found'."""
        store_dir = tmp_path / "telemetry"
        store_dir.mkdir()
        result = runner.invoke(app, [
            "monitor", "show",
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "No telemetry snapshots found" in result.output

    def _append_snapshots(
        self, store_dir: Path, n: int = 3, user_id: str | None = None,
    ) -> None:
        from taskclf.core.telemetry import TelemetrySnapshot, TelemetryStore

        store = TelemetryStore(store_dir)
        for i in range(n):
            snap = TelemetrySnapshot(
                timestamp=dt.datetime(2025, 6, 15, 10, i, tzinfo=dt.timezone.utc),
                user_id=user_id,
                total_windows=100 + i,
                reject_rate=0.05,
                mean_entropy=0.3,
            )
            store.append(snap)

    def test_populated_store(self, tmp_path: Path) -> None:
        """TC-CLI-MS-002: table rendered with snapshot data."""
        store_dir = tmp_path / "telemetry"
        self._append_snapshots(store_dir, n=3)
        result = runner.invoke(app, [
            "monitor", "show",
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Recent Telemetry" in result.output
        assert "100" in result.output

    def test_user_id_filter(self, tmp_path: Path) -> None:
        """TC-CLI-MS-003: --user-id shows only matching user."""
        store_dir = tmp_path / "telemetry"
        self._append_snapshots(store_dir, n=2, user_id="alice")
        self._append_snapshots(store_dir, n=1, user_id="bob")

        result = runner.invoke(app, [
            "monitor", "show",
            "--store-dir", str(store_dir),
            "--user-id", "alice",
        ])
        assert result.exit_code == 0, result.output
        assert "alice" in result.output
        assert "bob" not in result.output

    def test_last_n(self, tmp_path: Path) -> None:
        """TC-CLI-MS-004: --last caps number of snapshots shown."""
        store_dir = tmp_path / "telemetry"
        self._append_snapshots(store_dir, n=5)
        result = runner.invoke(app, [
            "monitor", "show",
            "--store-dir", str(store_dir),
            "--last", "2",
        ])
        assert result.exit_code == 0, result.output
        assert "2 snapshots" in result.output


# ── model-training helper ────────────────────────────────────────────────────


def _train_model(tmp_path: Path) -> Path:
    """Train a small synthetic LightGBM model and return the run directory."""
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


# ═══════════════════════════════════════════════════════════════════════════
# A6  train evaluate
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainEvaluate:
    """TC-CLI-EV: `taskclf train evaluate` CLI wiring."""

    def test_synthetic_evaluation(self, tmp_path: Path) -> None:
        """TC-CLI-EV-001: exit 0 and metrics table rendered."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "evaluate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Overall Metrics" in result.output

    def test_acceptance_checks_displayed(self, tmp_path: Path) -> None:
        """TC-CLI-EV-002: output contains PASS/FAIL markers."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "evaluate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Acceptance Checks" in result.output
        assert "PASS" in result.output or "FAIL" in result.output

    def test_evaluation_artifacts_written(self, tmp_path: Path) -> None:
        """TC-CLI-EV-003: evaluation.json created in --out-dir."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "evaluate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "evaluation.json").exists()

    def test_reject_threshold_affects_output(self, tmp_path: Path) -> None:
        """TC-CLI-EV-004: --reject-threshold 0.99 produces higher reject rate."""
        model_dir = _train_model(tmp_path)
        out_default = tmp_path / "art_default"
        out_high = tmp_path / "art_high"

        result_default = runner.invoke(app, [
            "train", "evaluate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_default),
        ])
        assert result_default.exit_code == 0, result_default.output

        result_high = runner.invoke(app, [
            "train", "evaluate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--reject-threshold", "0.99",
            "--out-dir", str(out_high),
        ])
        assert result_high.exit_code == 0, result_high.output

        default_json = json.loads((out_default / "evaluation.json").read_text())
        high_json = json.loads((out_high / "evaluation.json").read_text())
        assert high_json["reject_rate"] >= default_json["reject_rate"]


# ═══════════════════════════════════════════════════════════════════════════
# A7  train tune-reject
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainTuneReject:
    """TC-CLI-TR: `taskclf train tune-reject` CLI wiring."""

    def test_synthetic_sweep(self, tmp_path: Path) -> None:
        """TC-CLI-TR-001: exit 0 and sweep table rendered."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "tune-reject",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Reject Threshold Sweep" in result.output

    def test_json_report_written(self, tmp_path: Path) -> None:
        """TC-CLI-TR-002: reject_tuning.json created in --out-dir."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "tune-reject",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "reject_tuning.json").exists()

    def test_recommended_threshold_in_output(self, tmp_path: Path) -> None:
        """TC-CLI-TR-003: 'Recommended reject threshold' message present."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "tune-reject",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Recommended reject threshold" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A8  train calibrate
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainCalibrate:
    """TC-CLI-CA: `taskclf train calibrate` CLI wiring."""

    def test_synthetic_calibration(self, tmp_path: Path) -> None:
        """TC-CLI-CA-001: exit 0 and calibrator store directory created."""
        model_dir = _train_model(tmp_path)
        out = tmp_path / "cal_store"
        result = runner.invoke(app, [
            "train", "calibrate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert (out / "store.json").exists()

    def test_eligibility_table_rendered(self, tmp_path: Path) -> None:
        """TC-CLI-CA-002: output contains eligibility info."""
        model_dir = _train_model(tmp_path)
        out = tmp_path / "cal_store"
        result = runner.invoke(app, [
            "train", "calibrate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert "Eligibility" in result.output

    def test_isotonic_method(self, tmp_path: Path) -> None:
        """TC-CLI-CA-003: --method isotonic completes without error."""
        model_dir = _train_model(tmp_path)
        out = tmp_path / "cal_store"
        result = runner.invoke(app, [
            "train", "calibrate",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--method", "isotonic",
            "--out", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert (out / "store.json").exists()


# ═══════════════════════════════════════════════════════════════════════════
# A10  train check-retrain
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainCheckRetrain:
    """TC-CLI-CR: `taskclf train check-retrain` CLI wiring."""

    def test_no_models_due(self, tmp_path: Path) -> None:
        """TC-CLI-CR-001: no models directory -> DUE in output."""
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()
        result = runner.invoke(app, [
            "train", "check-retrain",
            "--models-dir", str(models_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "DUE" in result.output

    def test_fresh_model_ok(self, tmp_path: Path) -> None:
        """TC-CLI-CR-002: freshly trained model -> OK in output."""
        model_dir = _train_model(tmp_path)
        models_dir = model_dir.parent
        result = runner.invoke(app, [
            "train", "check-retrain",
            "--models-dir", str(models_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_calibrator_store_row(self, tmp_path: Path) -> None:
        """TC-CLI-CR-003: --calibrator-store adds calibrator row to table."""
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()
        cal_store = tmp_path / "cal_store"
        cal_store.mkdir(parents=True)
        result = runner.invoke(app, [
            "train", "check-retrain",
            "--models-dir", str(models_dir),
            "--calibrator-store", str(cal_store),
        ])
        assert result.exit_code == 0, result.output
        assert "Calibrator" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A12  infer compare
# ═══════════════════════════════════════════════════════════════════════════


class TestInferCompare:
    """TC-CLI-IC: `taskclf infer compare` CLI wiring."""

    def test_synthetic_comparison(self, tmp_path: Path) -> None:
        """TC-CLI-IC-001: exit 0 and comparison table rendered."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "compare",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Baseline vs Model" in result.output

    def test_comparison_json_written(self, tmp_path: Path) -> None:
        """TC-CLI-IC-002: baseline_vs_model.json written to --out-dir."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "compare",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert (out_dir / "baseline_vs_model.json").exists()

    def test_per_class_f1_table(self, tmp_path: Path) -> None:
        """TC-CLI-IC-003: per-class F1 table present in output."""
        model_dir = _train_model(tmp_path)
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "infer", "compare",
            "--model-dir", str(model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Per-Class F1" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A9  train retrain
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainRetrain:
    """TC-CLI-RT: `taskclf train retrain` CLI wiring."""

    def test_synthetic_force(self, tmp_path: Path) -> None:
        """TC-CLI-RT-001: --synthetic --force exits 0 with result table."""
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "retrain",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--force",
            "--models-dir", str(models_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Retrain Result" in result.output

    def test_dry_run(self, tmp_path: Path) -> None:
        """TC-CLI-RT-002: --dry-run prevents promotion."""
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "retrain",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--force",
            "--dry-run",
            "--models-dir", str(models_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "not promoted" in result.output.lower() or "dry" in result.output.lower()

    def test_gate_table_displayed(self, tmp_path: Path) -> None:
        """TC-CLI-RT-003: output contains PASS or FAIL gate rows."""
        model_dir = _train_model(tmp_path)
        models_dir = model_dir.parent
        result = runner.invoke(app, [
            "train", "retrain",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--force",
            "--models-dir", str(models_dir),
            "--out-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output or "FAIL" in result.output

    def test_dataset_hash_in_output(self, tmp_path: Path) -> None:
        """TC-CLI-RT-004: 'Dataset hash' row present in summary table."""
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "artifacts"
        result = runner.invoke(app, [
            "train", "retrain",
            "--from", "2025-06-14",
            "--to", "2025-06-15",
            "--synthetic",
            "--force",
            "--models-dir", str(models_dir),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Dataset hash" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A14  monitor drift-check  —  helpers + tests
# ═══════════════════════════════════════════════════════════════════════════


def _make_drift_fixtures(
    tmp_path: Path,
    *,
    n: int = 200,
    shift_col: str | None = None,
    shift_amount: float = 50.0,
) -> tuple[Path, Path, Path, Path]:
    """Create ref/cur feature parquets and predictions CSVs.

    When *shift_col* is set, the current features DataFrame has that column
    shifted by *shift_amount* relative to reference, triggering PSI drift.

    Returns (ref_features, cur_features, ref_predictions, cur_predictions).
    """
    rng = np.random.default_rng(42)

    base_data: dict[str, object] = {
        "bucket_start_ts": pd.date_range("2025-06-15T09:00", periods=n, freq="min"),
        "user_id": ["u1"] * n,
    }
    for feat in NUMERICAL_FEATURES:
        base_data[feat] = rng.normal(10.0, 2.0, size=n)
    ref_df = pd.DataFrame(base_data)

    cur_df = ref_df.copy()
    if shift_col is not None:
        cur_df[shift_col] = cur_df[shift_col] + shift_amount

    labels = (["Build", "Write", "Debug", "Review"] * (n // 4 + 1))[:n]
    confidences = rng.uniform(0.5, 1.0, size=n)
    probs = rng.dirichlet(np.ones(8), size=n)

    ref_feat_path = tmp_path / "ref_features.parquet"
    cur_feat_path = tmp_path / "cur_features.parquet"
    ref_pred_path = tmp_path / "ref_predictions.csv"
    cur_pred_path = tmp_path / "cur_predictions.csv"

    ref_df.to_parquet(ref_feat_path)
    cur_df.to_parquet(cur_feat_path)

    pred_df = pd.DataFrame({
        "predicted_label": labels,
        "confidence": confidences,
        "core_probs": [json.dumps(row.tolist()) for row in probs],
    })
    pred_df.to_csv(ref_pred_path, index=False)
    pred_df.to_csv(cur_pred_path, index=False)

    return ref_feat_path, cur_feat_path, ref_pred_path, cur_pred_path


class TestMonitorDriftCheck:
    """TC-CLI-DC: `taskclf monitor drift-check` CLI wiring."""

    def test_no_drift(self, tmp_path: Path) -> None:
        """TC-CLI-DC-001: identical ref/cur produces 'No drift detected'."""
        ref_feat, _, ref_pred, _ = _make_drift_fixtures(tmp_path)
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "monitor", "drift-check",
            "--ref-features", str(ref_feat),
            "--cur-features", str(ref_feat),
            "--ref-predictions", str(ref_pred),
            "--cur-predictions", str(ref_pred),
            "--no-auto-label",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "No drift detected" in result.output

    def test_drift_detected(self, tmp_path: Path) -> None:
        """TC-CLI-DC-002: shifted feature triggers alert + drift_report.json."""
        ref_feat, cur_feat, ref_pred, cur_pred = _make_drift_fixtures(
            tmp_path, shift_col="keys_per_min",
        )
        out_dir = tmp_path / "out"
        result = runner.invoke(app, [
            "monitor", "drift-check",
            "--ref-features", str(ref_feat),
            "--cur-features", str(cur_feat),
            "--ref-predictions", str(ref_pred),
            "--cur-predictions", str(cur_pred),
            "--no-auto-label",
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Drift Alerts" in result.output
        assert (out_dir / "drift_report.json").exists()

    def test_auto_label(self, tmp_path: Path) -> None:
        """TC-CLI-DC-003: --auto-label with drift prints 'Auto-enqueued'."""
        ref_feat, cur_feat, ref_pred, cur_pred = _make_drift_fixtures(
            tmp_path, shift_col="keys_per_min",
        )
        out_dir = tmp_path / "out"
        queue_path = tmp_path / "queue.json"
        result = runner.invoke(app, [
            "monitor", "drift-check",
            "--ref-features", str(ref_feat),
            "--cur-features", str(cur_feat),
            "--ref-predictions", str(ref_pred),
            "--cur-predictions", str(cur_pred),
            "--auto-label",
            "--queue-path", str(queue_path),
            "--out-dir", str(out_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Auto-enqueued" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A15  monitor telemetry
# ═══════════════════════════════════════════════════════════════════════════


def _make_telemetry_fixtures(tmp_path: Path, n: int = 100) -> tuple[Path, Path]:
    """Create a features parquet and predictions CSV for telemetry tests.

    Returns (features_path, predictions_path).
    """
    rng = np.random.default_rng(42)
    data: dict[str, object] = {
        "bucket_start_ts": pd.date_range("2025-06-15T09:00", periods=n, freq="min"),
        "user_id": ["u1"] * n,
    }
    for feat in NUMERICAL_FEATURES:
        data[feat] = rng.normal(10.0, 2.0, size=n)
    feat_df = pd.DataFrame(data)

    labels = ["Build", "Write", "Debug", "Review"] * (n // 4)
    confidences = rng.uniform(0.5, 1.0, size=n)
    probs = rng.dirichlet(np.ones(8), size=n)
    pred_df = pd.DataFrame({
        "predicted_label": labels,
        "confidence": confidences,
        "core_probs": [json.dumps(row.tolist()) for row in probs],
    })

    feat_path = tmp_path / "features.parquet"
    pred_path = tmp_path / "predictions.csv"
    feat_df.to_parquet(feat_path)
    pred_df.to_csv(pred_path, index=False)
    return feat_path, pred_path


class TestMonitorTelemetry:
    """TC-CLI-TEL: `taskclf monitor telemetry` CLI wiring."""

    def test_snapshot_stored(self, tmp_path: Path) -> None:
        """TC-CLI-TEL-001: exit 0 and snapshot file created in --store-dir."""
        feat_path, pred_path = _make_telemetry_fixtures(tmp_path)
        store_dir = tmp_path / "telemetry"
        result = runner.invoke(app, [
            "monitor", "telemetry",
            "--features", str(feat_path),
            "--predictions", str(pred_path),
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Telemetry snapshot appended" in result.output
        assert store_dir.exists()
        snapshot_files = list(store_dir.rglob("*.jsonl"))
        assert len(snapshot_files) >= 1

    def test_output_key_metrics(self, tmp_path: Path) -> None:
        """TC-CLI-TEL-002: output contains 'Windows', 'Reject rate', 'Confidence'."""
        feat_path, pred_path = _make_telemetry_fixtures(tmp_path)
        store_dir = tmp_path / "telemetry"
        result = runner.invoke(app, [
            "monitor", "telemetry",
            "--features", str(feat_path),
            "--predictions", str(pred_path),
            "--store-dir", str(store_dir),
        ])
        assert result.exit_code == 0, result.output
        assert "Windows" in result.output
        assert "Reject rate" in result.output
        assert "Confidence" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A13  infer online
# ═══════════════════════════════════════════════════════════════════════════


class TestInferOnline:
    """TC-CLI-IO: `taskclf infer online` CLI wiring."""

    def test_model_resolution_failure(self, tmp_path: Path) -> None:
        """TC-CLI-IO-001: ModelResolutionError → exit 1 with error message."""
        from taskclf.infer.resolve import ModelResolutionError

        with patch(
            "taskclf.infer.resolve.resolve_model_dir",
            side_effect=ModelResolutionError("No eligible model found in models/"),
        ):
            result = runner.invoke(app, [
                "infer", "online",
                "--models-dir", str(tmp_path / "models"),
            ])
        assert result.exit_code == 1
        assert "No eligible model found" in result.output

    def test_label_queue_constructs_path(self, tmp_path: Path) -> None:
        """TC-CLI-IO-002: --label-queue builds queue path from --data-dir."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        with patch(
            "taskclf.infer.resolve.resolve_model_dir",
            return_value=model_dir,
        ), patch(
            "taskclf.infer.online.run_online_loop",
        ) as mock_loop:
            result = runner.invoke(app, [
                "infer", "online",
                "--model-dir", str(model_dir),
                "--data-dir", str(data_dir),
                "--label-queue",
            ])
        assert result.exit_code == 0, result.output
        mock_loop.assert_called_once()
        call_kwargs = mock_loop.call_args[1]
        assert call_kwargs["label_queue_path"] == data_dir / "labels_v1" / "queue.json"
