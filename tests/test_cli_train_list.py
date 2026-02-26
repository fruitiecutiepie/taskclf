"""Tests for ``taskclf train list`` CLI command.

Covers:
- TC-TL-001: Default table output renders without error
- TC-TL-002: ``--eligible`` filters incompatible/invalid bundles
- TC-TL-003: ``--sort`` reorders output
- TC-TL-004: ``--json`` produces valid JSON with expected keys
- TC-TL-005: ``--schema-hash`` filtering
- TC-TL-006: Active bundle marked correctly
- TC-TL-007: Empty models directory handled gracefully
- TC-TL-008: Invalid --sort value rejected
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "models"

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
# TC-TL-001: Default table output
# ---------------------------------------------------------------------------


class TestDefaultTable:
    def test_table_renders_without_error(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR)])
        assert result.exit_code == 0
        assert "Model Bundles" in result.output

    def test_all_bundles_rendered(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = {r["model_id"] for r in data}
        assert ids == {"best_bundle", "good_bundle", "second_good_bundle", "bad_schema_bundle", "corrupt_json_bundle", "missing_metrics_bundle"}

    def test_metrics_present(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        by_id = {r["model_id"]: r for r in data}
        assert by_id["best_bundle"]["macro_f1"] == 0.88
        assert by_id["best_bundle"]["weighted_f1"] == 0.91

    def test_invalid_bundle_shows_notes(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        by_id = {r["model_id"]: r for r in data}
        assert by_id["missing_metrics_bundle"]["notes"] is not None
        assert "missing metrics.json" in by_id["missing_metrics_bundle"]["notes"]


# ---------------------------------------------------------------------------
# TC-TL-002: --eligible filter
# ---------------------------------------------------------------------------


class TestEligibleFilter:
    def test_eligible_excludes_invalid(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = {r["model_id"] for r in data}
        assert "best_bundle" in ids
        assert "good_bundle" in ids
        assert "second_good_bundle" in ids
        assert "corrupt_json_bundle" not in ids
        assert "missing_metrics_bundle" not in ids

    def test_eligible_excludes_incompatible_schema(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = {r["model_id"] for r in data}
        assert "bad_schema_bundle" not in ids

    def test_eligible_table_renders(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible"])
        assert result.exit_code == 0
        assert "Model Bundles" in result.output


# ---------------------------------------------------------------------------
# TC-TL-003: --sort
# ---------------------------------------------------------------------------


class TestSort:
    def test_sort_macro_f1_default(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        f1s = [r["macro_f1"] for r in data]
        assert f1s == sorted(f1s, reverse=True)

    def test_sort_weighted_f1(self) -> None:
        result = runner.invoke(
            app,
            ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible", "--sort", "weighted_f1", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        wf1s = [r["weighted_f1"] for r in data]
        assert wf1s == sorted(wf1s, reverse=True)

    def test_sort_created_at(self) -> None:
        result = runner.invoke(
            app,
            ["train", "list", "--models-dir", str(FIXTURES_DIR), "--eligible", "--sort", "created_at", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        dates = [r["created_at"] for r in data]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# TC-TL-004: --json output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_is_valid(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 6

    def test_json_keys(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        expected_keys = {"model_id", "created_at", "schema_hash", "macro_f1", "weighted_f1", "bi_prec", "min_prec", "eligible", "active", "notes"}
        for row in data:
            assert set(row.keys()) == expected_keys

    def test_json_eligible_flag_values(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        by_id = {r["model_id"]: r for r in data}
        assert by_id["best_bundle"]["eligible"] is True
        assert by_id["bad_schema_bundle"]["eligible"] is False
        assert by_id["corrupt_json_bundle"]["eligible"] is False

    def test_json_precision_values(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json", "--eligible"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for row in data:
            assert row["bi_prec"] is not None
            assert row["min_prec"] is not None
            assert 0.0 <= row["bi_prec"] <= 1.0
            assert 0.0 <= row["min_prec"] <= 1.0

    def test_json_null_metrics_for_invalid(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        by_id = {r["model_id"]: r for r in data}
        corrupt = by_id["corrupt_json_bundle"]
        assert corrupt["macro_f1"] is None
        assert corrupt["bi_prec"] is None


# ---------------------------------------------------------------------------
# TC-TL-005: --schema-hash filtering
# ---------------------------------------------------------------------------


class TestSchemaHashFilter:
    def test_custom_schema_hash(self, tmp_path: Path) -> None:
        custom_hash = "custom_test_hash_123"
        meta = {**_VALID_METADATA, "schema_hash": custom_hash}
        _write_bundle(tmp_path / "custom_model", meta, _VALID_METRICS)
        _write_bundle(tmp_path / "default_model", _VALID_METADATA, _VALID_METRICS)

        result = runner.invoke(
            app,
            ["train", "list", "--models-dir", str(tmp_path), "--schema-hash", custom_hash, "--eligible", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["model_id"] == "custom_model"


# ---------------------------------------------------------------------------
# TC-TL-006: Active bundle marking
# ---------------------------------------------------------------------------


class TestActiveMarking:
    def test_active_bundle_marked_json(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "model_a", _VALID_METADATA, _VALID_METRICS)
        _write_bundle(tmp_path / "model_b", _VALID_METADATA, _VALID_METRICS)

        pointer = {
            "model_dir": "model_a",
            "selected_at": "2026-02-01T00:00:00+00:00",
            "policy_version": 1,
            "model_id": "model_a",
        }
        (tmp_path / "active.json").write_text(json.dumps(pointer))

        result = runner.invoke(app, ["train", "list", "--models-dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        by_id = {r["model_id"]: r for r in data}
        assert by_id["model_a"]["active"] is True
        assert by_id["model_b"]["active"] is False

    def test_active_shown_in_table(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "active_model", _VALID_METADATA, _VALID_METRICS)
        pointer = {
            "model_dir": "active_model",
            "selected_at": "2026-02-01T00:00:00+00:00",
            "policy_version": 1,
        }
        (tmp_path / "active.json").write_text(json.dumps(pointer))

        result = runner.invoke(app, ["train", "list", "--models-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "*" in result.output


# ---------------------------------------------------------------------------
# TC-TL-007: Empty directory
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    def test_empty_models_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No model bundles found" in result.output


# ---------------------------------------------------------------------------
# TC-TL-008: Invalid --sort value
# ---------------------------------------------------------------------------


class TestInvalidSort:
    def test_bad_sort_value(self) -> None:
        result = runner.invoke(app, ["train", "list", "--models-dir", str(FIXTURES_DIR), "--sort", "nonsense"])
        assert result.exit_code == 1
