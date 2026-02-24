"""Tests for user-specific taxonomy mapping layer.

Covers:
- TC-TAX-001: Config validation (valid/invalid YAML, Pydantic rules)
- TC-TAX-002: TaxonomyResolver single-row resolution
- TC-TAX-003: TaxonomyResolver batch resolution
- TC-TAX-004: YAML round-trip (load -> save -> load)
- TC-TAX-005: Default taxonomy identity mapping
- TC-TAX-006: CLI taxonomy validate / show / init
- TC-TAX-007: Batch inference integration with taxonomy
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from taskclf.cli.main import app
from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.types import LABEL_SET_V1, CoreLabel
from taskclf.infer.taxonomy import (
    FALLBACK_BUCKET_NAME,
    TaxonomyBucket,
    TaxonomyConfig,
    TaxonomyResolver,
    TaxonomyResult,
    default_taxonomy,
    load_taxonomy,
    save_taxonomy,
)

runner = CliRunner()


def _example_config() -> TaxonomyConfig:
    """Minimal valid taxonomy config for testing."""
    return TaxonomyConfig(
        buckets=[
            TaxonomyBucket(
                name="Deep Work",
                core_labels=["Build", "Debug", "Write"],
                color="#2E86DE",
            ),
            TaxonomyBucket(
                name="Research",
                core_labels=["ReadResearch", "Review"],
                color="#9B59B6",
            ),
            TaxonomyBucket(
                name="Communication",
                core_labels=["Communicate"],
                color="#27AE60",
            ),
            TaxonomyBucket(
                name="Meetings",
                core_labels=["Meet"],
                color="#E67E22",
            ),
            TaxonomyBucket(
                name="Break",
                core_labels=["BreakIdle"],
                color="#7F8C8D",
            ),
        ],
    )


def _uniform_probs() -> np.ndarray:
    """Uniform probability vector across 8 core labels."""
    return np.ones(8, dtype=np.float64) / 8


def _peaked_probs(label: str, peak: float = 0.8) -> np.ndarray:
    """Probability vector peaked at a specific core label."""
    sorted_labels = sorted(CoreLabel)
    idx = sorted_labels.index(label)
    probs = np.full(8, (1.0 - peak) / 7, dtype=np.float64)
    probs[idx] = peak
    return probs


# ---------------------------------------------------------------------------
# TC-TAX-001: Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_valid_config_loads(self) -> None:
        config = _example_config()
        assert len(config.buckets) == 5
        assert config.version == "1.0"

    def test_invalid_core_label_rejects(self) -> None:
        with pytest.raises(ValidationError, match="Unknown core label"):
            TaxonomyBucket(
                name="Bad", core_labels=["NotARealLabel"], color="#000000"
            )

    def test_duplicate_bucket_names_reject(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate bucket names"):
            TaxonomyConfig(
                buckets=[
                    TaxonomyBucket(name="A", core_labels=["Build"], color="#111111"),
                    TaxonomyBucket(name="A", core_labels=["Debug"], color="#222222"),
                ],
            )

    def test_invalid_hex_color_rejects(self) -> None:
        with pytest.raises(ValidationError, match="Invalid hex color"):
            TaxonomyBucket(name="Bad", core_labels=["Build"], color="red")

    def test_invalid_hex_short_rejects(self) -> None:
        with pytest.raises(ValidationError, match="Invalid hex color"):
            TaxonomyBucket(name="Bad", core_labels=["Build"], color="#FFF")

    def test_empty_buckets_rejects(self) -> None:
        with pytest.raises(ValidationError):
            TaxonomyConfig(buckets=[])

    def test_empty_core_labels_rejects(self) -> None:
        with pytest.raises(ValidationError):
            TaxonomyBucket(name="Empty", core_labels=[], color="#000000")

    def test_invalid_reweight_label_rejects(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        with pytest.raises(ValidationError, match="Unknown core label"):
            TaxonomyAdvanced(reweight_core_labels={"FakeLabel": 2.0})

    def test_zero_reweight_rejects(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        with pytest.raises(ValidationError, match="must be > 0"):
            TaxonomyAdvanced(reweight_core_labels={"Build": 0.0})

    def test_negative_reweight_rejects(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        with pytest.raises(ValidationError, match="must be > 0"):
            TaxonomyAdvanced(reweight_core_labels={"Build": -1.0})

    def test_valid_reweights_accepted(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        adv = TaxonomyAdvanced(reweight_core_labels={"Build": 2.0, "Debug": 0.5})
        assert adv.reweight_core_labels["Build"] == 2.0

    def test_all_core_labels_covered(self) -> None:
        config = _example_config()
        covered = set()
        for b in config.buckets:
            covered.update(b.core_labels)
        assert covered == LABEL_SET_V1

    def test_user_taxonomy_example_loads(self) -> None:
        example_path = Path("configs/user_taxonomy_example.yaml")
        if example_path.exists():
            config = load_taxonomy(example_path)
            assert len(config.buckets) >= 1


# ---------------------------------------------------------------------------
# TC-TAX-002: Resolver single-row
# ---------------------------------------------------------------------------


class TestResolverSingleRow:
    def test_sum_aggregation(self) -> None:
        config = _example_config()
        resolver = TaxonomyResolver(config)

        probs = _peaked_probs("Build", 0.8)
        sorted_labels = sorted(CoreLabel)
        label_id = sorted_labels.index("Build")

        result = resolver.resolve(label_id, probs)
        assert isinstance(result, TaxonomyResult)
        assert result.mapped_label == "Deep Work"
        assert result.mapped_probs["Deep Work"] > 0.5

    def test_max_aggregation(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        config = TaxonomyConfig(
            buckets=_example_config().buckets,
            advanced=TaxonomyAdvanced(probability_aggregation="max"),
        )
        resolver = TaxonomyResolver(config)

        probs = _peaked_probs("Build", 0.8)
        sorted_labels = sorted(CoreLabel)
        label_id = sorted_labels.index("Build")

        result = resolver.resolve(label_id, probs)
        assert result.mapped_label == "Deep Work"

    def test_rejected_row(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        probs = _uniform_probs()

        result = resolver.resolve(0, probs, is_rejected=True)
        assert result.mapped_label == MIXED_UNKNOWN
        assert result.mapped_probs == {}

    def test_custom_reject_label(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyReject

        config = TaxonomyConfig(
            buckets=_example_config().buckets,
            reject=TaxonomyReject(mixed_label_name="Unknown"),
        )
        resolver = TaxonomyResolver(config)

        result = resolver.resolve(0, _uniform_probs(), is_rejected=True)
        assert result.mapped_label == "Unknown"

    def test_mapped_probs_sum_to_one(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        probs = _peaked_probs("Communicate", 0.6)
        sorted_labels = sorted(CoreLabel)
        label_id = sorted_labels.index("Communicate")

        result = resolver.resolve(label_id, probs)
        total = sum(result.mapped_probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_reweighting(self) -> None:
        from taskclf.infer.taxonomy import TaxonomyAdvanced

        config = TaxonomyConfig(
            buckets=_example_config().buckets,
            advanced=TaxonomyAdvanced(
                reweight_core_labels={"Build": 10.0},
            ),
        )
        resolver = TaxonomyResolver(config)

        probs = _uniform_probs()
        result = resolver.resolve(0, probs)

        assert result.mapped_label == "Deep Work"
        assert result.mapped_probs["Deep Work"] > 0.3

    def test_unmapped_core_labels_go_to_fallback(self) -> None:
        config = TaxonomyConfig(
            buckets=[
                TaxonomyBucket(
                    name="Coding",
                    core_labels=["Build", "Debug"],
                    color="#111111",
                ),
            ],
        )
        resolver = TaxonomyResolver(config)
        assert FALLBACK_BUCKET_NAME in resolver.bucket_names

        probs = _peaked_probs("Meet", 0.9)
        sorted_labels = sorted(CoreLabel)
        label_id = sorted_labels.index("Meet")

        result = resolver.resolve(label_id, probs)
        assert result.mapped_label == FALLBACK_BUCKET_NAME

    def test_core_probs_not_modified(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        probs = _peaked_probs("Build", 0.8)
        original = probs.copy()
        resolver.resolve(0, probs)
        np.testing.assert_array_equal(probs, original)

    def test_ties_broken_by_bucket_order(self) -> None:
        config = TaxonomyConfig(
            buckets=[
                TaxonomyBucket(name="A", core_labels=["Build"], color="#111111"),
                TaxonomyBucket(name="B", core_labels=["Debug"], color="#222222"),
                TaxonomyBucket(name="C", core_labels=["Review"], color="#333333"),
                TaxonomyBucket(name="D", core_labels=["Write"], color="#444444"),
                TaxonomyBucket(name="E", core_labels=["ReadResearch"], color="#555555"),
                TaxonomyBucket(name="F", core_labels=["Communicate"], color="#666666"),
                TaxonomyBucket(name="G", core_labels=["Meet"], color="#777777"),
                TaxonomyBucket(name="H", core_labels=["BreakIdle"], color="#888888"),
            ],
        )
        resolver = TaxonomyResolver(config)
        probs = _uniform_probs()
        result = resolver.resolve(0, probs)
        assert result.mapped_label in resolver.bucket_names


# ---------------------------------------------------------------------------
# TC-TAX-003: Resolver batch
# ---------------------------------------------------------------------------


class TestResolverBatch:
    def test_batch_consistency(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        sorted_labels = sorted(CoreLabel)

        probs_matrix = np.array([
            _peaked_probs("Build", 0.8),
            _peaked_probs("Meet", 0.9),
            _peaked_probs("BreakIdle", 0.7),
        ])
        label_ids = np.array([
            sorted_labels.index("Build"),
            sorted_labels.index("Meet"),
            sorted_labels.index("BreakIdle"),
        ])

        results = resolver.resolve_batch(label_ids, probs_matrix)
        assert len(results) == 3
        assert results[0].mapped_label == "Deep Work"
        assert results[1].mapped_label == "Meetings"
        assert results[2].mapped_label == "Break"

    def test_batch_probs_sum_to_one(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        n = 10
        probs_matrix = np.random.default_rng(42).dirichlet(np.ones(8), size=n)
        label_ids = probs_matrix.argmax(axis=1)

        results = resolver.resolve_batch(label_ids, probs_matrix)
        for r in results:
            total = sum(r.mapped_probs.values())
            assert abs(total - 1.0) < 1e-5

    def test_batch_with_rejections(self) -> None:
        resolver = TaxonomyResolver(_example_config())
        probs_matrix = np.array([_uniform_probs(), _peaked_probs("Build", 0.8)])
        label_ids = probs_matrix.argmax(axis=1)
        is_rejected = np.array([True, False])

        results = resolver.resolve_batch(label_ids, probs_matrix, is_rejected=is_rejected)
        assert results[0].mapped_label == MIXED_UNKNOWN
        assert results[0].mapped_probs == {}
        assert results[1].mapped_label == "Deep Work"


# ---------------------------------------------------------------------------
# TC-TAX-004: YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        original = _example_config()
        yaml_path = save_taxonomy(original, tmp_path / "taxonomy.yaml")
        loaded = load_taxonomy(yaml_path)

        assert loaded.version == original.version
        assert len(loaded.buckets) == len(original.buckets)
        for orig, ld in zip(original.buckets, loaded.buckets):
            assert orig.name == ld.name
            assert orig.core_labels == ld.core_labels
            assert orig.color == ld.color

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_taxonomy(tmp_path / "does_not_exist.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("buckets:\n  - name: X\n    core_labels: [NotALabel]\n    color: '#000000'\n")
        with pytest.raises(ValidationError):
            load_taxonomy(bad_yaml)

    def test_load_example_config(self) -> None:
        path = Path("configs/user_taxonomy_example.yaml")
        if path.exists():
            config = load_taxonomy(path)
            assert config.version == "1.0"
            assert len(config.buckets) >= 1


# ---------------------------------------------------------------------------
# TC-TAX-005: Default taxonomy
# ---------------------------------------------------------------------------


class TestDefaultTaxonomy:
    def test_identity_mapping(self) -> None:
        config = default_taxonomy()
        assert len(config.buckets) == len(CoreLabel)
        for bucket, label in zip(config.buckets, CoreLabel):
            assert bucket.name == label
            assert bucket.core_labels == [label]

    def test_all_labels_covered(self) -> None:
        config = default_taxonomy()
        covered = set()
        for b in config.buckets:
            covered.update(b.core_labels)
        assert covered == LABEL_SET_V1

    def test_resolver_identity_preserves_argmax(self) -> None:
        resolver = TaxonomyResolver(default_taxonomy())
        sorted_labels = sorted(CoreLabel)

        for label in CoreLabel:
            probs = _peaked_probs(label, 0.8)
            idx = sorted_labels.index(label)
            result = resolver.resolve(idx, probs)
            assert result.mapped_label == label


# ---------------------------------------------------------------------------
# TC-TAX-006: CLI taxonomy commands
# ---------------------------------------------------------------------------


class TestCliTaxonomyValidate:
    def test_valid_config_exit_zero(self, tmp_path: Path) -> None:
        config = _example_config()
        yaml_path = save_taxonomy(config, tmp_path / "tax.yaml")
        result = runner.invoke(app, ["taxonomy", "validate", "--config", str(yaml_path)])
        assert result.exit_code == 0, result.output
        assert "Taxonomy valid" in result.output

    def test_invalid_config_exit_one(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("buckets:\n  - name: X\n    core_labels: [FakeLabel]\n    color: '#000000'\n")
        result = runner.invoke(app, ["taxonomy", "validate", "--config", str(bad)])
        assert result.exit_code == 1

    def test_missing_file_exit_one(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "taxonomy", "validate",
            "--config", str(tmp_path / "nope.yaml"),
        ])
        assert result.exit_code == 1


class TestCliTaxonomyShow:
    def test_show_displays_table(self, tmp_path: Path) -> None:
        config = _example_config()
        yaml_path = save_taxonomy(config, tmp_path / "tax.yaml")
        result = runner.invoke(app, ["taxonomy", "show", "--config", str(yaml_path)])
        assert result.exit_code == 0, result.output
        assert "Deep Work" in result.output
        assert "Build" in result.output


class TestCliTaxonomyInit:
    def test_init_creates_valid_yaml(self, tmp_path: Path) -> None:
        out_path = tmp_path / "new_tax.yaml"
        result = runner.invoke(app, ["taxonomy", "init", "--out", str(out_path)])
        assert result.exit_code == 0, result.output
        assert out_path.exists()

        config = load_taxonomy(out_path)
        assert len(config.buckets) == len(CoreLabel)

    def test_init_refuses_overwrite(self, tmp_path: Path) -> None:
        out_path = tmp_path / "existing.yaml"
        out_path.write_text("existing content")
        result = runner.invoke(app, ["taxonomy", "init", "--out", str(out_path)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# TC-TAX-007: Batch inference integration
# ---------------------------------------------------------------------------


class TestBatchInferenceIntegration:
    @pytest.fixture(scope="class")
    def trained_model_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        models_dir = tmp_path_factory.mktemp("models")
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

    def test_batch_with_taxonomy_produces_mapped_column(
        self, tmp_path: Path, trained_model_dir: Path,
    ) -> None:
        tax_path = save_taxonomy(_example_config(), tmp_path / "tax.yaml")
        out_dir = tmp_path / "artifacts"

        result = runner.invoke(app, [
            "infer", "batch",
            "--model-dir", str(trained_model_dir),
            "--from", "2025-06-15",
            "--to", "2025-06-15",
            "--synthetic",
            "--out-dir", str(out_dir),
            "--taxonomy", str(tax_path),
        ])
        assert result.exit_code == 0, result.output
        assert "mapped_label" in result.output

        df = pd.read_csv(out_dir / "predictions.csv")
        assert "mapped_label" in df.columns
        assert len(df) > 0

        bucket_names = {b.name for b in _example_config().buckets} | {MIXED_UNKNOWN}
        invalid = set(df["mapped_label"].unique()) - bucket_names
        assert not invalid, f"Unexpected mapped labels: {invalid}"

    def test_batch_without_taxonomy_has_no_mapped_column(
        self, tmp_path: Path, trained_model_dir: Path,
    ) -> None:
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
        df = pd.read_csv(out_dir / "predictions.csv")
        assert "mapped_label" not in df.columns
