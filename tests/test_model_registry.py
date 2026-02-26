"""Tests for model registry: scanning, compatibility, constraints, ranking, and selection.

Covers:
- TC-REG-001: list_bundles scans valid bundles
- TC-REG-002: list_bundles handles missing/corrupt files gracefully
- TC-REG-003: is_compatible checks schema hash + label set
- TC-REG-004: passes_constraints (v1 baseline)
- TC-REG-005: score produces deterministic ranking
- TC-SEL-001..010: find_best_model end-to-end selection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskclf.core.model_io import ModelMetadata
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1
from taskclf.model_registry import (
    BundleMetrics,
    ExclusionRecord,
    ModelBundle,
    SelectionPolicy,
    SelectionReport,
    find_best_model,
    is_compatible,
    list_bundles,
    passes_constraints,
    score,
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
# TC-REG-001: list_bundles — valid scanning
# ---------------------------------------------------------------------------


class TestListBundles:
    def test_scans_valid_bundle(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "bundle_a", _VALID_METADATA, _VALID_METRICS)
        bundles = list_bundles(tmp_path)

        assert len(bundles) == 1
        b = bundles[0]
        assert b.model_id == "bundle_a"
        assert b.valid is True
        assert b.metadata is not None
        assert b.metrics is not None
        assert b.metrics.macro_f1 == 0.82
        assert b.created_at is not None

    def test_scans_multiple_bundles(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "bundle_b", _VALID_METADATA, _VALID_METRICS)
        _write_bundle(tmp_path / "bundle_a", _VALID_METADATA, _VALID_METRICS)
        _write_bundle(tmp_path / "bundle_c", _VALID_METADATA, _VALID_METRICS)

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 3

    def test_returns_deterministic_order_by_model_id(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "zzz_bundle", _VALID_METADATA, _VALID_METRICS)
        _write_bundle(tmp_path / "aaa_bundle", _VALID_METADATA, _VALID_METRICS)
        _write_bundle(tmp_path / "mmm_bundle", _VALID_METADATA, _VALID_METRICS)

        bundles = list_bundles(tmp_path)
        ids = [b.model_id for b in bundles]
        assert ids == ["aaa_bundle", "mmm_bundle", "zzz_bundle"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_bundles(tmp_path) == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_bundles(tmp_path / "nonexistent") == []

    def test_ignores_plain_files(self, tmp_path: Path) -> None:
        (tmp_path / "not_a_bundle.txt").write_text("hello")
        _write_bundle(tmp_path / "real_bundle", _VALID_METADATA, _VALID_METRICS)

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].model_id == "real_bundle"


# ---------------------------------------------------------------------------
# TC-REG-002: list_bundles — graceful error handling
# ---------------------------------------------------------------------------


class TestListBundlesInvalid:
    def test_missing_metadata_json(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "no_meta"
        bundle_dir.mkdir()
        (bundle_dir / "metrics.json").write_text(json.dumps(_VALID_METRICS))

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "missing metadata.json" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_missing_metrics_json(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "no_metrics", _VALID_METADATA, metrics=None)

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "missing metrics.json" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_corrupt_metadata_json(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "corrupt_meta"
        bundle_dir.mkdir()
        (bundle_dir / "metadata.json").write_text("{not valid json")
        (bundle_dir / "metrics.json").write_text(json.dumps(_VALID_METRICS))

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "metadata.json parse error" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_corrupt_metrics_json(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "corrupt_metrics", _VALID_METADATA, metrics="{bad json")

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "metrics.json parse error" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_missing_required_metadata_keys(self, tmp_path: Path) -> None:
        _write_bundle(
            tmp_path / "bad_keys",
            metadata={"schema_version": "v1"},
            metrics=_VALID_METRICS,
        )

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "metadata.json validation error" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_missing_required_metrics_keys(self, tmp_path: Path) -> None:
        _write_bundle(
            tmp_path / "bad_metrics_keys",
            metadata=_VALID_METADATA,
            metrics={"macro_f1": 0.5},
        )

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "metrics.json validation error" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_invalid_created_at_timestamp(self, tmp_path: Path) -> None:
        bad_meta = _VALID_METADATA.model_copy(update={"created_at": "not-a-timestamp"})
        _write_bundle(tmp_path / "bad_ts", bad_meta, _VALID_METRICS)

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert bundles[0].valid is False
        assert "created_at parse error" in bundles[0].invalid_reason  # type: ignore[operator]

    def test_invalid_bundles_do_not_crash_valid_ones(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "good", _VALID_METADATA, _VALID_METRICS)
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "metadata.json").write_text("{invalid}")

        bundles = list_bundles(tmp_path)
        assert len(bundles) == 2
        valid = [b for b in bundles if b.valid]
        invalid = [b for b in bundles if not b.valid]
        assert len(valid) == 1
        assert len(invalid) == 1
        assert valid[0].model_id == "good"


# ---------------------------------------------------------------------------
# TC-REG-003: is_compatible
# ---------------------------------------------------------------------------


class TestIsCompatible:
    def test_compatible_bundle(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "compat", _VALID_METADATA, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]
        assert is_compatible(bundle) is True

    def test_schema_hash_mismatch(self, tmp_path: Path) -> None:
        bad = _VALID_METADATA.model_copy(update={"schema_hash": "wrong_hash"})
        _write_bundle(tmp_path / "bad_hash", bad, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]
        assert is_compatible(bundle) is False

    def test_label_set_mismatch(self, tmp_path: Path) -> None:
        bad = _VALID_METADATA.model_copy(update={"label_set": ["Build", "Debug"]})
        _write_bundle(tmp_path / "bad_labels", bad, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]
        assert is_compatible(bundle) is False

    def test_invalid_bundle_is_not_compatible(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "invalid"
        bundle_dir.mkdir()
        bundles = list_bundles(tmp_path)
        assert len(bundles) == 1
        assert is_compatible(bundles[0]) is False

    def test_custom_schema_hash(self, tmp_path: Path) -> None:
        custom_hash = "custom_hash_123"
        meta = _VALID_METADATA.model_copy(update={"schema_hash": custom_hash})
        _write_bundle(tmp_path / "custom", meta, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]

        assert is_compatible(bundle, required_schema_hash=custom_hash) is True
        assert is_compatible(bundle) is False

    def test_custom_label_set(self, tmp_path: Path) -> None:
        custom_labels = frozenset(["Alpha", "Beta"])
        meta = _VALID_METADATA.model_copy(update={"label_set": sorted(custom_labels)})
        _write_bundle(tmp_path / "custom_labels", meta, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]

        assert is_compatible(
            bundle,
            required_schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            required_label_set=custom_labels,
        ) is True
        assert is_compatible(bundle) is False


# ---------------------------------------------------------------------------
# TC-REG-004: passes_constraints
# ---------------------------------------------------------------------------


class TestPassesConstraints:
    def test_valid_bundle_passes(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "ok", _VALID_METADATA, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]
        assert passes_constraints(bundle, SelectionPolicy()) is True

    def test_invalid_bundle_fails(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "invalid"
        bundle_dir.mkdir()
        bundle = list_bundles(tmp_path)[0]
        assert passes_constraints(bundle, SelectionPolicy()) is False

    def test_bundle_without_metrics_fails(self) -> None:
        bundle = ModelBundle(
            model_id="no_metrics",
            path=Path("/fake"),
            valid=True,
            metrics=None,
        )
        assert passes_constraints(bundle, SelectionPolicy()) is False


# ---------------------------------------------------------------------------
# TC-REG-005: score
# ---------------------------------------------------------------------------


class TestScore:
    def test_returns_correct_tuple(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "scored", _VALID_METADATA, _VALID_METRICS)
        bundle = list_bundles(tmp_path)[0]
        s = score(bundle, SelectionPolicy())
        assert s == (0.82, 0.85, "2026-02-01T00:00:00+00:00")

    def test_higher_macro_f1_wins(self, tmp_path: Path) -> None:
        low_metrics = {**_VALID_METRICS, "macro_f1": 0.70}
        high_metrics = {**_VALID_METRICS, "macro_f1": 0.90}

        _write_bundle(tmp_path / "low", _VALID_METADATA, low_metrics)
        _write_bundle(tmp_path / "high", _VALID_METADATA, high_metrics)

        bundles = list_bundles(tmp_path)
        ranked = sorted(bundles, key=lambda b: score(b, SelectionPolicy()), reverse=True)
        assert ranked[0].model_id == "high"
        assert ranked[1].model_id == "low"

    def test_tie_break_by_weighted_f1(self, tmp_path: Path) -> None:
        m1 = {**_VALID_METRICS, "macro_f1": 0.80, "weighted_f1": 0.90}
        m2 = {**_VALID_METRICS, "macro_f1": 0.80, "weighted_f1": 0.70}

        _write_bundle(tmp_path / "higher_wf1", _VALID_METADATA, m1)
        _write_bundle(tmp_path / "lower_wf1", _VALID_METADATA, m2)

        bundles = list_bundles(tmp_path)
        ranked = sorted(bundles, key=lambda b: score(b, SelectionPolicy()), reverse=True)
        assert ranked[0].model_id == "higher_wf1"

    def test_tie_break_by_created_at(self, tmp_path: Path) -> None:
        older_meta = _VALID_METADATA.model_copy(
            update={"created_at": "2026-01-01T00:00:00+00:00"}
        )
        newer_meta = _VALID_METADATA.model_copy(
            update={"created_at": "2026-02-15T00:00:00+00:00"}
        )
        same_metrics = {**_VALID_METRICS, "macro_f1": 0.80, "weighted_f1": 0.85}

        _write_bundle(tmp_path / "older", older_meta, same_metrics)
        _write_bundle(tmp_path / "newer", newer_meta, same_metrics)

        bundles = list_bundles(tmp_path)
        ranked = sorted(bundles, key=lambda b: score(b, SelectionPolicy()), reverse=True)
        assert ranked[0].model_id == "newer"

    def test_invalid_bundle_raises(self) -> None:
        bundle = ModelBundle(
            model_id="bad", path=Path("/fake"), valid=False, invalid_reason="test"
        )
        with pytest.raises(ValueError, match="Cannot score invalid bundle"):
            score(bundle, SelectionPolicy())

    def test_ranking_is_deterministic(self, tmp_path: Path) -> None:
        for i in range(5):
            meta = _VALID_METADATA.model_copy(
                update={"created_at": f"2026-01-{10 + i:02d}T00:00:00+00:00"}
            )
            metrics = {**_VALID_METRICS, "macro_f1": 0.70 + i * 0.02}
            _write_bundle(tmp_path / f"run_{i:03d}", meta, metrics)

        policy = SelectionPolicy()
        bundles = list_bundles(tmp_path)
        ranked_a = sorted(bundles, key=lambda b: score(b, policy), reverse=True)
        ranked_b = sorted(bundles, key=lambda b: score(b, policy), reverse=True)
        assert [b.model_id for b in ranked_a] == [b.model_id for b in ranked_b]
        assert ranked_a[0].model_id == "run_004"


# ---------------------------------------------------------------------------
# TC-SEL-001..010: find_best_model
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "models"


class TestFindBestModel:
    """End-to-end tests for :func:`find_best_model`."""

    # TC-SEL-001: empty / nonexistent directory
    def test_empty_dir_returns_no_best(self, tmp_path: Path) -> None:
        report = find_best_model(tmp_path)
        assert report.best is None
        assert report.ranked == []
        assert report.excluded == []

    def test_nonexistent_dir_returns_no_best(self, tmp_path: Path) -> None:
        report = find_best_model(tmp_path / "does_not_exist")
        assert report.best is None

    # TC-SEL-002: all invalid bundles
    def test_all_invalid_bundles(self, tmp_path: Path) -> None:
        (tmp_path / "bad_a").mkdir()
        _write_bundle(tmp_path / "bad_b", _VALID_METADATA, metrics="{corrupt")

        report = find_best_model(tmp_path)
        assert report.best is None
        assert report.ranked == []
        assert len(report.excluded) == 2
        reasons = {e.model_id: e.reason for e in report.excluded}
        assert "invalid:" in reasons["bad_a"]
        assert "invalid:" in reasons["bad_b"]

    # TC-SEL-003: schema mismatch excluded
    def test_schema_mismatch_excluded(self, tmp_path: Path) -> None:
        bad_meta = _VALID_METADATA.model_copy(update={"schema_hash": "wrong"})
        _write_bundle(tmp_path / "wrong_hash", bad_meta, _VALID_METRICS)
        _write_bundle(tmp_path / "good", _VALID_METADATA, _VALID_METRICS)

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "good"
        assert len(report.ranked) == 1
        excluded_ids = {e.model_id for e in report.excluded}
        assert "wrong_hash" in excluded_ids
        wrong = next(e for e in report.excluded if e.model_id == "wrong_hash")
        assert "schema_hash mismatch" in wrong.reason

    # TC-SEL-004: label set mismatch excluded
    def test_label_set_mismatch_excluded(self, tmp_path: Path) -> None:
        bad_meta = _VALID_METADATA.model_copy(update={"label_set": ["X", "Y"]})
        _write_bundle(tmp_path / "bad_labels", bad_meta, _VALID_METRICS)
        _write_bundle(tmp_path / "good", _VALID_METADATA, _VALID_METRICS)

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "good"
        bad = next(e for e in report.excluded if e.model_id == "bad_labels")
        assert "label_set mismatch" in bad.reason

    # TC-SEL-005: ranking selects highest macro_f1
    def test_ranking_selects_highest_macro_f1(self, tmp_path: Path) -> None:
        for name, f1 in [("low", 0.60), ("mid", 0.75), ("high", 0.90)]:
            _write_bundle(
                tmp_path / name,
                _VALID_METADATA,
                {**_VALID_METRICS, "macro_f1": f1},
            )

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "high"
        assert [b.model_id for b in report.ranked] == ["high", "mid", "low"]

    # TC-SEL-006: tie-break weighted_f1 then created_at
    def test_tie_break_weighted_f1(self, tmp_path: Path) -> None:
        same_f1 = {**_VALID_METRICS, "macro_f1": 0.80}
        _write_bundle(
            tmp_path / "low_w", _VALID_METADATA, {**same_f1, "weighted_f1": 0.70}
        )
        _write_bundle(
            tmp_path / "high_w", _VALID_METADATA, {**same_f1, "weighted_f1": 0.90}
        )

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "high_w"

    def test_tie_break_created_at(self, tmp_path: Path) -> None:
        same = {**_VALID_METRICS, "macro_f1": 0.80, "weighted_f1": 0.85}
        old_meta = _VALID_METADATA.model_copy(
            update={"created_at": "2026-01-01T00:00:00+00:00"}
        )
        new_meta = _VALID_METADATA.model_copy(
            update={"created_at": "2026-02-20T00:00:00+00:00"}
        )
        _write_bundle(tmp_path / "older", old_meta, same)
        _write_bundle(tmp_path / "newer", new_meta, same)

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "newer"

    # TC-SEL-007: mixed valid/invalid/incompatible
    def test_mixed_bundles(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "good", _VALID_METADATA, _VALID_METRICS)
        bad_hash = _VALID_METADATA.model_copy(update={"schema_hash": "nope"})
        _write_bundle(tmp_path / "incompat", bad_hash, _VALID_METRICS)
        (tmp_path / "broken").mkdir()
        _write_bundle(tmp_path / "also_good", _VALID_METADATA, {
            **_VALID_METRICS, "macro_f1": 0.90,
        })

        report = find_best_model(tmp_path)
        assert report.best is not None
        assert report.best.model_id == "also_good"
        assert len(report.ranked) == 2
        assert len(report.excluded) == 2
        assert len(report.ranked) + len(report.excluded) == 4

    # TC-SEL-008: deterministic across calls
    def test_deterministic(self, tmp_path: Path) -> None:
        for i in range(5):
            meta = _VALID_METADATA.model_copy(
                update={"created_at": f"2026-01-{10 + i:02d}T00:00:00+00:00"}
            )
            _write_bundle(
                tmp_path / f"run_{i:03d}",
                meta,
                {**_VALID_METRICS, "macro_f1": 0.70 + i * 0.02},
            )

        r1 = find_best_model(tmp_path)
        r2 = find_best_model(tmp_path)
        assert r1.best is not None
        assert r1.best.model_id == r2.best.model_id  # type: ignore[union-attr]
        assert [b.model_id for b in r1.ranked] == [b.model_id for b in r2.ranked]

    # TC-SEL-009: report completeness (ranked + excluded == total)
    def test_report_completeness(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path / "ok", _VALID_METADATA, _VALID_METRICS)
        (tmp_path / "empty").mkdir()

        report = find_best_model(tmp_path)
        total_scanned = len(list_bundles(tmp_path))
        assert len(report.ranked) + len(report.excluded) == total_scanned

    # TC-SEL-010: fixture-based integration test
    def test_fixture_bundles(self) -> None:
        report = find_best_model(FIXTURES_DIR)

        assert report.best is not None
        assert report.best.model_id == "best_bundle"

        ranked_ids = [b.model_id for b in report.ranked]
        assert ranked_ids == ["best_bundle", "good_bundle", "second_good_bundle"]

        excluded_ids = {e.model_id for e in report.excluded}
        assert "bad_schema_bundle" in excluded_ids
        assert "corrupt_json_bundle" in excluded_ids
        assert "missing_metrics_bundle" in excluded_ids

        assert len(report.ranked) + len(report.excluded) == 6

    # TC-SEL-011: custom required_schema_hash override
    def test_custom_schema_hash_selects_different_bundle(self, tmp_path: Path) -> None:
        custom = "custom_hash_999"
        meta_custom = _VALID_METADATA.model_copy(update={"schema_hash": custom})
        _write_bundle(tmp_path / "custom", meta_custom, _VALID_METRICS)
        _write_bundle(tmp_path / "default", _VALID_METADATA, _VALID_METRICS)

        report = find_best_model(tmp_path, required_schema_hash=custom)
        assert report.best is not None
        assert report.best.model_id == "custom"
        assert report.required_schema_hash == custom
        excluded_ids = {e.model_id for e in report.excluded}
        assert "default" in excluded_ids
