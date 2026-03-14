"""Tests for the retraining workflow: cadence, hashing, regression gates, pipeline.

Covers:
- TC-RETRAIN-001: dataset hash determinism
- TC-RETRAIN-002: dataset hash sensitivity to changes
- TC-RETRAIN-003: cadence check with fresh model
- TC-RETRAIN-004: cadence check with stale model
- TC-RETRAIN-005: cadence check with no model
- TC-RETRAIN-006: comparative regression gates pass (challenger >= champion)
- TC-RETRAIN-007: comparative regression gates fail on macro-F1 regression
- TC-RETRAIN-008: comparative regression gates exclude candidate gates
- TC-RETRAIN-010: config roundtrip from YAML
- TC-RETRAIN-011: full pipeline with synthetic data
- TC-RETRAIN-012: pipeline does not promote on gate failure
- TC-RETRAIN-013: promoted model metadata contains dataset_hash
- TC-RETRAIN-014: check_candidate_gates standalone
- TC-RETRAIN-015: champion resolved from active pointer
- TC-RETRAIN-016: pipeline updates active pointer on promotion
- TC-RETRAIN-020..025: check_calibrator_update_due
- TC-RETRAIN-026..030: find_latest_model isolated
- TC-RETRAIN-031..036: TrainParams, DatasetSnapshot, RegressionGate constructor tests
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from taskclf.core.model_io import build_metadata, save_model_bundle
from taskclf.core.types import LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.store import generate_dummy_labels
from taskclf.train.evaluate import EvaluationReport
from taskclf.model_registry import read_active
from taskclf.train.retrain import (
    DatasetSnapshot,
    RetrainConfig,
    RegressionGate,
    TrainParams,
    check_calibrator_update_due,
    check_candidate_gates,
    check_regression_gates,
    check_retrain_due,
    compute_dataset_hash,
    find_latest_model,
    load_retrain_config,
    run_retrain_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features_and_labels(
    dates: list[dt.date],
    n_rows: int = 60,
) -> tuple[pd.DataFrame, list[LabelSpan]]:
    all_features: list[pd.DataFrame] = []
    all_labels: list[LabelSpan] = []
    for d in dates:
        rows = generate_dummy_features(d, n_rows=n_rows)
        df = pd.DataFrame([r.model_dump() for r in rows])
        labels = generate_dummy_labels(d, n_rows=n_rows)
        all_features.append(df)
        all_labels.extend(labels)
    return pd.concat(all_features, ignore_index=True), all_labels


def _make_eval_report(
    macro_f1: float = 0.75,
    weighted_f1: float = 0.80,
    breakidle_precision: float = 0.98,
    breakidle_recall: float = 0.95,
    all_class_precision: float = 0.70,
    reject_rate: float = 0.10,
) -> EvaluationReport:
    """Construct a minimal EvaluationReport for gate testing."""
    from taskclf.core.types import LABEL_SET_V1

    label_names = sorted(LABEL_SET_V1)
    per_class = {}
    for name in label_names:
        prec = breakidle_precision if name == "BreakIdle" else all_class_precision
        rec = breakidle_recall if name == "BreakIdle" else 0.70
        per_class[name] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        }

    acceptance_checks = {
        "macro_f1": macro_f1 >= 0.65,
        "weighted_f1": weighted_f1 >= 0.70,
        "breakidle_precision": breakidle_precision >= 0.95,
        "breakidle_recall": breakidle_recall >= 0.90,
        "no_class_below_50_precision": all_class_precision >= 0.50,
        "reject_rate_bounds": 0.05 <= reject_rate <= 0.30,
    }
    acceptance_details = {
        k: f"{'PASS' if v else 'FAIL'}" for k, v in acceptance_checks.items()
    }

    return EvaluationReport(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=[[0] * len(label_names) for _ in label_names],
        label_names=label_names,
        per_user={},
        calibration={},
        stratification={},
        reject_rate=reject_rate,
        acceptance_checks=acceptance_checks,
        acceptance_details=acceptance_details,
    )


def _create_dummy_model_bundle(
    models_dir: Path,
    created_at: str | None = None,
) -> Path:
    """Create a minimal model bundle with a metadata.json."""
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.train.dataset import split_by_time
    from taskclf.train.lgbm import train_lgbm

    dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2)]
    features_df, labels = _make_features_and_labels(dates, n_rows=20)
    labeled = project_blocks_to_windows(features_df, labels)
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df,
        val_df,
        num_boost_round=5,
    )
    dataset_hash = compute_dataset_hash(features_df, labels)
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dates[0],
        train_date_to=dates[-1],
        params=params,
        dataset_hash=dataset_hash,
        data_provenance="synthetic",
    )

    run_dir = save_model_bundle(
        model,
        metadata,
        metrics,
        cm_df,
        models_dir,
        cat_encoders=cat_encoders,
    )

    if created_at is not None:
        meta_path = run_dir / "metadata.json"
        raw = json.loads(meta_path.read_text())
        raw["created_at"] = created_at
        meta_path.write_text(json.dumps(raw, indent=2))

    return run_dir


# ---------------------------------------------------------------------------
# TC-RETRAIN-001 / 002: dataset hash
# ---------------------------------------------------------------------------


class TestComputeDatasetHash:
    def test_deterministic(self) -> None:
        """Same data produces the same hash."""
        dates = [dt.date(2025, 6, 14)]
        features, labels = _make_features_and_labels(dates)
        h1 = compute_dataset_hash(features, labels)
        h2 = compute_dataset_hash(features, labels)
        assert h1 == h2
        assert len(h1) == 16

    def test_differs_on_data_change(self) -> None:
        """Different data produces a different hash."""
        dates_a = [dt.date(2025, 6, 14)]
        dates_b = [dt.date(2025, 6, 15)]
        features_a, labels_a = _make_features_and_labels(dates_a)
        features_b, labels_b = _make_features_and_labels(dates_b)
        h_a = compute_dataset_hash(features_a, labels_a)
        h_b = compute_dataset_hash(features_b, labels_b)
        assert h_a != h_b


# ---------------------------------------------------------------------------
# TC-RETRAIN-003 / 004 / 005: cadence checks
# ---------------------------------------------------------------------------


class TestCheckRetrainDue:
    def test_fresh_model_not_due(self, tmp_path: Path) -> None:
        """Model created moments ago should not trigger retrain."""
        _create_dummy_model_bundle(tmp_path)
        assert check_retrain_due(tmp_path, cadence_days=7) is False

    def test_stale_model_is_due(self, tmp_path: Path) -> None:
        """Model older than cadence triggers retrain."""
        old_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(days=30)).isoformat()
        _create_dummy_model_bundle(tmp_path, created_at=old_ts)
        assert check_retrain_due(tmp_path, cadence_days=7) is True

    def test_no_model_is_due(self, tmp_path: Path) -> None:
        """Empty models directory triggers retrain."""
        assert check_retrain_due(tmp_path, cadence_days=7) is True


# ---------------------------------------------------------------------------
# TC-RETRAIN-006 / 007 / 008: comparative regression gates
# ---------------------------------------------------------------------------


class TestRegressionGates:
    def test_pass_when_challenger_better(self) -> None:
        champion = _make_eval_report(macro_f1=0.70)
        challenger = _make_eval_report(macro_f1=0.75)
        config = RetrainConfig(regression_tolerance=0.02)

        result = check_regression_gates(champion, challenger, config)
        assert result.all_passed is True
        assert all(g.passed for g in result.gates)

    def test_fail_macro_f1_regression(self) -> None:
        champion = _make_eval_report(macro_f1=0.80)
        challenger = _make_eval_report(macro_f1=0.70)
        config = RetrainConfig(regression_tolerance=0.02)

        result = check_regression_gates(champion, challenger, config)
        assert result.all_passed is False
        macro_gate = next(g for g in result.gates if g.name == "macro_f1_no_regression")
        assert macro_gate.passed is False

    def test_pass_within_tolerance(self) -> None:
        """Small regression within tolerance should pass."""
        champion = _make_eval_report(macro_f1=0.75)
        challenger = _make_eval_report(macro_f1=0.74)
        config = RetrainConfig(regression_tolerance=0.02)

        result = check_regression_gates(champion, challenger, config)
        macro_gate = next(g for g in result.gates if g.name == "macro_f1_no_regression")
        assert macro_gate.passed is True

    def test_does_not_include_candidate_gates(self) -> None:
        """Regression gates must only contain comparative checks."""
        champion = _make_eval_report(macro_f1=0.70)
        challenger = _make_eval_report(macro_f1=0.75, breakidle_precision=0.80)
        config = RetrainConfig()

        result = check_regression_gates(champion, challenger, config)
        gate_names = {g.name for g in result.gates}
        assert gate_names == {"macro_f1_no_regression"}
        assert "breakidle_precision" not in gate_names
        assert "no_class_below_50_precision" not in gate_names
        assert "challenger_acceptance" not in gate_names


# ---------------------------------------------------------------------------
# TC-RETRAIN-010: config roundtrip
# ---------------------------------------------------------------------------


class TestRetrainConfig:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "retrain.yaml"
        config_path.write_text(
            "global_retrain_cadence_days: 14\n"
            "calibrator_update_cadence_days: 3\n"
            "data_lookback_days: 60\n"
            "regression_tolerance: 0.05\n"
            "require_baseline_improvement: false\n"
            "auto_promote: true\n"
            "train_params:\n"
            "  num_boost_round: 200\n"
            "  class_weight: none\n"
        )
        config = load_retrain_config(config_path)
        assert config.global_retrain_cadence_days == 14
        assert config.calibrator_update_cadence_days == 3
        assert config.data_lookback_days == 60
        assert config.regression_tolerance == 0.05
        assert config.require_baseline_improvement is False
        assert config.auto_promote is True
        assert config.train_params.num_boost_round == 200
        assert config.train_params.class_weight == "none"

    def test_defaults(self) -> None:
        config = RetrainConfig()
        assert config.global_retrain_cadence_days == 7
        assert config.regression_tolerance == 0.02
        assert config.train_params.num_boost_round == 100


# ---------------------------------------------------------------------------
# TC-RETRAIN-011 / 012 / 013: pipeline integration
# ---------------------------------------------------------------------------


class TestRetrainPipeline:
    def test_full_pipeline_synthetic(self, tmp_path: Path) -> None:
        """End-to-end pipeline with synthetic data should complete."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )

        assert result.dataset_snapshot.dataset_hash
        assert result.dataset_snapshot.row_count > 0
        assert result.challenger_macro_f1 >= 0.0
        assert result.run_dir

    def test_no_promote_on_dry_run(self, tmp_path: Path) -> None:
        """Dry-run should never promote."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            dry_run=True,
            data_provenance="synthetic",
        )

        assert result.promoted is False
        assert "Dry run" in result.reason
        assert find_latest_model(models_dir) is None

    def test_dataset_hash_in_promoted_metadata(self, tmp_path: Path) -> None:
        """Promoted model's metadata.json must contain dataset_hash."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )

        if result.promoted:
            meta_path = Path(result.run_dir) / "metadata.json"
            raw = json.loads(meta_path.read_text())
            assert "dataset_hash" in raw
            assert raw["dataset_hash"] == result.dataset_snapshot.dataset_hash
        else:
            rejected_dir = out_dir / "rejected_models"
            latest = find_latest_model(rejected_dir)
            assert latest is not None
            meta_path = latest / "metadata.json"
            raw = json.loads(meta_path.read_text())
            assert "dataset_hash" in raw

    def test_result_contains_candidate_gates(self, tmp_path: Path) -> None:
        """RetrainResult must include candidate_gates regardless of promotion."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )

        assert result.candidate_gates is not None
        gate_names = {g.name for g in result.candidate_gates.gates}
        assert "breakidle_precision" in gate_names
        assert "no_class_below_50_precision" in gate_names
        assert "challenger_acceptance" in gate_names

    def test_active_pointer_updated_on_promotion(self, tmp_path: Path) -> None:
        """Promoted model should update active.json when it is the best."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )

        if result.promoted:
            assert result.active_updated is True
            pointer = read_active(models_dir)
            assert pointer is not None
            bundle_dir = Path(result.run_dir)
            assert bundle_dir.name in pointer.model_dir

    def test_date_range_from_labeled_data(self, tmp_path: Path) -> None:
        """Regression: snapshot date range must come from labeled data, not all features."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        result = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )

        snap = result.dataset_snapshot
        assert snap.date_from != ""
        assert snap.date_to != ""
        date_from = pd.Timestamp(snap.date_from)
        date_to = pd.Timestamp(snap.date_to)
        assert date_to >= date_from

    def test_champion_source_reported(self, tmp_path: Path) -> None:
        """When a champion exists, champion_source should name it."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        out_dir = tmp_path / "artifacts"
        config = RetrainConfig()

        # First run: no champion
        dates = [dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
        features_df, labels = _make_features_and_labels(dates)

        r1 = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )
        assert r1.champion_source is None

        # Second run: champion should be the first promoted model
        r2 = run_retrain_pipeline(
            config,
            features_df,
            labels,
            models_dir=models_dir,
            out_dir=out_dir,
            force=True,
            data_provenance="synthetic",
        )
        if r1.promoted:
            assert r2.champion_source is not None


# ---------------------------------------------------------------------------
# TC-RETRAIN-014: check_candidate_gates standalone
# ---------------------------------------------------------------------------


class TestCandidateGates:
    def test_all_pass(self) -> None:
        report = _make_eval_report(
            macro_f1=0.75,
            breakidle_precision=0.98,
            all_class_precision=0.70,
        )
        result = check_candidate_gates(report)
        assert result.all_passed is True
        assert len(result.gates) == 3

    def test_fail_breakidle_precision(self) -> None:
        report = _make_eval_report(breakidle_precision=0.80)
        result = check_candidate_gates(report)
        assert result.all_passed is False
        bi_gate = next(g for g in result.gates if g.name == "breakidle_precision")
        assert bi_gate.passed is False

    def test_fail_class_precision_below_50(self) -> None:
        report = _make_eval_report(all_class_precision=0.40)
        result = check_candidate_gates(report)
        assert result.all_passed is False
        prec_gate = next(
            g for g in result.gates if g.name == "no_class_below_50_precision"
        )
        assert prec_gate.passed is False

    def test_fail_acceptance(self) -> None:
        report = _make_eval_report(reject_rate=0.50)
        result = check_candidate_gates(report)
        assert result.all_passed is False
        acc_gate = next(g for g in result.gates if g.name == "challenger_acceptance")
        assert acc_gate.passed is False

    def test_does_not_include_comparative_gate(self) -> None:
        """Candidate gates must not include macro_f1_no_regression."""
        report = _make_eval_report()
        result = check_candidate_gates(report)
        gate_names = {g.name for g in result.gates}
        assert "macro_f1_no_regression" not in gate_names

    def test_regression_gates_excludes_candidate_gates(self) -> None:
        """check_regression_gates must NOT include candidate gate names."""
        champion = _make_eval_report(macro_f1=0.70)
        challenger = _make_eval_report(macro_f1=0.75)
        config = RetrainConfig()

        comparative = check_regression_gates(champion, challenger, config)
        candidate_only = check_candidate_gates(challenger)

        comparative_names = {g.name for g in comparative.gates}
        candidate_names = {g.name for g in candidate_only.gates}

        assert comparative_names == {"macro_f1_no_regression"}
        assert comparative_names.isdisjoint(candidate_names)


# ---------------------------------------------------------------------------
# TC-RETRAIN-020..025: check_calibrator_update_due
# ---------------------------------------------------------------------------


class TestCheckCalibratorUpdateDue:
    def test_no_store_file(self, tmp_path: Path) -> None:
        """TC-RETRAIN-020: missing store.json → update due."""
        assert check_calibrator_update_due(tmp_path) is True

    def test_fresh_store(self, tmp_path: Path) -> None:
        """TC-RETRAIN-021: freshly created store → not due."""
        store = tmp_path / "store.json"
        store.write_text(
            json.dumps({"created_at": dt.datetime.now(dt.UTC).isoformat()})
        )
        assert check_calibrator_update_due(tmp_path) is False

    def test_stale_store(self, tmp_path: Path) -> None:
        """TC-RETRAIN-022: store older than cadence → due."""
        old = dt.datetime.now(dt.UTC) - dt.timedelta(days=30)
        store = tmp_path / "store.json"
        store.write_text(json.dumps({"created_at": old.isoformat()}))
        assert check_calibrator_update_due(tmp_path) is True

    def test_missing_created_at(self, tmp_path: Path) -> None:
        """TC-RETRAIN-023: store.json without created_at → due."""
        store = tmp_path / "store.json"
        store.write_text(json.dumps({"method": "temperature"}))
        assert check_calibrator_update_due(tmp_path) is True

    def test_malformed_json(self, tmp_path: Path) -> None:
        """TC-RETRAIN-024: malformed JSON in store.json → due."""
        store = tmp_path / "store.json"
        store.write_text("{bad json!!!")
        assert check_calibrator_update_due(tmp_path) is True

    def test_custom_cadence(self, tmp_path: Path) -> None:
        """TC-RETRAIN-025: custom cadence_days=1 with 2-day-old store → due."""
        old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
        store = tmp_path / "store.json"
        store.write_text(json.dumps({"created_at": old.isoformat()}))
        assert check_calibrator_update_due(tmp_path, cadence_days=1) is True


# ---------------------------------------------------------------------------
# TC-RETRAIN-026..030: find_latest_model (isolated)
# ---------------------------------------------------------------------------


class TestFindLatestModel:
    def test_empty_directory(self, tmp_path: Path) -> None:
        """TC-RETRAIN-026: empty directory returns None."""
        assert find_latest_model(tmp_path) is None

    def test_single_bundle(self, tmp_path: Path) -> None:
        """TC-RETRAIN-027: single valid bundle is returned."""
        run = tmp_path / "run_001"
        run.mkdir()
        (run / "metadata.json").write_text(
            json.dumps(
                {
                    "created_at": "2025-07-01T00:00:00",
                }
            )
        )
        result = find_latest_model(tmp_path)
        assert result == run

    def test_multiple_bundles_returns_latest(self, tmp_path: Path) -> None:
        """TC-RETRAIN-028: with multiple bundles, the latest is returned."""
        for i, ts in enumerate(
            ["2025-07-01T00:00:00", "2025-07-03T00:00:00", "2025-07-02T00:00:00"]
        ):
            run = tmp_path / f"run_{i:03d}"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({"created_at": ts}))
        result = find_latest_model(tmp_path)
        assert result is not None
        assert result.name == "run_001"

    def test_unreadable_metadata_skipped(self, tmp_path: Path) -> None:
        """TC-RETRAIN-029: bundle with unreadable metadata is skipped."""
        bad_run = tmp_path / "bad_run"
        bad_run.mkdir()
        (bad_run / "metadata.json").write_text("not valid json")

        good_run = tmp_path / "good_run"
        good_run.mkdir()
        (good_run / "metadata.json").write_text(
            json.dumps(
                {
                    "created_at": "2025-07-01T00:00:00",
                }
            )
        )

        result = find_latest_model(tmp_path)
        assert result == good_run

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """TC-RETRAIN-030: non-existent directory returns None."""
        assert find_latest_model(tmp_path / "does_not_exist") is None


class TestTrainParamsConstructor:
    """TC-RETRAIN-031/032: TrainParams defaults and custom values."""

    def test_defaults(self) -> None:
        """TC-RETRAIN-031: TrainParams() uses configured defaults."""
        from taskclf.core.defaults import DEFAULT_NUM_BOOST_ROUND

        params = TrainParams()
        assert params.num_boost_round == DEFAULT_NUM_BOOST_ROUND
        assert params.class_weight == "balanced"

    def test_custom_values(self) -> None:
        """TC-RETRAIN-032: TrainParams with explicit overrides."""
        params = TrainParams(num_boost_round=500, class_weight="none")
        assert params.num_boost_round == 500
        assert params.class_weight == "none"


class TestDatasetSnapshotConstructor:
    """TC-RETRAIN-033/034: DatasetSnapshot construction and immutability."""

    def test_construction(self) -> None:
        """TC-RETRAIN-033: DatasetSnapshot stores all fields."""
        snap = DatasetSnapshot(
            dataset_hash="abc123",
            row_count=1000,
            date_from="2025-06-01",
            date_to="2025-06-30",
            user_count=3,
            class_distribution={"Build": 400, "Write": 300, "BreakIdle": 300},
        )
        assert snap.dataset_hash == "abc123"
        assert snap.row_count == 1000
        assert snap.date_from == "2025-06-01"
        assert snap.date_to == "2025-06-30"
        assert snap.user_count == 3
        assert snap.class_distribution["Build"] == 400

    def test_frozen_immutability(self) -> None:
        """TC-RETRAIN-034: DatasetSnapshot is frozen (immutable)."""
        snap = DatasetSnapshot(
            dataset_hash="abc",
            row_count=10,
            date_from="2025-01-01",
            date_to="2025-01-31",
            user_count=1,
            class_distribution={"Build": 10},
        )
        with pytest.raises(Exception):
            snap.row_count = 999  # type: ignore[misc]


class TestRegressionGateConstructor:
    """TC-RETRAIN-035/036: RegressionGate construction and immutability."""

    def test_construction(self) -> None:
        """TC-RETRAIN-035: RegressionGate stores name, passed, detail."""
        gate = RegressionGate(
            name="macro_f1",
            passed=True,
            detail="challenger 0.85 >= champion 0.80",
        )
        assert gate.name == "macro_f1"
        assert gate.passed is True
        assert "0.85" in gate.detail

    def test_frozen_immutability(self) -> None:
        """TC-RETRAIN-036: RegressionGate is frozen (immutable)."""
        gate = RegressionGate(name="test", passed=False, detail="fail")
        with pytest.raises(Exception):
            gate.passed = True  # type: ignore[misc]
