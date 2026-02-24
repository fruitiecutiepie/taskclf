"""Tests for reject-threshold tuning.

Covers: TC-REJECT-005 (sweep table structure), TC-REJECT-006 (best threshold
within bounds), TC-REJECT-007 (custom thresholds list).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from taskclf.core.types import LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import train_lgbm


def _build_model_and_val():
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=20))

    features_df = pd.DataFrame([r.model_dump() for r in all_rows])

    spans: list[LabelSpan] = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day)
        spans.extend([
            LabelSpan(start_ts=base.replace(hour=9), end_ts=base.replace(hour=12),
                       label="Build", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=12), end_ts=base.replace(hour=14),
                       label="Write", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=14), end_ts=base.replace(hour=16),
                       label="Communicate", provenance="test"),
            LabelSpan(start_ts=base.replace(hour=16), end_ts=base.replace(hour=17),
                       label="BreakIdle", provenance="test"),
        ])

    labeled = project_blocks_to_windows(features_df, spans)
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

    model, _, _, _, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=5, class_weight="balanced",
    )
    return model, cat_encoders, val_df


@pytest.fixture(scope="module")
def tuning_artifacts():
    return _build_model_and_val()


# ---------------------------------------------------------------------------
# TC-REJECT-005: sweep table structure
# ---------------------------------------------------------------------------

class TestTuneRejectThresholdSweep:
    def test_sweep_has_expected_keys(self, tuning_artifacts) -> None:
        from taskclf.train.evaluate import tune_reject_threshold

        model, cat_encoders, val_df = tuning_artifacts
        result = tune_reject_threshold(model, val_df, cat_encoders=cat_encoders)

        assert len(result.sweep) > 0
        expected_keys = {"threshold", "accuracy_on_accepted", "reject_rate", "coverage", "macro_f1"}
        for row in result.sweep:
            assert set(row.keys()) == expected_keys

    def test_coverage_plus_reject_equals_one(self, tuning_artifacts) -> None:
        from taskclf.train.evaluate import tune_reject_threshold

        model, cat_encoders, val_df = tuning_artifacts
        result = tune_reject_threshold(model, val_df, cat_encoders=cat_encoders)

        for row in result.sweep:
            assert abs(row["coverage"] + row["reject_rate"] - 1.0) < 0.01

    def test_sweep_is_sorted_by_threshold(self, tuning_artifacts) -> None:
        from taskclf.train.evaluate import tune_reject_threshold

        model, cat_encoders, val_df = tuning_artifacts
        result = tune_reject_threshold(model, val_df, cat_encoders=cat_encoders)

        thresholds = [r["threshold"] for r in result.sweep]
        assert thresholds == sorted(thresholds)


# ---------------------------------------------------------------------------
# TC-REJECT-006: best threshold within acceptance bounds
# ---------------------------------------------------------------------------

class TestTuneRejectThresholdBest:
    def test_best_threshold_is_valid_float(self, tuning_artifacts) -> None:
        from taskclf.train.evaluate import tune_reject_threshold

        model, cat_encoders, val_df = tuning_artifacts
        result = tune_reject_threshold(model, val_df, cat_encoders=cat_encoders)

        assert isinstance(result.best_threshold, float)
        assert 0.0 < result.best_threshold < 1.0


# ---------------------------------------------------------------------------
# TC-REJECT-007: custom thresholds list
# ---------------------------------------------------------------------------

class TestTuneRejectCustomThresholds:
    def test_custom_thresholds_are_used(self, tuning_artifacts) -> None:
        from taskclf.train.evaluate import tune_reject_threshold

        model, cat_encoders, val_df = tuning_artifacts
        custom = [0.3, 0.5, 0.7, 0.9]
        result = tune_reject_threshold(
            model, val_df, cat_encoders=cat_encoders, thresholds=custom,
        )

        assert len(result.sweep) == len(custom)
        assert [r["threshold"] for r in result.sweep] == custom
