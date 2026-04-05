"""Tests for taskclf.model_inspection helpers and inspect_model."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from taskclf.core.model_io import build_metadata, save_model_bundle
from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.model_inspection import (
    inspect_bundle_only,
    inspect_model,
    per_class_metrics_from_confusion_matrix,
    replay_test_evaluation,
)
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import train_lgbm


def _build_labeled_df() -> pd.DataFrame:
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=20))
    features_df = pd.DataFrame([r.model_dump() for r in all_rows])
    spans: list[LabelSpan] = []
    for d in dates:
        base = dt.datetime(d.year, d.month, d.day)
        spans.extend(
            [
                LabelSpan(
                    start_ts=base.replace(hour=9),
                    end_ts=base.replace(hour=12),
                    label="Build",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=12),
                    end_ts=base.replace(hour=14),
                    label="Write",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=14),
                    end_ts=base.replace(hour=16),
                    label="Communicate",
                    provenance="test",
                ),
                LabelSpan(
                    start_ts=base.replace(hour=16),
                    end_ts=base.replace(hour=17),
                    label="BreakIdle",
                    provenance="test",
                ),
            ]
        )
    return project_blocks_to_windows(features_df, spans)


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory: pytest.TempPathFactory):
    labeled = _build_labeled_df()
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)
    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df,
        val_df,
        num_boost_round=5,
    )
    base_dir = tmp_path_factory.mktemp("models")
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dt.date(2025, 6, 14),
        train_date_to=dt.date(2025, 6, 15),
        params=params,
        dataset_hash="test_hash",
        data_provenance="synthetic",
    )
    run_dir = save_model_bundle(
        model, metadata, metrics, cm_df, base_dir, cat_encoders=cat_encoders
    )
    return run_dir


class TestPerClassFromConfusionMatrix:
    def test_matches_sklearn(self) -> None:
        rng = np.random.default_rng(42)
        labels = sorted(LABEL_SET_V1)
        n = len(labels)
        y_true = [labels[i] for i in rng.integers(0, n, size=200)]
        y_pred = [labels[i] for i in rng.integers(0, n, size=200)]
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        derived = per_class_metrics_from_confusion_matrix(cm, labels)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        for i, name in enumerate(labels):
            assert derived[name]["precision"] == round(float(prec[i]), 4)
            assert derived[name]["recall"] == round(float(rec[i]), 4)
            assert derived[name]["f1"] == round(float(f1[i]), 4)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="confusion matrix"):
            per_class_metrics_from_confusion_matrix([[1, 2]], ["A", "B"])


class TestInspectBundleOnly:
    def test_inspect_bundle_only(self, trained_bundle) -> None:
        path, metadata, section = inspect_bundle_only(trained_bundle)
        assert path == trained_bundle.resolve()
        assert metadata.schema_hash
        assert section.macro_f1 == section.macro_f1
        assert len(section.label_names) == len(LABEL_SET_V1)
        assert len(section.confusion_matrix) == len(section.label_names)


class TestInspectModel:
    def test_bundle_only_no_replay(self, trained_bundle) -> None:
        r = inspect_model(trained_bundle)
        assert r.bundle_path
        assert r.replayed_test_evaluation is None
        assert r.replay_error is None
        assert r.prediction_logic.multilabel is False

    def test_replay_synthetic(self, trained_bundle) -> None:
        r = inspect_model(
            trained_bundle,
            date_from=dt.date(2025, 6, 15),
            date_to=dt.date(2025, 6, 15),
            data_dir=None,
            synthetic=True,
        )
        assert r.replayed_test_evaluation is not None
        assert r.replayed_test_evaluation.test_row_count > 0
        assert "macro_f1" in r.replayed_test_evaluation.report
        assert r.replay_error is None


class TestReplayTestEvaluation:
    def test_returns_test_distribution(self, trained_bundle) -> None:
        rt = replay_test_evaluation(
            trained_bundle,
            dt.date(2025, 6, 15),
            dt.date(2025, 6, 15),
            data_dir=trained_bundle.parent,  # unused for synthetic
            synthetic=True,
        )
        assert rt.test_row_count > 0
        total = sum(
            int(rt.test_class_distribution[c]["count"])
            for c in rt.test_class_distribution
        )
        assert total == rt.test_row_count
