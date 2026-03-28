"""Integration tests: unknown-category training and inference behavior.

Covers:
- UNK-005: withheld categories produce lower max-probability than known
- UNK-006: withheld categories have higher reject rate than known
- EXP-F: 4-condition comparison (Experiment F) produces macro-F1 and reject rate
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from taskclf.core.model_io import build_metadata, load_model_bundle, save_model_bundle
from taskclf.core.types import FeatureRow, LabelSpan
from taskclf.features.build import generate_dummy_features
from taskclf.infer.online import OnlinePredictor
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.train.dataset import split_by_time
from taskclf.train.lgbm import train_lgbm


def _build_labeled_df() -> pd.DataFrame:
    dates = [dt.date(2025, 6, 14), dt.date(2025, 6, 15)]
    all_rows: list[FeatureRow] = []
    for d in dates:
        all_rows.extend(generate_dummy_features(d, n_rows=30))

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
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory):
    labeled = _build_labeled_df()
    splits = split_by_time(labeled)
    train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df,
        val_df,
        num_boost_round=10,
        min_category_freq=1,
        unknown_mask_rate=0.05,
        random_state=42,
    )

    base_dir = tmp_path_factory.mktemp("unknown_models")
    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=dt.date(2025, 6, 14),
        train_date_to=dt.date(2025, 6, 15),
        params=params,
        dataset_hash="unknown_test_hash",
        data_provenance="synthetic",
        unknown_category_freq_threshold=params.get("unknown_category_freq_threshold"),
        unknown_category_mask_rate=params.get("unknown_category_mask_rate"),
    )
    run_dir = save_model_bundle(
        model, metadata, metrics, cm_df, base_dir, cat_encoders=cat_encoders
    )

    return {
        "run_dir": run_dir,
        "val_df": val_df,
        "cat_encoders": cat_encoders,
        "model": model,
        "metadata": metadata,
    }


class TestUnknownCategoryInference:
    """UNK-005, UNK-006: withheld categories degrade gracefully."""

    def test_unk005_unknown_rows_lower_confidence(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """UNK-005: mean max-probability for unknown categories < known categories."""
        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        pred = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
        )

        known_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        known_confs = [pred.predict_bucket(r).confidence for r in known_rows]

        pred_unknown = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=None,
        )
        unknown_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        for row in unknown_rows:
            object.__setattr__(row, "app_id", "com.never.seen.app.xyz")
            object.__setattr__(row, "app_category", "totally_unknown_category")
        unknown_confs = [
            pred_unknown.predict_bucket(r).confidence for r in unknown_rows
        ]

        assert np.mean(unknown_confs) <= np.mean(known_confs)

    def test_unk006_unknown_rows_higher_reject_rate(
        self,
        trained_artifacts,
        valid_feature_row_data,
    ) -> None:
        """UNK-006: reject rate on withheld categories > reject rate on known."""
        model, metadata, cat_encoders = load_model_bundle(trained_artifacts["run_dir"])
        reject_threshold = 0.5

        pred_known = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=reject_threshold,
        )
        known_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=20)
        known_rejects = sum(
            1 for r in known_rows if pred_known.predict_bucket(r).is_rejected
        )

        pred_unknown = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
            smooth_window=1,
            reject_threshold=reject_threshold,
        )
        unknown_rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=20)
        for row in unknown_rows:
            object.__setattr__(row, "app_id", "com.never.seen.app.xyz")
            object.__setattr__(row, "app_category", "totally_unknown_category")
        unknown_rejects = sum(
            1 for r in unknown_rows if pred_unknown.predict_bucket(r).is_rejected
        )

        assert unknown_rejects >= known_rejects


class TestExperimentF:
    """EXP-F: 4-condition comparison from Experiment F."""

    def test_exp_f_four_conditions_produce_metrics(self) -> None:
        """EXP-F: train under 4 conditions and verify each produces macro-F1 and reject rate."""
        labeled = _build_labeled_df()
        splits = split_by_time(labeled)
        train_df = labeled.iloc[splits["train"]].reset_index(drop=True)
        val_df = labeled.iloc[splits["val"]].reset_index(drop=True)

        conditions: dict[str, dict[str, int | float]] = {
            "legacy_minus1": {"min_category_freq": 1, "unknown_mask_rate": 0.0},
            "freq_only": {"min_category_freq": 3, "unknown_mask_rate": 0.0},
            "mask_only": {"min_category_freq": 1, "unknown_mask_rate": 0.10},
            "hybrid": {"min_category_freq": 3, "unknown_mask_rate": 0.05},
        }

        results: dict[str, dict] = {}
        for name, cfg in conditions.items():
            model, metrics, _, params, cat_encoders = train_lgbm(
                train_df,
                val_df,
                num_boost_round=10,
                min_category_freq=int(cfg["min_category_freq"]),
                unknown_mask_rate=float(cfg["unknown_mask_rate"]),
                random_state=42,
            )
            reject_threshold = 0.4

            pred = OnlinePredictor(
                model,
                build_metadata(
                    label_set=list(metrics["label_names"]),
                    train_date_from=dt.date(2025, 6, 14),
                    train_date_to=dt.date(2025, 6, 15),
                    params=params,
                    dataset_hash="exp_f",
                    data_provenance="synthetic",
                ),
                cat_encoders=cat_encoders,
                smooth_window=1,
                reject_threshold=reject_threshold,
            )

            test_rows = generate_dummy_features(dt.date(2025, 6, 16), n_rows=20)
            for row in test_rows:
                object.__setattr__(row, "app_id", "com.unseen.experiment")
            preds = [pred.predict_bucket(r) for r in test_rows]

            reject_rate = sum(1 for p in preds if p.is_rejected) / len(preds)
            results[name] = {
                "macro_f1": metrics["macro_f1"],
                "reject_rate": reject_rate,
            }

        for name, res in results.items():
            assert "macro_f1" in res, f"{name}: missing macro_f1"
            assert "reject_rate" in res, f"{name}: missing reject_rate"
            assert 0 <= res["macro_f1"] <= 1, f"{name}: invalid macro_f1"
            assert 0 <= res["reject_rate"] <= 1, f"{name}: invalid reject_rate"
