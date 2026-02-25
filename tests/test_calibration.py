"""Tests for the personalization calibration pipeline.

Covers:
- IsotonicCalibrator basics
- CalibratorStore (global fallback, per-user lookup, calibrate_batch)
- save/load round-trips for individual calibrators and stores
- PersonalizationEligibility checks
- Temperature calibrator fitting
- Isotonic calibrator fitting
- fit_calibrator_store end-to-end
- Batch inference integration with CalibratorStore
- Online predictor integration with CalibratorStore
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.calibration import (
    CalibratorStore,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    load_calibrator,
    load_calibrator_store,
    save_calibrator,
    save_calibrator_store,
)


_N_CLASSES = len(LABEL_SET_V1)
_SORTED_LABELS = sorted(LABEL_SET_V1)


def _make_isotonic_regressors(n_classes: int = _N_CLASSES) -> list[IsotonicRegression]:
    """Fit toy isotonic regressors (identity-ish) for testing."""
    rng = np.random.RandomState(42)
    regs = []
    for _ in range(n_classes):
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        x = np.sort(rng.rand(50))
        y = x + rng.randn(50) * 0.05
        y = np.clip(y, 0, 1)
        reg.fit(x, y)
        regs.append(reg)
    return regs


def _random_probs(n_rows: int, n_classes: int = _N_CLASSES, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw = rng.rand(n_rows, n_classes)
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_output_sums_to_one(self):
        cal = IsotonicCalibrator(_make_isotonic_regressors())
        probs = _random_probs(10)
        calibrated = cal.calibrate(probs)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_preserves_shape_matrix(self):
        cal = IsotonicCalibrator(_make_isotonic_regressors())
        probs = _random_probs(5)
        assert cal.calibrate(probs).shape == probs.shape

    def test_preserves_shape_vector(self):
        cal = IsotonicCalibrator(_make_isotonic_regressors())
        probs = _random_probs(1)[0]
        result = cal.calibrate(probs)
        assert result.shape == probs.shape
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_deterministic(self):
        cal = IsotonicCalibrator(_make_isotonic_regressors())
        probs = _random_probs(5)
        r1 = cal.calibrate(probs)
        r2 = cal.calibrate(probs)
        np.testing.assert_array_equal(r1, r2)

    def test_rejects_empty_regressors(self):
        with pytest.raises(ValueError, match="must not be empty"):
            IsotonicCalibrator([])

    def test_n_classes_property(self):
        regs = _make_isotonic_regressors(4)
        assert IsotonicCalibrator(regs).n_classes == 4


# ---------------------------------------------------------------------------
# CalibratorStore
# ---------------------------------------------------------------------------


class TestCalibratorStore:
    def test_global_fallback(self):
        global_cal = TemperatureCalibrator(temperature=2.0)
        store = CalibratorStore(global_calibrator=global_cal)
        assert store.get_calibrator("unknown-user") is global_cal

    def test_per_user_lookup(self):
        global_cal = IdentityCalibrator()
        user_cal = TemperatureCalibrator(temperature=0.5)
        store = CalibratorStore(
            global_calibrator=global_cal,
            user_calibrators={"user-A": user_cal},
        )
        assert store.get_calibrator("user-A") is user_cal
        assert store.get_calibrator("user-B") is global_cal

    def test_calibrate_batch(self):
        global_cal = IdentityCalibrator()
        user_cal = TemperatureCalibrator(temperature=2.0)
        store = CalibratorStore(
            global_calibrator=global_cal,
            user_calibrators={"user-A": user_cal},
        )

        probs = _random_probs(4)
        user_ids = ["user-A", "user-B", "user-A", "user-B"]
        result = store.calibrate_batch(probs, user_ids)

        # user-B rows should be identical (identity)
        np.testing.assert_array_equal(result[1], probs[1])
        np.testing.assert_array_equal(result[3], probs[3])

        # user-A rows should differ (temperature != 1)
        assert not np.allclose(result[0], probs[0])
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_user_ids_property(self):
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={"z-user": IdentityCalibrator(), "a-user": IdentityCalibrator()},
        )
        assert store.user_ids == ["a-user", "z-user"]

    def test_method_attribute(self):
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(), method="isotonic",
        )
        assert store.method == "isotonic"


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestCalibrationSerialization:
    def test_temperature_round_trip(self, tmp_path: Path):
        cal = TemperatureCalibrator(temperature=1.5)
        path = tmp_path / "temp.json"
        save_calibrator(cal, path)
        loaded = load_calibrator(path)
        assert isinstance(loaded, TemperatureCalibrator)
        assert loaded.temperature == 1.5

    def test_identity_round_trip(self, tmp_path: Path):
        cal = IdentityCalibrator()
        path = tmp_path / "id.json"
        save_calibrator(cal, path)
        loaded = load_calibrator(path)
        assert isinstance(loaded, IdentityCalibrator)

    def test_isotonic_round_trip(self, tmp_path: Path):
        regs = _make_isotonic_regressors()
        cal = IsotonicCalibrator(regs)
        path = tmp_path / "iso.json"
        save_calibrator(cal, path)
        loaded = load_calibrator(path)

        assert isinstance(loaded, IsotonicCalibrator)
        assert loaded.n_classes == cal.n_classes

        probs = _random_probs(5)
        np.testing.assert_allclose(
            cal.calibrate(probs), loaded.calibrate(probs), atol=1e-10,
        )

    def test_store_round_trip_temperature(self, tmp_path: Path):
        store = CalibratorStore(
            global_calibrator=TemperatureCalibrator(temperature=1.3),
            user_calibrators={
                "user-A": TemperatureCalibrator(temperature=0.8),
                "user-B": TemperatureCalibrator(temperature=2.1),
            },
            method="temperature",
        )
        store_dir = tmp_path / "store"
        save_calibrator_store(store, store_dir)
        loaded = load_calibrator_store(store_dir)

        assert loaded.method == "temperature"
        assert sorted(loaded.user_calibrators) == ["user-A", "user-B"]
        assert isinstance(loaded.global_calibrator, TemperatureCalibrator)
        assert loaded.global_calibrator.temperature == 1.3

    def test_store_round_trip_isotonic(self, tmp_path: Path):
        regs_global = _make_isotonic_regressors()
        regs_user = _make_isotonic_regressors()
        store = CalibratorStore(
            global_calibrator=IsotonicCalibrator(regs_global),
            user_calibrators={"user-X": IsotonicCalibrator(regs_user)},
            method="isotonic",
        )
        store_dir = tmp_path / "iso_store"
        save_calibrator_store(store, store_dir)
        loaded = load_calibrator_store(store_dir)

        assert loaded.method == "isotonic"
        assert list(loaded.user_calibrators) == ["user-X"]
        assert isinstance(loaded.global_calibrator, IsotonicCalibrator)

        probs = _random_probs(3)
        np.testing.assert_allclose(
            store.global_calibrator.calibrate(probs),
            loaded.global_calibrator.calibrate(probs),
            atol=1e-10,
        )

    def test_store_empty_users(self, tmp_path: Path):
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(), method="temperature",
        )
        store_dir = tmp_path / "empty_store"
        save_calibrator_store(store, store_dir)
        loaded = load_calibrator_store(store_dir)
        assert len(loaded.user_calibrators) == 0


# ---------------------------------------------------------------------------
# PersonalizationEligibility
# ---------------------------------------------------------------------------


class TestPersonalizationEligibility:
    def _make_df(
        self,
        user_id: str,
        n_windows: int,
        n_days: int,
        n_labels: int,
    ) -> pd.DataFrame:
        labels = _SORTED_LABELS[:n_labels]
        rows = []
        for i in range(n_windows):
            day_offset = i % n_days
            rows.append({
                "user_id": user_id,
                "bucket_start_ts": pd.Timestamp(
                    f"2025-06-{15 + day_offset:02d}T{10 + i % 12:02d}:00:00"
                ),
                "label": labels[i % n_labels],
            })
        return pd.DataFrame(rows)

    def test_eligible(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=250, n_days=5, n_labels=4)
        result = check_personalization_eligible(df, "user-A")
        assert result.is_eligible is True
        assert result.labeled_windows == 250
        assert result.labeled_days == 5
        assert result.distinct_labels == 4

    def test_ineligible_too_few_windows(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=50, n_days=5, n_labels=4)
        result = check_personalization_eligible(df, "user-A")
        assert result.is_eligible is False

    def test_ineligible_too_few_days(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=250, n_days=2, n_labels=4)
        result = check_personalization_eligible(df, "user-A")
        assert result.is_eligible is False

    def test_ineligible_too_few_labels(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=250, n_days=5, n_labels=2)
        result = check_personalization_eligible(df, "user-A")
        assert result.is_eligible is False

    def test_missing_user(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=250, n_days=5, n_labels=4)
        result = check_personalization_eligible(df, "user-nonexistent")
        assert result.is_eligible is False
        assert result.labeled_windows == 0

    def test_custom_thresholds(self):
        from taskclf.train.calibrate import check_personalization_eligible

        df = self._make_df("user-A", n_windows=50, n_days=3, n_labels=3)
        result = check_personalization_eligible(
            df, "user-A", min_windows=40, min_days=2, min_labels=2,
        )
        assert result.is_eligible is True


# ---------------------------------------------------------------------------
# Calibrator fitting
# ---------------------------------------------------------------------------


class TestCalibratorFitting:
    def test_fit_temperature(self):
        from taskclf.train.calibrate import fit_temperature_calibrator

        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, _N_CLASSES, size=n)
        y_proba = _random_probs(n, seed=42)
        # Bias the probabilities slightly toward the true class
        for i in range(n):
            y_proba[i, y_true[i]] += 0.3
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        cal = fit_temperature_calibrator(y_true, y_proba)
        assert isinstance(cal, TemperatureCalibrator)
        assert cal.temperature > 0
        # Calibrated probs should still sum to 1
        calibrated = cal.calibrate(y_proba)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_isotonic(self):
        from taskclf.train.calibrate import fit_isotonic_calibrator

        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, _N_CLASSES, size=n)
        y_proba = _random_probs(n, seed=42)

        cal = fit_isotonic_calibrator(y_true, y_proba, _N_CLASSES)
        assert isinstance(cal, IsotonicCalibrator)
        assert cal.n_classes == _N_CLASSES

        calibrated = cal.calibrate(y_proba)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# fit_calibrator_store end-to-end (synthetic data, no real model)
# ---------------------------------------------------------------------------


def _build_synthetic_labeled_df(
    users: dict[str, int],
    n_days: int = 5,
) -> pd.DataFrame:
    """Build a minimal labeled DataFrame with realistic features for testing.

    ``users`` maps user_id -> number of windows.
    """
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.store import generate_dummy_labels

    all_frames = []
    for uid, n_windows in users.items():
        days_needed = max(1, n_days)
        per_day = max(1, n_windows // days_needed)
        for d in range(days_needed):
            if len(all_frames) > 0 and sum(len(f) for f in all_frames) >= n_windows:
                break
            date = dt.date(2025, 6, 15) + dt.timedelta(days=d)
            rows = generate_dummy_features(date, n_rows=per_day, user_id=uid)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(date, n_rows=per_day)
            for span in labels:
                mask = (
                    (df["bucket_start_ts"] >= span.start_ts)
                    & (df["bucket_start_ts"] < span.end_ts)
                )
                df.loc[mask, "label"] = span.label
            all_frames.append(df)

    result = pd.concat(all_frames, ignore_index=True)
    if "label" not in result.columns:
        result["label"] = _SORTED_LABELS[0]
    result["label"] = result["label"].fillna(_SORTED_LABELS[0])
    return result


class TestFitCalibratorStore:
    """Integration test: train a small LightGBM, then fit a calibrator store."""

    def _train_tiny_model(self):
        """Train a throwaway model on synthetic data for calibration tests."""
        from taskclf.features.build import generate_dummy_features
        from taskclf.labels.store import generate_dummy_labels
        from taskclf.train.lgbm import train_lgbm

        from taskclf.labels.projection import project_blocks_to_windows

        all_feats = []
        all_labels = []
        for d in range(3):
            date = dt.date(2025, 6, 15) + dt.timedelta(days=d)
            rows = generate_dummy_features(date, n_rows=60)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(date, n_rows=60)
            all_feats.append(df)
            all_labels.extend(labels)

        features_df = pd.concat(all_feats, ignore_index=True)
        labeled_df = project_blocks_to_windows(features_df, all_labels)

        if labeled_df.empty:
            pytest.skip("No labeled windows from synthetic data")

        from taskclf.train.dataset import split_by_time

        splits = split_by_time(labeled_df)
        train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
        val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)

        if train_df.empty or val_df.empty:
            pytest.skip("Insufficient data for train/val split")

        model, _metrics, _cm, _params, cat_encoders = train_lgbm(
            train_df, val_df, num_boost_round=5,
        )
        return model, cat_encoders, val_df

    def test_fit_store_temperature(self):
        from taskclf.train.calibrate import fit_calibrator_store

        model, cat_encoders, val_df = self._train_tiny_model()
        store, eligibility = fit_calibrator_store(
            model, val_df,
            cat_encoders=cat_encoders,
            method="temperature",
            min_windows=5,
            min_days=1,
            min_labels=1,
        )

        assert isinstance(store, CalibratorStore)
        assert store.method == "temperature"
        assert isinstance(store.global_calibrator, TemperatureCalibrator)
        assert len(eligibility) > 0

    def test_fit_store_isotonic(self):
        from taskclf.train.calibrate import fit_calibrator_store

        model, cat_encoders, val_df = self._train_tiny_model()
        store, eligibility = fit_calibrator_store(
            model, val_df,
            cat_encoders=cat_encoders,
            method="isotonic",
            min_windows=5,
            min_days=1,
            min_labels=1,
        )

        assert isinstance(store, CalibratorStore)
        assert store.method == "isotonic"
        assert isinstance(store.global_calibrator, IsotonicCalibrator)

    def test_ineligible_users_get_no_calibrator(self):
        from taskclf.train.calibrate import fit_calibrator_store

        model, cat_encoders, val_df = self._train_tiny_model()
        store, eligibility = fit_calibrator_store(
            model, val_df,
            cat_encoders=cat_encoders,
            method="temperature",
            min_windows=999999,
        )

        assert len(store.user_calibrators) == 0
        for e in eligibility:
            assert e.is_eligible is False


# ---------------------------------------------------------------------------
# Batch inference integration
# ---------------------------------------------------------------------------


class TestBatchInferenceWithStore:
    """Verify that run_batch_inference correctly uses a CalibratorStore."""

    def test_store_applied_per_user(self):
        from taskclf.infer.batch import run_batch_inference

        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={"user-A": TemperatureCalibrator(temperature=2.0)},
            method="temperature",
        )

        from taskclf.features.build import generate_dummy_features
        from taskclf.labels.projection import project_blocks_to_windows
        from taskclf.labels.store import generate_dummy_labels
        from taskclf.train.lgbm import train_lgbm
        from taskclf.train.dataset import split_by_time

        all_feats = []
        all_labels = []
        for d in range(3):
            date = dt.date(2025, 6, 15) + dt.timedelta(days=d)
            rows = generate_dummy_features(date, n_rows=60)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(date, n_rows=60)
            all_feats.append(df)
            all_labels.extend(labels)

        features_df = pd.concat(all_feats, ignore_index=True)
        labeled_df = project_blocks_to_windows(features_df, all_labels)

        if labeled_df.empty:
            pytest.skip("No labeled windows")

        splits = split_by_time(labeled_df)
        train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
        val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)

        if train_df.empty or val_df.empty:
            pytest.skip("Insufficient split data")

        model, _m, _c, _p, cat_encoders = train_lgbm(
            train_df, val_df, num_boost_round=5,
        )

        result_with_store = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
            calibrator_store=store,
        )

        result_without = run_batch_inference(
            model, features_df,
            cat_encoders=cat_encoders,
        )

        assert len(result_with_store.smoothed_labels) == len(result_without.smoothed_labels)
        assert result_with_store.core_probs.shape == result_without.core_probs.shape
