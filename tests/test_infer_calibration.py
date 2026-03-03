"""Tests for IsotonicCalibrator, CalibratorStore, and store persistence.

Covers the untested parts of ``taskclf.infer.calibration`` (TODO Item 20):
- IsotonicCalibrator (TC-CAL-001 through TC-CAL-006)
- CalibratorStore (TC-CAL-007 through TC-CAL-012)
- save_calibrator_store / load_calibrator_store (TC-CAL-013 through TC-CAL-017)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from taskclf.infer.calibration import (
    Calibrator,
    CalibratorStore,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    load_calibrator,
    load_calibrator_store,
    save_calibrator,
    save_calibrator_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_isotonic_regressors(n_classes: int = 8) -> list[IsotonicRegression]:
    """Create fitted IsotonicRegression instances with synthetic data."""
    regressors = []
    rng = np.random.default_rng(42)
    for _ in range(n_classes):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.sort(rng.uniform(0.0, 1.0, size=5))
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        reg.fit(x, y)
        regressors.append(reg)
    return regressors


def _make_isotonic_calibrator(n_classes: int = 8) -> IsotonicCalibrator:
    return IsotonicCalibrator(_make_isotonic_regressors(n_classes))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    """TC-CAL-001 through TC-CAL-006."""

    def test_1d_calibrate(self) -> None:
        """TC-CAL-001: 1D input -> 1D output summing to 1.0."""
        cal = _make_isotonic_calibrator()
        probs = np.array([0.125] * 8)
        result = cal.calibrate(probs)
        assert result.shape == (8,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_2d_calibrate(self) -> None:
        """TC-CAL-002: 2D input -> 2D output, each row sums to 1.0."""
        cal = _make_isotonic_calibrator()
        probs = np.array([[0.125] * 8, [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]])
        result = cal.calibrate(probs)
        assert result.shape == (2, 8)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_empty_regressors_raises(self) -> None:
        """TC-CAL-003: Empty regressors list raises ValueError."""
        with pytest.raises(ValueError, match="regressors list must not be empty"):
            IsotonicCalibrator([])

    def test_n_classes_property(self) -> None:
        """TC-CAL-004: n_classes matches number of regressors."""
        cal = _make_isotonic_calibrator(8)
        assert cal.n_classes == 8
        cal3 = _make_isotonic_calibrator(3)
        assert cal3.n_classes == 3

    def test_satisfies_calibrator_protocol(self) -> None:
        """TC-CAL-005: IsotonicCalibrator satisfies the Calibrator protocol."""
        cal = _make_isotonic_calibrator()
        assert isinstance(cal, Calibrator)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """TC-CAL-006: save then load produces same calibration results."""
        cal = _make_isotonic_calibrator()
        probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        expected = cal.calibrate(probs)

        path = save_calibrator(cal, tmp_path / "iso.json")
        loaded = load_calibrator(path)

        assert isinstance(loaded, IsotonicCalibrator)
        result = loaded.calibrate(probs)
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# CalibratorStore
# ---------------------------------------------------------------------------


class TestCalibratorStore:
    """TC-CAL-007 through TC-CAL-012."""

    def test_get_calibrator_returns_per_user(self) -> None:
        """TC-CAL-007: Known user_id returns that user's calibrator."""
        user_cal = TemperatureCalibrator(2.0)
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={"alice": user_cal},
        )
        assert store.get_calibrator("alice") is user_cal

    def test_get_calibrator_falls_back_to_global(self) -> None:
        """TC-CAL-008: Unknown user_id falls back to global."""
        global_cal = IdentityCalibrator()
        store = CalibratorStore(
            global_calibrator=global_cal,
            user_calibrators={"alice": TemperatureCalibrator(2.0)},
        )
        assert store.get_calibrator("bob") is global_cal

    def test_calibrate_batch_per_user(self) -> None:
        """TC-CAL-009: Different users get different calibrations."""
        probs = np.array([
            [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025],
            [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025],
        ])
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={"alice": TemperatureCalibrator(5.0)},
        )
        result = store.calibrate_batch(probs, ["alice", "bob"])
        # alice gets temperature scaling (softer), bob gets identity
        np.testing.assert_array_equal(result[1], probs[1])
        assert not np.allclose(result[0], probs[0])

    def test_calibrate_batch_shape_preserved(self) -> None:
        """TC-CAL-010: Input (N, K) -> output (N, K)."""
        probs = np.array([[0.125] * 8] * 5)
        store = CalibratorStore(global_calibrator=IdentityCalibrator())
        result = store.calibrate_batch(probs, ["u"] * 5)
        assert result.shape == (5, 8)

    def test_user_ids_property(self) -> None:
        """TC-CAL-011: Returns sorted list of per-user keys."""
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={
                "charlie": TemperatureCalibrator(1.5),
                "alice": TemperatureCalibrator(2.0),
            },
        )
        assert store.user_ids == ["alice", "charlie"]

    def test_empty_user_calibrators_fallback(self) -> None:
        """TC-CAL-012: All users fall back to global when no per-user cals."""
        global_cal = IdentityCalibrator()
        store = CalibratorStore(global_calibrator=global_cal)
        assert store.get_calibrator("anyone") is global_cal
        assert store.user_ids == []


# ---------------------------------------------------------------------------
# save_calibrator_store / load_calibrator_store
# ---------------------------------------------------------------------------


class TestCalibratorStorePersistence:
    """TC-CAL-013 through TC-CAL-017."""

    def test_roundtrip_temperature_with_per_user(self, tmp_path: Path) -> None:
        """TC-CAL-013: Round-trip with temperature global + 2 per-user."""
        store = CalibratorStore(
            global_calibrator=TemperatureCalibrator(1.5),
            user_calibrators={
                "alice": TemperatureCalibrator(2.0),
                "bob": TemperatureCalibrator(3.0),
            },
            method="temperature",
        )
        store_path = tmp_path / "cal_store"
        save_calibrator_store(store, store_path)
        loaded = load_calibrator_store(store_path)

        assert isinstance(loaded.global_calibrator, TemperatureCalibrator)
        assert loaded.global_calibrator.temperature == 1.5
        assert loaded.user_ids == ["alice", "bob"]
        assert isinstance(loaded.user_calibrators["alice"], TemperatureCalibrator)
        assert loaded.user_calibrators["alice"].temperature == 2.0
        assert loaded.user_calibrators["bob"].temperature == 3.0

    def test_roundtrip_isotonic_global(self, tmp_path: Path) -> None:
        """TC-CAL-014: Isotonic global calibrator survives serialization."""
        cal = _make_isotonic_calibrator()
        probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        expected = cal.calibrate(probs)

        store = CalibratorStore(
            global_calibrator=cal,
            method="isotonic",
        )
        store_path = tmp_path / "iso_store"
        save_calibrator_store(store, store_path)
        loaded = load_calibrator_store(store_path)

        assert isinstance(loaded.global_calibrator, IsotonicCalibrator)
        result = loaded.global_calibrator.calibrate(probs)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_directory_layout(self, tmp_path: Path) -> None:
        """TC-CAL-015: Correct files created under store dir."""
        store = CalibratorStore(
            global_calibrator=TemperatureCalibrator(1.0),
            user_calibrators={"u1": TemperatureCalibrator(2.0)},
        )
        store_path = tmp_path / "store_dir"
        save_calibrator_store(store, store_path)

        assert (store_path / "store.json").exists()
        assert (store_path / "global.json").exists()
        assert (store_path / "users" / "u1.json").exists()

    def test_store_json_metadata(self, tmp_path: Path) -> None:
        """TC-CAL-016: store.json contains method, user_count, user_ids."""
        store = CalibratorStore(
            global_calibrator=IdentityCalibrator(),
            user_calibrators={"a": TemperatureCalibrator(1.5), "b": TemperatureCalibrator(2.5)},
            method="temperature",
        )
        store_path = tmp_path / "meta_store"
        save_calibrator_store(store, store_path)

        meta = json.loads((store_path / "store.json").read_text())
        assert meta["method"] == "temperature"
        assert meta["user_count"] == 2
        assert sorted(meta["user_ids"]) == ["a", "b"]

    def test_empty_per_user_dict(self, tmp_path: Path) -> None:
        """TC-CAL-017: No users/ directory created, still loads fine."""
        store = CalibratorStore(
            global_calibrator=TemperatureCalibrator(1.0),
            method="temperature",
        )
        store_path = tmp_path / "empty_users"
        save_calibrator_store(store, store_path)

        assert not (store_path / "users").exists()

        loaded = load_calibrator_store(store_path)
        assert loaded.user_ids == []
        assert isinstance(loaded.global_calibrator, TemperatureCalibrator)
