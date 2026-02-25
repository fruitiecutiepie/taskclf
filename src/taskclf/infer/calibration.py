"""Probability calibration hooks for post-model adjustment.

Provides a ``Calibrator`` protocol and concrete implementations:

* :class:`IdentityCalibrator` — no-op pass-through (default).
* :class:`TemperatureCalibrator` — single-parameter temperature scaling.
* :class:`IsotonicCalibrator` — per-class isotonic regression.

Per-user calibration is managed through :class:`CalibratorStore`, which
maps ``user_id`` to a fitted calibrator and falls back to a global
calibrator for users without enough labeled data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Calibrator(Protocol):
    """Minimal contract for a probability calibrator.

    Implementations must:

    * Preserve the shape of *core_probs*.
    * Ensure the output sums to 1.0 along the class axis.
    * Be deterministic (same input → same output).
    """

    def calibrate(self, core_probs: np.ndarray) -> np.ndarray:
        """Adjust a probability vector (or matrix) and return calibrated probabilities.

        Args:
            core_probs: Array of shape ``(n_classes,)`` or ``(n_rows, n_classes)``.

        Returns:
            Calibrated probabilities with the same shape.
        """
        ...  # pragma: no cover


class IdentityCalibrator:
    """No-op calibrator that returns probabilities unchanged."""

    def calibrate(self, core_probs: np.ndarray) -> np.ndarray:
        return core_probs


class TemperatureCalibrator:
    """Scale logits by a learned temperature before softmax.

    A temperature > 1 softens the distribution (less confident);
    a temperature < 1 sharpens it (more confident).

    Args:
        temperature: Positive scalar.  Defaults to 1.0 (identity).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def calibrate(self, core_probs: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return core_probs

        eps = 1e-12
        logits = np.log(np.clip(core_probs, eps, None))
        scaled = logits / self.temperature

        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum(axis=-1, keepdims=True)


class IsotonicCalibrator:
    """Per-class isotonic regression calibrator.

    Wraps one ``sklearn.isotonic.IsotonicRegression`` per class.  Each
    regressor maps the model's raw probability for that class to a
    calibrated value.  After per-class transformation the vector is
    renormalized to sum to 1.0.

    Args:
        regressors: List of fitted ``IsotonicRegression`` instances,
            one per class, ordered by label ID.
    """

    def __init__(self, regressors: list) -> None:
        if not regressors:
            raise ValueError("regressors list must not be empty")
        self._regressors = regressors

    @property
    def n_classes(self) -> int:
        return len(self._regressors)

    def calibrate(self, core_probs: np.ndarray) -> np.ndarray:
        single = core_probs.ndim == 1
        probs = core_probs[np.newaxis, :] if single else core_probs

        calibrated = np.empty_like(probs)
        for c, reg in enumerate(self._regressors):
            calibrated[:, c] = np.clip(reg.predict(probs[:, c]), 1e-12, None)

        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums

        return calibrated[0] if single else calibrated


class CalibratorStore:
    """Per-user calibrator registry with global fallback.

    Holds a global calibrator (fitted on all users' validation data)
    and optional per-user calibrators for users that meet the
    personalization eligibility thresholds.

    Args:
        global_calibrator: Calibrator applied to users without a
            dedicated calibrator.
        user_calibrators: Mapping from ``user_id`` to a fitted
            calibrator.  ``None`` or empty dict means all users fall
            back to the global calibrator.
        method: Calibration method label (``"temperature"`` or
            ``"isotonic"``).
    """

    def __init__(
        self,
        global_calibrator: Calibrator,
        user_calibrators: dict[str, Calibrator] | None = None,
        method: str = "temperature",
    ) -> None:
        self.global_calibrator = global_calibrator
        self.user_calibrators: dict[str, Calibrator] = user_calibrators or {}
        self.method = method

    def get_calibrator(self, user_id: str) -> Calibrator:
        """Return the per-user calibrator if available, else the global one."""
        return self.user_calibrators.get(user_id, self.global_calibrator)

    def calibrate_batch(
        self,
        core_probs: np.ndarray,
        user_ids: Sequence[str],
    ) -> np.ndarray:
        """Apply per-user calibration row by row.

        Args:
            core_probs: Probability matrix ``(n_rows, n_classes)``.
            user_ids: Sequence of user_id strings aligned with rows.

        Returns:
            Calibrated probability matrix of the same shape.
        """
        result = np.empty_like(core_probs)
        for i, uid in enumerate(user_ids):
            cal = self.get_calibrator(uid)
            result[i] = cal.calibrate(core_probs[i : i + 1])[0]
        return result

    @property
    def user_ids(self) -> list[str]:
        """Return sorted list of user IDs with per-user calibrators."""
        return sorted(self.user_calibrators)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_isotonic(calibrator: IsotonicCalibrator) -> dict:
    """Convert an IsotonicCalibrator to a JSON-safe dict."""
    regressors_data = []
    for reg in calibrator._regressors:
        regressors_data.append({
            "X_thresholds_": reg.X_thresholds_.tolist(),
            "y_thresholds_": reg.y_thresholds_.tolist(),
            "X_min_": float(reg.X_min_),
            "X_max_": float(reg.X_max_),
            "increasing_": bool(reg.increasing_),
            "out_of_bounds": reg.out_of_bounds,
        })
    return {"type": "isotonic", "regressors": regressors_data}


def _deserialize_isotonic(data: dict) -> IsotonicCalibrator:
    """Reconstruct an IsotonicCalibrator from a serialized dict."""
    from sklearn.isotonic import IsotonicRegression

    regressors = []
    for rd in data["regressors"]:
        reg = IsotonicRegression(
            increasing=rd["increasing_"],
            out_of_bounds=rd["out_of_bounds"],
            y_min=0.0,
            y_max=1.0,
        )
        x_thresh = np.array(rd["X_thresholds_"])
        y_thresh = np.array(rd["y_thresholds_"])
        reg.fit(x_thresh, y_thresh)
        regressors.append(reg)
    return IsotonicCalibrator(regressors)


def save_calibrator(calibrator: Calibrator, path: Path) -> Path:
    """Persist a calibrator to JSON.

    Supports :class:`IdentityCalibrator`, :class:`TemperatureCalibrator`,
    and :class:`IsotonicCalibrator`.

    Args:
        calibrator: Calibrator instance to serialize.
        path: Destination file path.

    Returns:
        The *path* that was written.
    """
    if isinstance(calibrator, TemperatureCalibrator):
        data = {"type": "temperature", "temperature": calibrator.temperature}
    elif isinstance(calibrator, IsotonicCalibrator):
        data = _serialize_isotonic(calibrator)
    elif isinstance(calibrator, IdentityCalibrator):
        data = {"type": "identity"}
    else:
        raise TypeError(f"Cannot serialize calibrator of type {type(calibrator).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_calibrator(path: Path) -> Calibrator:
    """Load a calibrator from a JSON file written by :func:`save_calibrator`.

    Args:
        path: Path to a calibrator JSON file.

    Returns:
        A :class:`Calibrator` instance.

    Raises:
        ValueError: If the file contains an unknown calibrator type.
    """
    data = json.loads(path.read_text())
    cal_type = data.get("type", "identity")

    if cal_type == "identity":
        return IdentityCalibrator()
    if cal_type == "temperature":
        return TemperatureCalibrator(temperature=data["temperature"])
    if cal_type == "isotonic":
        return _deserialize_isotonic(data)

    raise ValueError(f"Unknown calibrator type: {cal_type!r}")


def save_calibrator_store(store: CalibratorStore, path: Path) -> Path:
    """Persist a :class:`CalibratorStore` to a directory.

    Layout::

        path/
            store.json          # metadata + global calibrator
            users/
                <user_id>.json  # per-user calibrator

    Args:
        store: Store to serialize.
        path: Target directory (created if needed).

    Returns:
        The directory *path*.
    """
    path.mkdir(parents=True, exist_ok=True)

    global_path = path / "global.json"
    save_calibrator(store.global_calibrator, global_path)

    meta = {
        "method": store.method,
        "user_count": len(store.user_calibrators),
        "user_ids": sorted(store.user_calibrators),
    }
    (path / "store.json").write_text(json.dumps(meta, indent=2))

    if store.user_calibrators:
        users_dir = path / "users"
        users_dir.mkdir(exist_ok=True)
        for uid, cal in store.user_calibrators.items():
            save_calibrator(cal, users_dir / f"{uid}.json")

    return path


def load_calibrator_store(path: Path) -> CalibratorStore:
    """Load a :class:`CalibratorStore` from a directory.

    Args:
        path: Directory previously written by :func:`save_calibrator_store`.

    Returns:
        A populated ``CalibratorStore``.
    """
    meta = json.loads((path / "store.json").read_text())
    global_cal = load_calibrator(path / "global.json")

    user_cals: dict[str, Calibrator] = {}
    users_dir = path / "users"
    for uid in meta.get("user_ids", []):
        user_path = users_dir / f"{uid}.json"
        if user_path.exists():
            user_cals[uid] = load_calibrator(user_path)

    return CalibratorStore(
        global_calibrator=global_cal,
        user_calibrators=user_cals,
        method=meta.get("method", "temperature"),
    )
