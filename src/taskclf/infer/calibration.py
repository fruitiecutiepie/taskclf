"""Probability calibration hooks for post-model adjustment.

Provides a ``Calibrator`` protocol and two concrete implementations:

* :class:`IdentityCalibrator` — no-op pass-through (default).
* :class:`TemperatureCalibrator` — single-parameter temperature scaling.

The actual *training* of per-user calibrators is out of scope here
(see TODO 6, section 20).  This module only defines the interface so
that :class:`~taskclf.infer.online.OnlinePredictor` can optionally
apply a calibrator between raw model output and the reject decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

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


def save_calibrator(calibrator: Calibrator, path: Path) -> Path:
    """Persist a calibrator to JSON.

    Currently supports :class:`IdentityCalibrator` and
    :class:`TemperatureCalibrator`.

    Args:
        calibrator: Calibrator instance to serialize.
        path: Destination file path.

    Returns:
        The *path* that was written.
    """
    if isinstance(calibrator, TemperatureCalibrator):
        data = {"type": "temperature", "temperature": calibrator.temperature}
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

    raise ValueError(f"Unknown calibrator type: {cal_type!r}")
