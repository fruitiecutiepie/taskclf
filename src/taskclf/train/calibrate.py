"""Per-user probability calibration: eligibility checks and calibrator fitting.

Provides the training-side logic for the personalization pipeline:

* :func:`check_personalization_eligible` — gate that ensures a user has
  enough labeled data before fitting a per-user calibrator.
* :func:`fit_temperature_calibrator` — optimizes a temperature scalar
  that minimizes NLL on held-out probabilities.
* :func:`fit_isotonic_calibrator` — fits per-class isotonic regression.
* :func:`fit_calibrator_store` — orchestrates the full flow: predict on
  validation data, fit a global calibrator, check each user's eligibility,
  and fit per-user calibrators for qualifying users.
"""

from __future__ import annotations

import logging
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import (
    DEFAULT_CALIBRATION_METHOD,
    DEFAULT_MIN_DISTINCT_LABELS,
    DEFAULT_MIN_LABELED_DAYS,
    DEFAULT_MIN_LABELED_WINDOWS,
)
from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.batch import predict_proba
from taskclf.infer.calibration import (
    CalibratorStore,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
)

logger = logging.getLogger(__name__)


class PersonalizationEligibility(BaseModel, frozen=True):
    """Result of checking whether a user qualifies for per-user calibration."""

    user_id: str
    labeled_windows: int
    labeled_days: int
    distinct_labels: int
    is_eligible: bool


def check_personalization_eligible(
    df: pd.DataFrame,
    user_id: str,
    *,
    min_windows: int = DEFAULT_MIN_LABELED_WINDOWS,
    min_days: int = DEFAULT_MIN_LABELED_DAYS,
    min_labels: int = DEFAULT_MIN_DISTINCT_LABELS,
) -> PersonalizationEligibility:
    """Check whether *user_id* has enough labeled data for per-user calibration.

    The eligibility thresholds follow ``docs/guide/acceptance.md`` Section 8.

    Args:
        df: Labeled DataFrame with ``user_id``, ``bucket_start_ts``, and
            ``label`` columns.
        user_id: The user to check.
        min_windows: Minimum labeled window count.
        min_days: Minimum number of distinct calendar days.
        min_labels: Minimum number of distinct core labels observed.

    Returns:
        A :class:`PersonalizationEligibility` report.
    """
    user_df = df[df["user_id"] == user_id]
    n_windows = len(user_df)

    if n_windows == 0:
        return PersonalizationEligibility(
            user_id=user_id,
            labeled_windows=0,
            labeled_days=0,
            distinct_labels=0,
            is_eligible=False,
        )

    n_days = user_df["bucket_start_ts"].dt.date.nunique()
    n_labels = user_df["label"].nunique()
    eligible = n_windows >= min_windows and n_days >= min_days and n_labels >= min_labels

    return PersonalizationEligibility(
        user_id=user_id,
        labeled_windows=n_windows,
        labeled_days=n_days,
        distinct_labels=n_labels,
        is_eligible=eligible,
    )


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of the true class probabilities."""
    eps = 1e-12
    selected = probs[np.arange(len(labels)), labels]
    return -float(np.mean(np.log(np.clip(selected, eps, None))))


def fit_temperature_calibrator(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
) -> TemperatureCalibrator:
    """Find the temperature that minimizes NLL on validation data.

    Uses a two-pass grid search: coarse (0.1–5.0 step 0.1), then fine
    (±0.1 around best at step 0.01).

    Args:
        y_true_indices: Integer-encoded true labels, shape ``(n,)``.
        y_proba: Raw model probabilities, shape ``(n, n_classes)``.

    Returns:
        A fitted :class:`TemperatureCalibrator`.
    """
    eps = 1e-12
    logits = np.log(np.clip(y_proba, eps, None))

    def _apply_temp(t: float) -> float:
        scaled = logits / t
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum(axis=-1, keepdims=True)
        return _nll(probs, y_true_indices)

    # Coarse pass
    best_t = 1.0
    best_nll = _apply_temp(1.0)
    for t in np.arange(0.1, 5.05, 0.1):
        nll = _apply_temp(float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    # Fine pass around best
    lo = max(0.01, best_t - 0.1)
    hi = best_t + 0.1
    for t in np.arange(lo, hi + 0.005, 0.01):
        nll = _apply_temp(float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return TemperatureCalibrator(temperature=round(best_t, 4))


def fit_isotonic_calibrator(
    y_true_indices: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> IsotonicCalibrator:
    """Fit per-class isotonic regression on validation data.

    For each class *c*, fits ``IsotonicRegression`` on
    ``(y_proba[:, c], (y_true == c).astype(float))``.

    Args:
        y_true_indices: Integer-encoded true labels, shape ``(n,)``.
        y_proba: Raw model probabilities, shape ``(n, n_classes)``.
        n_classes: Number of classes.

    Returns:
        A fitted :class:`IsotonicCalibrator`.
    """
    regressors: list[IsotonicRegression] = []
    for c in range(n_classes):
        binary_target = (y_true_indices == c).astype(np.float64)
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        reg.fit(y_proba[:, c], binary_target)
        regressors.append(reg)
    return IsotonicCalibrator(regressors)


def fit_calibrator_store(
    model: lgb.Booster,
    labeled_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    method: Literal["temperature", "isotonic"] = DEFAULT_CALIBRATION_METHOD,  # type: ignore[assignment]
    min_windows: int = DEFAULT_MIN_LABELED_WINDOWS,
    min_days: int = DEFAULT_MIN_LABELED_DAYS,
    min_labels: int = DEFAULT_MIN_DISTINCT_LABELS,
) -> tuple[CalibratorStore, list[PersonalizationEligibility]]:
    """Fit a global calibrator and per-user calibrators for eligible users.

    1. Predicts on the full *labeled_df* to get raw probabilities.
    2. Fits a global calibrator on all validation data.
    3. Checks each user's eligibility; for eligible users fits a
       per-user calibrator.

    Args:
        model: Trained LightGBM booster (frozen — not retrained).
        labeled_df: Labeled validation DataFrame with ``user_id``,
            ``bucket_start_ts``, ``label``, and ``FEATURE_COLUMNS``.
        cat_encoders: Pre-fitted categorical encoders from training.
        method: Calibration method — ``"temperature"`` or ``"isotonic"``.
        min_windows: Minimum labeled windows for per-user eligibility.
        min_days: Minimum distinct days for per-user eligibility.
        min_labels: Minimum distinct labels for per-user eligibility.

    Returns:
        ``(store, eligibility_reports)`` — a :class:`CalibratorStore`
        and a list of :class:`PersonalizationEligibility` for every
        unique user in *labeled_df*.
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))
    n_classes = len(le.classes_)

    y_proba = predict_proba(model, labeled_df, cat_encoders)
    y_true = le.transform(labeled_df["label"].values)

    # Global calibrator
    if method == "isotonic":
        global_cal = fit_isotonic_calibrator(y_true, y_proba, n_classes)
    else:
        global_cal = fit_temperature_calibrator(y_true, y_proba)

    # Per-user eligibility and calibration
    user_ids = sorted(labeled_df["user_id"].unique())
    eligibility_reports: list[PersonalizationEligibility] = []
    user_calibrators: dict[str, TemperatureCalibrator | IsotonicCalibrator] = {}

    for uid in user_ids:
        elig = check_personalization_eligible(
            labeled_df, uid,
            min_windows=min_windows,
            min_days=min_days,
            min_labels=min_labels,
        )
        eligibility_reports.append(elig)

        if not elig.is_eligible:
            logger.info(
                "User %s ineligible (windows=%d, days=%d, labels=%d)",
                uid, elig.labeled_windows, elig.labeled_days, elig.distinct_labels,
            )
            continue

        mask = labeled_df["user_id"].values == uid
        user_proba = y_proba[mask]
        user_true = y_true[mask]

        if method == "isotonic":
            user_cal = fit_isotonic_calibrator(user_true, user_proba, n_classes)
        else:
            user_cal = fit_temperature_calibrator(user_true, user_proba)

        user_calibrators[uid] = user_cal
        logger.info("Fitted %s calibrator for user %s", method, uid)

    store = CalibratorStore(
        global_calibrator=global_cal,
        user_calibrators=user_calibrators,
        method=method,
    )
    return store, eligibility_reports
