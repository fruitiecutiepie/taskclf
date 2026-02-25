"""Pure statistical functions for drift and distribution-shift detection.

All functions operate on numerical aggregates and prediction outputs only.
No raw content (keystrokes, titles, URLs) is ever accessed or stored.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from pydantic import BaseModel

from taskclf.core.defaults import (
    DEFAULT_CLASS_SHIFT_THRESHOLD,
    DEFAULT_ENTROPY_SPIKE_MULTIPLIER,
    DEFAULT_KS_ALPHA,
    DEFAULT_PSI_BINS,
    DEFAULT_PSI_THRESHOLD,
    DEFAULT_REJECT_RATE_INCREASE_THRESHOLD,
    MIXED_UNKNOWN,
)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class KsResult(BaseModel):
    """Result of a two-sample Kolmogorov-Smirnov test."""

    statistic: float
    p_value: float

    def is_significant(self, alpha: float = DEFAULT_KS_ALPHA) -> bool:
        return self.p_value < alpha


class FeatureDriftResult(BaseModel):
    """Drift metrics for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    is_drifted: bool


class FeatureDriftReport(BaseModel):
    """Aggregated drift report across multiple features."""

    results: list[FeatureDriftResult]
    flagged_features: list[str]
    timestamp: datetime


class RejectRateDrift(BaseModel):
    """Reject-rate comparison between reference and current windows."""

    ref_rate: float
    cur_rate: float
    increase: float
    is_flagged: bool


class EntropyDrift(BaseModel):
    """Mean prediction-entropy comparison."""

    ref_mean_entropy: float
    cur_mean_entropy: float
    ratio: float
    is_flagged: bool


class ClassShiftResult(BaseModel):
    """Class-distribution shift between reference and current."""

    ref_dist: dict[str, float]
    cur_dist: dict[str, float]
    max_shift: float
    shifted_classes: list[str]
    is_flagged: bool


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = DEFAULT_PSI_BINS,
) -> float:
    """Population Stability Index between two 1-D distributions.

    Uses equal-frequency (quantile) binning derived from *reference*.
    Returns 0.0 when either array is empty or constant.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]

    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref_clean, quantiles))
    if len(edges) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(cur_clean, bins=edges)[0].astype(np.float64)

    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()
    if ref_total == 0 or cur_total == 0:
        return 0.0

    ref_pct = ref_counts / ref_total
    cur_pct = cur_counts / cur_total

    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> KsResult:
    """Two-sample Kolmogorov-Smirnov test.

    Returns ``KsResult`` with ``statistic`` and ``p_value``.
    Requires ``scipy`` (transitively available via scikit-learn).
    """
    from scipy.stats import ks_2samp

    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]

    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return KsResult(statistic=0.0, p_value=1.0)

    stat, p = ks_2samp(ref_clean, cur_clean)
    return KsResult(statistic=float(stat), p_value=float(p))


def _prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy of a probability matrix."""
    probs = np.clip(probs, _EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


# ---------------------------------------------------------------------------
# High-level detectors
# ---------------------------------------------------------------------------


def feature_drift_report(
    ref_df: "pd.DataFrame",
    cur_df: "pd.DataFrame",
    numerical_features: Sequence[str],
    *,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> FeatureDriftReport:
    """Run PSI + KS across *numerical_features* and flag drifted ones.

    Args:
        ref_df: Reference-period feature DataFrame.
        cur_df: Current-period feature DataFrame.
        numerical_features: Column names to test.
        psi_threshold: PSI above this marks a feature as drifted.
        ks_alpha: Significance level for the KS test.

    Returns:
        A :class:`FeatureDriftReport` with per-feature results.
    """
    results: list[FeatureDriftResult] = []
    flagged: list[str] = []

    for feat in numerical_features:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue

        ref_vals = ref_df[feat].to_numpy(dtype=np.float64, na_value=np.nan)
        cur_vals = cur_df[feat].to_numpy(dtype=np.float64, na_value=np.nan)

        psi = compute_psi(ref_vals, cur_vals)
        ks = compute_ks(ref_vals, cur_vals)
        drifted = psi > psi_threshold or ks.is_significant(ks_alpha)

        results.append(FeatureDriftResult(
            feature=feat,
            psi=round(psi, 6),
            ks_statistic=round(ks.statistic, 6),
            ks_p_value=round(ks.p_value, 6),
            is_drifted=drifted,
        ))
        if drifted:
            flagged.append(feat)

    return FeatureDriftReport(
        results=results,
        flagged_features=flagged,
        timestamp=datetime.now(tz=timezone.utc),
    )


def detect_reject_rate_increase(
    ref_labels: Sequence[str],
    cur_labels: Sequence[str],
    *,
    threshold: float = DEFAULT_REJECT_RATE_INCREASE_THRESHOLD,
    reject_label: str = MIXED_UNKNOWN,
) -> RejectRateDrift:
    """Flag if reject rate increased by >= *threshold* (absolute).

    Args:
        ref_labels: Reference-period predicted labels.
        cur_labels: Current-period predicted labels.
        threshold: Absolute increase that triggers the flag.
        reject_label: The label treated as a reject.

    Returns:
        A :class:`RejectRateDrift` with comparison data.
    """
    def _rate(labels: Sequence[str]) -> float:
        if not labels:
            return 0.0
        return sum(1 for l in labels if l == reject_label) / len(labels)

    ref_rate = _rate(ref_labels)
    cur_rate = _rate(cur_labels)
    increase = cur_rate - ref_rate

    return RejectRateDrift(
        ref_rate=round(ref_rate, 4),
        cur_rate=round(cur_rate, 4),
        increase=round(increase, 4),
        is_flagged=increase >= threshold,
    )


def detect_entropy_spike(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
    *,
    spike_multiplier: float = DEFAULT_ENTROPY_SPIKE_MULTIPLIER,
) -> EntropyDrift:
    """Flag if mean prediction entropy spiked.

    Args:
        ref_probs: Reference probability matrix ``(n, k)``.
        cur_probs: Current probability matrix ``(n, k)``.
        spike_multiplier: Current mean entropy must exceed
            ``spike_multiplier * reference`` to be flagged.

    Returns:
        An :class:`EntropyDrift` with comparison data.
    """
    ref_probs = np.asarray(ref_probs, dtype=np.float64)
    cur_probs = np.asarray(cur_probs, dtype=np.float64)

    ref_ent = float(np.mean(_prediction_entropy(ref_probs))) if len(ref_probs) else 0.0
    cur_ent = float(np.mean(_prediction_entropy(cur_probs))) if len(cur_probs) else 0.0

    ratio = cur_ent / ref_ent if ref_ent > _EPS else 0.0

    return EntropyDrift(
        ref_mean_entropy=round(ref_ent, 6),
        cur_mean_entropy=round(cur_ent, 6),
        ratio=round(ratio, 4),
        is_flagged=cur_ent > spike_multiplier * ref_ent if ref_ent > _EPS else False,
    )


def detect_class_shift(
    ref_labels: Sequence[str],
    cur_labels: Sequence[str],
    *,
    threshold: float = DEFAULT_CLASS_SHIFT_THRESHOLD,
) -> ClassShiftResult:
    """Flag if any class proportion changed by more than *threshold*.

    Args:
        ref_labels: Reference-period labels.
        cur_labels: Current-period labels.
        threshold: Absolute proportion change that triggers the flag.

    Returns:
        A :class:`ClassShiftResult` with per-class distributions.
    """
    def _dist(labels: Sequence[str]) -> dict[str, float]:
        counts = Counter(labels)
        total = sum(counts.values()) or 1
        return {k: round(v / total, 4) for k, v in sorted(counts.items())}

    ref_dist = _dist(ref_labels)
    cur_dist = _dist(cur_labels)

    all_classes = sorted(set(ref_dist) | set(cur_dist))
    shifted: list[str] = []
    max_shift = 0.0

    for cls in all_classes:
        delta = abs(cur_dist.get(cls, 0.0) - ref_dist.get(cls, 0.0))
        if delta > max_shift:
            max_shift = delta
        if delta > threshold:
            shifted.append(cls)

    return ClassShiftResult(
        ref_dist=ref_dist,
        cur_dist=cur_dist,
        max_shift=round(max_shift, 4),
        shifted_classes=shifted,
        is_flagged=len(shifted) > 0,
    )
