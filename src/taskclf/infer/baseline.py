"""Rule-based baseline classifier (no ML).

Applies heuristic rules to feature windows in priority order:

1. BreakIdle — near-zero activity or long idle run
2. ReadResearch — browser foreground with high scroll and low typing
3. Build — editor/terminal foreground with high typing and shortcuts
4. Mixed/Unknown — fallback reject label

This establishes the cold-start performance floor that ML models must beat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from taskclf.core.defaults import (
    BASELINE_IDLE_ACTIVE_THRESHOLD,
    BASELINE_IDLE_RUN_THRESHOLD,
    BASELINE_KEYS_HIGH,
    BASELINE_KEYS_LOW,
    BASELINE_SCROLL_HIGH,
    BASELINE_SHORTCUT_HIGH,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_SMOOTH_WINDOW,
    MIXED_UNKNOWN,
)
from taskclf.core.types import CoreLabel
from taskclf.infer.smooth import Segment, rolling_majority, segmentize


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to float, treating None/NaN as *default*."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def classify_single_row(
    row: pd.Series,
    *,
    idle_active_threshold: float = BASELINE_IDLE_ACTIVE_THRESHOLD,
    idle_run_threshold: float = BASELINE_IDLE_RUN_THRESHOLD,
    scroll_high: float = BASELINE_SCROLL_HIGH,
    keys_low: float = BASELINE_KEYS_LOW,
    keys_high: float = BASELINE_KEYS_HIGH,
    shortcut_high: float = BASELINE_SHORTCUT_HIGH,
) -> str:
    """Classify a single feature row using heuristic rules.

    Rules are applied in strict priority order so that BreakIdle always
    overrides other signals (e.g. an idle browser window is BreakIdle,
    not ReadResearch).

    Args:
        row: A pandas Series with feature columns from ``features_v1``.
        idle_active_threshold: Max ``active_seconds_any`` to be considered idle.
        idle_run_threshold: Min ``max_idle_run_seconds`` to be considered idle.
        scroll_high: Min ``scroll_events_per_min`` for "high scroll".
        keys_low: Max ``keys_per_min`` for "low typing".
        keys_high: Min ``keys_per_min`` for "high typing".
        shortcut_high: Min ``shortcut_rate`` for "high shortcuts".

    Returns:
        A label string: one of the :class:`CoreLabel` values or
        :data:`MIXED_UNKNOWN`.
    """
    active_any = row.get("active_seconds_any")
    max_idle = row.get("max_idle_run_seconds")

    # Rule 1: BreakIdle — near-zero activity or dominant idle run
    active_val = _safe_float(active_any, default=-1.0)
    idle_val = _safe_float(max_idle, default=0.0)

    if active_val < 0:
        # null active_seconds_any → no collector data → treat as idle
        return CoreLabel.BreakIdle
    if active_val < idle_active_threshold:
        return CoreLabel.BreakIdle
    if idle_val > idle_run_threshold:
        return CoreLabel.BreakIdle

    # Rule 2: ReadResearch — browser + high scroll + low keys
    is_browser = bool(row.get("is_browser", False))
    scroll = _safe_float(row.get("scroll_events_per_min"))
    keys = _safe_float(row.get("keys_per_min"))

    if is_browser and scroll > scroll_high and keys < keys_low:
        return CoreLabel.ReadResearch

    # Rule 3: Build — editor/terminal + high keys + shortcuts
    is_editor = bool(row.get("is_editor", False))
    is_terminal = bool(row.get("is_terminal", False))
    shortcut = _safe_float(row.get("shortcut_rate"))

    if (is_editor or is_terminal) and keys > keys_high and shortcut > shortcut_high:
        return CoreLabel.Build

    # Rule 4: fallback
    return MIXED_UNKNOWN


def predict_baseline(
    features_df: pd.DataFrame,
    *,
    idle_active_threshold: float = BASELINE_IDLE_ACTIVE_THRESHOLD,
    idle_run_threshold: float = BASELINE_IDLE_RUN_THRESHOLD,
    scroll_high: float = BASELINE_SCROLL_HIGH,
    keys_low: float = BASELINE_KEYS_LOW,
    keys_high: float = BASELINE_KEYS_HIGH,
    shortcut_high: float = BASELINE_SHORTCUT_HIGH,
) -> list[str]:
    """Apply heuristic rules to every row and return one label per window.

    Args:
        features_df: Feature DataFrame conforming to ``features_v1``.
        idle_active_threshold: Threshold for ``active_seconds_any``.
        idle_run_threshold: Threshold for ``max_idle_run_seconds``.
        scroll_high: Threshold for ``scroll_events_per_min``.
        keys_low: Upper bound on ``keys_per_min`` for "low typing".
        keys_high: Lower bound on ``keys_per_min`` for "high typing".
        shortcut_high: Lower bound on ``shortcut_rate``.

    Returns:
        Predicted label per row (same length as *features_df*).
    """
    return [
        classify_single_row(
            row,
            idle_active_threshold=idle_active_threshold,
            idle_run_threshold=idle_run_threshold,
            scroll_high=scroll_high,
            keys_low=keys_low,
            keys_high=keys_high,
            shortcut_high=shortcut_high,
        )
        for _, row in features_df.iterrows()
    ]


def run_baseline_inference(
    features_df: pd.DataFrame,
    *,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    idle_active_threshold: float = BASELINE_IDLE_ACTIVE_THRESHOLD,
    idle_run_threshold: float = BASELINE_IDLE_RUN_THRESHOLD,
    scroll_high: float = BASELINE_SCROLL_HIGH,
    keys_low: float = BASELINE_KEYS_LOW,
    keys_high: float = BASELINE_KEYS_HIGH,
    shortcut_high: float = BASELINE_SHORTCUT_HIGH,
) -> tuple[list[str], list[Segment]]:
    """Predict, smooth, and segmentize using the rule baseline.

    Mirrors :func:`taskclf.infer.batch.run_batch_inference` but uses
    heuristic rules instead of a trained model.

    Args:
        features_df: Feature DataFrame (must contain feature columns
            and ``bucket_start_ts``).
        smooth_window: Window size for rolling-majority smoothing.
        bucket_seconds: Width of each time bucket in seconds.
        idle_active_threshold: Threshold for ``active_seconds_any``.
        idle_run_threshold: Threshold for ``max_idle_run_seconds``.
        scroll_high: Threshold for ``scroll_events_per_min``.
        keys_low: Upper bound on ``keys_per_min`` for "low typing".
        keys_high: Lower bound on ``keys_per_min`` for "high typing".
        shortcut_high: Lower bound on ``shortcut_rate``.

    Returns:
        A ``(smoothed_labels, segments)`` tuple.
    """
    raw_labels = predict_baseline(
        features_df,
        idle_active_threshold=idle_active_threshold,
        idle_run_threshold=idle_run_threshold,
        scroll_high=scroll_high,
        keys_low=keys_low,
        keys_high=keys_high,
        shortcut_high=shortcut_high,
    )
    smoothed = rolling_majority(raw_labels, window=smooth_window)

    bucket_starts: list[datetime] = [
        pd.Timestamp(ts).to_pydatetime()
        for ts in features_df["bucket_start_ts"].values
    ]
    segments = segmentize(bucket_starts, smoothed, bucket_seconds=bucket_seconds)

    return smoothed, segments
