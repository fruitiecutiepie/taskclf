"""Point-in-time quality telemetry: collection and persistence.

Stores only aggregate statistics -- never raw content.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.defaults import DEFAULT_TELEMETRY_DIR, MIXED_UNKNOWN

_EPS = 1e-8

NUMERICAL_FEATURES: list[str] = [
    "app_switch_count_last_5m",
    "app_foreground_time_ratio",
    "app_change_count",
    "hour_of_day",
    "session_length_so_far",
    "keys_per_min",
    "backspace_ratio",
    "shortcut_rate",
    "clicks_per_min",
    "scroll_events_per_min",
    "mouse_distance",
    "active_seconds_keyboard",
    "active_seconds_mouse",
    "active_seconds_any",
    "max_idle_run_seconds",
    "event_density",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConfidenceStats(BaseModel):
    """Aggregate statistics on prediction confidence."""

    mean: float
    median: float
    p5: float
    p95: float
    std: float


class TelemetrySnapshot(BaseModel):
    """Point-in-time quality snapshot of a prediction window."""

    timestamp: datetime
    user_id: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    total_windows: int
    feature_missingness: dict[str, float] = Field(default_factory=dict)
    confidence_stats: ConfidenceStats | None = None
    reject_rate: float = 0.0
    mean_entropy: float = 0.0
    class_distribution: dict[str, float] = Field(default_factory=dict)
    schema_version: str = "features_v1"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_telemetry(
    features_df: pd.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    confidences: np.ndarray | Sequence[float] | None = None,
    core_probs: np.ndarray | None = None,
    user_id: str | None = None,
    reject_label: str = MIXED_UNKNOWN,
) -> TelemetrySnapshot:
    """Compute a telemetry snapshot from features and prediction outputs.

    Args:
        features_df: Feature DataFrame (one row per bucket).
        labels: Predicted labels (post-reject).
        confidences: ``max(proba)`` per row.
        core_probs: Full probability matrix ``(n, k)`` for entropy.
        user_id: Optional user to scope the snapshot.
        reject_label: Label used for rejected predictions.

    Returns:
        A populated :class:`TelemetrySnapshot`.
    """
    n = len(features_df)
    now = datetime.now(tz=timezone.utc)

    missingness: dict[str, float] = {}
    for col in NUMERICAL_FEATURES:
        if col in features_df.columns:
            null_frac = float(features_df[col].isna().mean())
            missingness[col] = round(null_frac, 4)

    window_start: datetime | None = None
    window_end: datetime | None = None
    if "bucket_start_ts" in features_df.columns and n > 0:
        window_start = pd.Timestamp(features_df["bucket_start_ts"].min()).to_pydatetime()
        window_end = pd.Timestamp(features_df["bucket_start_ts"].max()).to_pydatetime()

    conf_stats: ConfidenceStats | None = None
    if confidences is not None:
        c = np.asarray(confidences, dtype=np.float64)
        if len(c) > 0:
            conf_stats = ConfidenceStats(
                mean=round(float(np.mean(c)), 4),
                median=round(float(np.median(c)), 4),
                p5=round(float(np.percentile(c, 5)), 4),
                p95=round(float(np.percentile(c, 95)), 4),
                std=round(float(np.std(c)), 4),
            )

    rr = 0.0
    class_dist: dict[str, float] = {}
    if labels is not None and len(labels) > 0:
        rr = sum(1 for l in labels if l == reject_label) / len(labels)
        from collections import Counter

        counts = Counter(labels)
        total = sum(counts.values())
        class_dist = {k: round(v / total, 4) for k, v in sorted(counts.items())}

    mean_ent = 0.0
    if core_probs is not None:
        p = np.asarray(core_probs, dtype=np.float64)
        if len(p) > 0:
            p_clipped = np.clip(p, _EPS, 1.0)
            entropies = -np.sum(p_clipped * np.log(p_clipped), axis=1)
            mean_ent = float(np.mean(entropies))

    return TelemetrySnapshot(
        timestamp=now,
        user_id=user_id,
        window_start=window_start,
        window_end=window_end,
        total_windows=n,
        feature_missingness=missingness,
        confidence_stats=conf_stats,
        reject_rate=round(rr, 4),
        mean_entropy=round(mean_ent, 6),
        class_distribution=class_dist,
    )


# ---------------------------------------------------------------------------
# Persistence (append-only JSONL)
# ---------------------------------------------------------------------------


class TelemetryStore:
    """Append-only JSONL store for telemetry snapshots."""

    def __init__(self, store_dir: str | Path = DEFAULT_TELEMETRY_DIR) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, user_id: str | None) -> Path:
        name = f"telemetry_{user_id}.jsonl" if user_id else "telemetry_global.jsonl"
        return self._dir / name

    def append(self, snapshot: TelemetrySnapshot) -> Path:
        """Append a snapshot to the appropriate JSONL file.

        Returns:
            Path of the file written to.
        """
        path = self._path_for(snapshot.user_id)
        line = snapshot.model_dump_json() + "\n"

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mktemp(dir=self._dir, suffix=".tmp"))
        try:
            existing = path.read_text() if path.exists() else ""
            tmp.write_text(existing + line)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def read_recent(
        self,
        n: int = 10,
        *,
        user_id: str | None = None,
    ) -> list[TelemetrySnapshot]:
        """Read the last *n* snapshots.

        Args:
            n: Maximum number of snapshots to return.
            user_id: Scope to a specific user (``None`` = global).

        Returns:
            List of :class:`TelemetrySnapshot` (newest last).
        """
        path = self._path_for(user_id)
        if not path.exists():
            return []
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        return [TelemetrySnapshot.model_validate_json(l) for l in lines[-n:]]

    def read_range(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str | None = None,
    ) -> list[TelemetrySnapshot]:
        """Read snapshots whose timestamp falls within [*start*, *end*].

        Args:
            start: Inclusive lower bound.
            end: Inclusive upper bound.
            user_id: Scope to a specific user (``None`` = global).

        Returns:
            Matching snapshots ordered by timestamp.
        """
        all_snaps = self.read_recent(n=10_000, user_id=user_id)
        return [
            s for s in all_snaps
            if start <= s.timestamp <= end
        ]
