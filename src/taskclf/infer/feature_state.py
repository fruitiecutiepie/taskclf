"""Persistent rolling feature state for the online inference loop.

The online loop polls ActivityWatch in short windows, so
:func:`~taskclf.features.build.build_features_from_aw_events` only sees
a narrow slice of recent events.  Rolling features (15-minute switch
counts, rolling keyboard/mouse means, deltas, session length) are
therefore truncated to the poll window.

:class:`OnlineFeatureState` solves this by maintaining a circular buffer
of recent :class:`~taskclf.core.types.FeatureRow` values across poll
cycles.  After each row is built, it is pushed into the state, and
``get_context()`` returns corrected rolling aggregates that can be
overlaid onto the row before prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from taskclf.core.defaults import (
    DEFAULT_APP_SWITCH_WINDOW_15M,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_IDLE_GAP_SECONDS,
    DEFAULT_ROLLING_WINDOW_5,
    DEFAULT_ROLLING_WINDOW_15,
)
from taskclf.core.types import FeatureRow
from taskclf.features.dynamics import DynamicsTracker


@dataclass(eq=False)
class OnlineFeatureState:
    """Circular buffer of recent feature rows with rolling aggregate computation.

    Preserves enough history for all derived features that span beyond
    a single poll window: 5/15-minute rolling means, deltas, app switch
    counts, and session length.

    Args:
        buffer_minutes: How many minutes of history to retain.
        bucket_seconds: Width of each time bucket in seconds.
        idle_gap_seconds: Gap (seconds) between consecutive rows that
            triggers a session reset.
    """

    buffer_minutes: int = DEFAULT_APP_SWITCH_WINDOW_15M
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS
    idle_gap_seconds: float = DEFAULT_IDLE_GAP_SECONDS

    _capacity: int = field(init=False)
    _buffer: deque[FeatureRow] = field(init=False)
    _dynamics: DynamicsTracker = field(init=False)
    _session_start_ts: datetime | None = field(init=False, default=None)
    _last_dynamics: dict[str, float | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        buckets_per_minute = 60 / self.bucket_seconds
        self._capacity = max(int(self.buffer_minutes * buckets_per_minute), 1)
        self._buffer = deque(maxlen=self._capacity)
        self._dynamics = DynamicsTracker(
            rolling_5=DEFAULT_ROLLING_WINDOW_5,
            rolling_15=DEFAULT_ROLLING_WINDOW_15,
        )

    def push(self, row: FeatureRow) -> None:
        """Record a newly built feature row.

        Feeds the row's input metrics into the internal
        :class:`~taskclf.features.dynamics.DynamicsTracker` and detects
        idle-gap session boundaries.
        """
        if self._buffer:
            prev = self._buffer[-1]
            gap = (row.bucket_start_ts - prev.bucket_start_ts).total_seconds()
            if gap >= self.idle_gap_seconds:
                self._session_start_ts = row.bucket_start_ts

        if self._session_start_ts is None:
            self._session_start_ts = row.bucket_start_ts

        self._last_dynamics = self._dynamics.update(
            row.keys_per_min,
            row.clicks_per_min,
            row.mouse_distance,
        )
        self._buffer.append(row)

    def get_context(self) -> dict[str, int | float | None]:
        """Return rolling aggregates derived from the full buffer.

        The returned dict contains keys that directly correspond to
        :class:`~taskclf.core.types.FeatureRow` field names and can be
        used with ``row.model_copy(update=context)`` to overlay the
        corrected values.
        """
        if not self._buffer:
            return {}

        current = self._buffer[-1]

        switch_window = timedelta(minutes=DEFAULT_APP_SWITCH_WINDOW_15M)
        cutoff = current.bucket_start_ts - switch_window
        apps_in_window: set[str] = set()
        for row in self._buffer:
            if row.bucket_start_ts >= cutoff:
                apps_in_window.add(row.app_id)
        app_switch_count_15m = max(0, len(apps_in_window) - 1)

        session_length = 0.0
        if self._session_start_ts is not None:
            session_length = round(
                (current.bucket_start_ts - self._session_start_ts).total_seconds()
                / 60.0,
                2,
            )

        return {
            "app_switch_count_last_15m": app_switch_count_15m,
            "keys_per_min_rolling_5": self._last_dynamics.get("keys_per_min_rolling_5"),
            "keys_per_min_rolling_15": self._last_dynamics.get(
                "keys_per_min_rolling_15"
            ),
            "mouse_distance_rolling_5": self._last_dynamics.get(
                "mouse_distance_rolling_5"
            ),
            "mouse_distance_rolling_15": self._last_dynamics.get(
                "mouse_distance_rolling_15"
            ),
            "keys_per_min_delta": self._last_dynamics.get("keys_per_min_delta"),
            "clicks_per_min_delta": self._last_dynamics.get("clicks_per_min_delta"),
            "mouse_distance_delta": self._last_dynamics.get("mouse_distance_delta"),
            "session_length_so_far": session_length,
        }
