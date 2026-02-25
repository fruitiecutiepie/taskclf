"""Temporal dynamics: rolling means and inter-bucket deltas.

These features capture how interaction patterns *change* over time,
which is a strong signal for distinguishing tasks that look similar
in a single bucket but differ in trajectory (e.g. sustained coding
vs. switching between reading and chatting).

All functions are pure — they take a history buffer and return a
single value (or ``None`` when insufficient history is available).
"""

from __future__ import annotations

from collections import deque


def rolling_mean(
    history: deque[float | None],
    window: int,
) -> float | None:
    """Compute the mean of the last *window* non-None values in *history*.

    Returns ``None`` when there are zero non-None values in the window.

    Args:
        history: Ordered buffer of recent metric values (newest last).
        window: Number of recent entries to consider.

    Returns:
        Mean of available values, or ``None``.
    """
    values = [v for v in list(history)[-window:] if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def delta_from_previous(
    current: float | None,
    previous: float | None,
) -> float | None:
    """Compute the change from *previous* to *current*.

    Returns ``None`` when either value is ``None``.

    Args:
        current: Current bucket's metric value.
        previous: Previous bucket's metric value.

    Returns:
        ``current - previous``, or ``None``.
    """
    if current is None or previous is None:
        return None
    return round(current - previous, 4)


class DynamicsTracker:
    """Stateful tracker that accumulates per-bucket metrics and emits dynamics features.

    Maintains rolling buffers for ``keys_per_min``, ``clicks_per_min``,
    and ``mouse_distance`` so that each call to :meth:`update` returns
    the computed rolling means and deltas for the latest bucket.

    For batch mode, call :meth:`compute_batch` instead.
    """

    def __init__(self, rolling_5: int = 5, rolling_15: int = 15) -> None:
        self._rolling_5 = rolling_5
        self._rolling_15 = rolling_15
        max_len = max(rolling_5, rolling_15)
        self._keys_buf: deque[float | None] = deque(maxlen=max_len)
        self._clicks_buf: deque[float | None] = deque(maxlen=max_len)
        self._mouse_buf: deque[float | None] = deque(maxlen=max_len)
        self._prev_keys: float | None = None
        self._prev_clicks: float | None = None
        self._prev_mouse: float | None = None

    def update(
        self,
        keys_per_min: float | None,
        clicks_per_min: float | None,
        mouse_distance: float | None,
    ) -> dict[str, float | None]:
        """Record one bucket's metrics and return dynamics features.

        Args:
            keys_per_min: Current bucket's keys_per_min (may be None).
            clicks_per_min: Current bucket's clicks_per_min (may be None).
            mouse_distance: Current bucket's mouse_distance (may be None).

        Returns:
            Dict with keys matching the FeatureRow field names.
        """
        self._keys_buf.append(keys_per_min)
        self._clicks_buf.append(clicks_per_min)
        self._mouse_buf.append(mouse_distance)

        result = {
            "keys_per_min_rolling_5": rolling_mean(self._keys_buf, self._rolling_5),
            "keys_per_min_rolling_15": rolling_mean(self._keys_buf, self._rolling_15),
            "mouse_distance_rolling_5": rolling_mean(self._mouse_buf, self._rolling_5),
            "mouse_distance_rolling_15": rolling_mean(self._mouse_buf, self._rolling_15),
            "keys_per_min_delta": delta_from_previous(keys_per_min, self._prev_keys),
            "clicks_per_min_delta": delta_from_previous(clicks_per_min, self._prev_clicks),
            "mouse_distance_delta": delta_from_previous(mouse_distance, self._prev_mouse),
        }

        self._prev_keys = keys_per_min
        self._prev_clicks = clicks_per_min
        self._prev_mouse = mouse_distance

        return result

    @staticmethod
    def compute_batch(
        keys_series: list[float | None],
        clicks_series: list[float | None],
        mouse_series: list[float | None],
        *,
        rolling_5: int = 5,
        rolling_15: int = 15,
    ) -> list[dict[str, float | None]]:
        """Compute dynamics features for an entire ordered sequence of buckets.

        Args:
            keys_series: keys_per_min values, one per bucket.
            clicks_series: clicks_per_min values, one per bucket.
            mouse_series: mouse_distance values, one per bucket.
            rolling_5: Short rolling window size.
            rolling_15: Long rolling window size.

        Returns:
            List of dicts (one per bucket) with dynamics feature values.
        """
        tracker = DynamicsTracker(rolling_5=rolling_5, rolling_15=rolling_15)
        return [
            tracker.update(k, c, m)
            for k, c, m in zip(keys_series, clicks_series, mouse_series)
        ]
