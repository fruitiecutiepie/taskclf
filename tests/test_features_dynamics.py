"""Tests for temporal dynamics features: rolling means and deltas."""

from __future__ import annotations

from collections import deque

from taskclf.features.dynamics import DynamicsTracker, delta_from_previous, rolling_mean


class TestRollingMean:
    def test_empty_history(self) -> None:
        assert rolling_mean(deque(), window=5) is None

    def test_all_none_values(self) -> None:
        assert rolling_mean(deque([None, None, None]), window=3) is None

    def test_single_value(self) -> None:
        assert rolling_mean(deque([10.0]), window=5) == 10.0

    def test_exact_window(self) -> None:
        result = rolling_mean(deque([10.0, 20.0, 30.0]), window=3)
        assert result == 20.0

    def test_window_larger_than_history(self) -> None:
        result = rolling_mean(deque([10.0, 20.0]), window=5)
        assert result == 15.0

    def test_window_smaller_than_history(self) -> None:
        result = rolling_mean(deque([10.0, 20.0, 30.0, 40.0, 50.0]), window=3)
        assert result == 40.0  # mean of [30, 40, 50]

    def test_skips_none_values(self) -> None:
        result = rolling_mean(deque([10.0, None, 30.0]), window=3)
        assert result == 20.0  # mean of [10, 30]

    def test_result_rounded(self) -> None:
        result = rolling_mean(deque([1.0, 2.0, 3.0]), window=3)
        assert result == 2.0
        assert isinstance(result, float)


class TestDeltaFromPrevious:
    def test_both_none(self) -> None:
        assert delta_from_previous(None, None) is None

    def test_current_none(self) -> None:
        assert delta_from_previous(None, 5.0) is None

    def test_previous_none(self) -> None:
        assert delta_from_previous(5.0, None) is None

    def test_positive_delta(self) -> None:
        assert delta_from_previous(10.0, 5.0) == 5.0

    def test_negative_delta(self) -> None:
        assert delta_from_previous(3.0, 8.0) == -5.0

    def test_zero_delta(self) -> None:
        assert delta_from_previous(7.0, 7.0) == 0.0


class TestDynamicsTracker:
    def test_first_bucket_has_no_delta(self) -> None:
        tracker = DynamicsTracker()
        result = tracker.update(60.0, 5.0, 300.0)
        assert result["keys_per_min_delta"] is None
        assert result["clicks_per_min_delta"] is None
        assert result["mouse_distance_delta"] is None

    def test_first_bucket_rolling_equals_current(self) -> None:
        tracker = DynamicsTracker()
        result = tracker.update(60.0, 5.0, 300.0)
        assert result["keys_per_min_rolling_5"] == 60.0
        assert result["keys_per_min_rolling_15"] == 60.0
        assert result["mouse_distance_rolling_5"] == 300.0

    def test_second_bucket_computes_delta(self) -> None:
        tracker = DynamicsTracker()
        tracker.update(60.0, 5.0, 300.0)
        result = tracker.update(80.0, 3.0, 250.0)
        assert result["keys_per_min_delta"] == 20.0
        assert result["clicks_per_min_delta"] == -2.0
        assert result["mouse_distance_delta"] == -50.0

    def test_rolling_mean_accumulates(self) -> None:
        tracker = DynamicsTracker(rolling_5=3)
        tracker.update(10.0, 1.0, 100.0)
        tracker.update(20.0, 2.0, 200.0)
        result = tracker.update(30.0, 3.0, 300.0)
        assert result["keys_per_min_rolling_5"] == 20.0  # mean(10,20,30)
        assert result["mouse_distance_rolling_5"] == 200.0

    def test_none_input_handled(self) -> None:
        tracker = DynamicsTracker()
        result = tracker.update(None, None, None)
        assert result["keys_per_min_rolling_5"] is None
        assert result["keys_per_min_delta"] is None
        assert result["mouse_distance_rolling_5"] is None

    def test_none_then_value_has_no_delta(self) -> None:
        tracker = DynamicsTracker()
        tracker.update(None, None, None)
        result = tracker.update(50.0, 3.0, 200.0)
        assert result["keys_per_min_delta"] is None  # prev was None
        assert result["keys_per_min_rolling_5"] == 50.0

    def test_compute_batch(self) -> None:
        keys = [10.0, 20.0, 30.0, 40.0, 50.0]
        clicks = [1.0, 2.0, 3.0, 4.0, 5.0]
        mouse = [100.0, 200.0, 300.0, 400.0, 500.0]

        results = DynamicsTracker.compute_batch(keys, clicks, mouse, rolling_5=3)
        assert len(results) == 5

        assert results[0]["keys_per_min_delta"] is None
        assert results[1]["keys_per_min_delta"] == 10.0
        assert results[4]["keys_per_min_delta"] == 10.0

        assert results[2]["keys_per_min_rolling_5"] == 20.0  # mean(10,20,30)

    def test_compute_batch_with_nones(self) -> None:
        keys = [None, 20.0, None, 40.0]
        clicks = [None, 2.0, None, 4.0]
        mouse = [None, 200.0, None, 400.0]

        results = DynamicsTracker.compute_batch(keys, clicks, mouse)
        assert results[0]["keys_per_min_rolling_5"] is None
        assert results[1]["keys_per_min_rolling_5"] == 20.0
        assert results[3]["keys_per_min_delta"] is None  # prev was None
