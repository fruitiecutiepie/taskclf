"""Tests for labels.queue: ActiveLabelingQueue lifecycle and persistence.

Covers: enqueue, dedup, daily limit, mark_done, JSON round-trip.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from taskclf.labels.queue import ActiveLabelingQueue


def _predictions_df(
    n: int = 5,
    confidence: float = 0.3,
    user_id: str = "u1",
) -> pd.DataFrame:
    base = dt.datetime(2025, 6, 15, 10, 0)
    rows = []
    for i in range(n):
        rows.append({
            "user_id": user_id,
            "bucket_start_ts": base + dt.timedelta(minutes=i),
            "bucket_end_ts": base + dt.timedelta(minutes=i + 1),
            "confidence": confidence,
            "predicted_label": "Build",
        })
    return pd.DataFrame(rows)


class TestEnqueueLowConfidence:
    def test_adds_below_threshold(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        df = _predictions_df(n=3, confidence=0.3)
        added = queue.enqueue_low_confidence(df, threshold=0.5)
        assert added == 3
        assert len(queue.all_items) == 3

    def test_skips_above_threshold(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        df = _predictions_df(n=3, confidence=0.8)
        added = queue.enqueue_low_confidence(df, threshold=0.5)
        assert added == 0

    def test_no_duplicates(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        df = _predictions_df(n=2, confidence=0.3)
        queue.enqueue_low_confidence(df, threshold=0.5)
        added = queue.enqueue_low_confidence(df, threshold=0.5)
        assert added == 0
        assert len(queue.all_items) == 2


class TestEnqueueDrift:
    def test_adds_drift_buckets(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        buckets = [
            {
                "user_id": "u1",
                "bucket_start_ts": dt.datetime(2025, 6, 15, 10, 0),
                "bucket_end_ts": dt.datetime(2025, 6, 15, 10, 1),
                "predicted_label": "Build",
                "confidence": 0.4,
            }
        ]
        added = queue.enqueue_drift(buckets)
        assert added == 1
        assert queue.all_items[0].reason == "drift"


class TestGetPending:
    def test_respects_limit(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        queue.enqueue_low_confidence(
            _predictions_df(n=10, confidence=0.2), threshold=0.5
        )
        pending = queue.get_pending(limit=3)
        assert len(pending) <= 3

    def test_sorted_by_confidence(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        base = dt.datetime(2025, 6, 15, 10, 0)
        rows = [
            {
                "user_id": "u1",
                "bucket_start_ts": base,
                "bucket_end_ts": base + dt.timedelta(minutes=1),
                "confidence": 0.4,
                "predicted_label": "Build",
            },
            {
                "user_id": "u1",
                "bucket_start_ts": base + dt.timedelta(minutes=1),
                "bucket_end_ts": base + dt.timedelta(minutes=2),
                "confidence": 0.1,
                "predicted_label": "Debug",
            },
        ]
        queue.enqueue_low_confidence(pd.DataFrame(rows), threshold=0.5)
        pending = queue.get_pending()
        assert pending[0].confidence <= pending[1].confidence  # type: ignore[operator]

    def test_filters_by_user(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        queue.enqueue_low_confidence(
            _predictions_df(n=3, confidence=0.2, user_id="u1"), threshold=0.5
        )
        queue.enqueue_low_confidence(
            _predictions_df(n=2, confidence=0.2, user_id="u2"), threshold=0.5
        )
        pending = queue.get_pending(user_id="u1")
        assert all(r.user_id == "u1" for r in pending)


class TestDailyLimit:
    def test_respects_max_asks(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json", max_asks_per_day=2)
        queue.enqueue_low_confidence(
            _predictions_df(n=5, confidence=0.1), threshold=0.5
        )
        items = queue.get_pending()
        assert len(items) <= 2


class TestMarkDone:
    def test_labeled_transition(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        queue.enqueue_low_confidence(
            _predictions_df(n=1, confidence=0.2), threshold=0.5
        )
        req = queue.all_items[0]
        updated = queue.mark_done(req.request_id, status="labeled")
        assert updated is not None
        assert updated.status == "labeled"

    def test_skipped_transition(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        queue.enqueue_low_confidence(
            _predictions_df(n=1, confidence=0.2), threshold=0.5
        )
        req = queue.all_items[0]
        updated = queue.mark_done(req.request_id, status="skipped")
        assert updated is not None
        assert updated.status == "skipped"

    def test_unknown_id_returns_none(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "q.json")
        result = queue.mark_done("nonexistent-id")
        assert result is None


class TestPersistence:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        queue1 = ActiveLabelingQueue(path)
        queue1.enqueue_low_confidence(
            _predictions_df(n=3, confidence=0.2), threshold=0.5
        )

        queue2 = ActiveLabelingQueue(path)
        assert len(queue2.all_items) == 3
        for a, b in zip(queue1.all_items, queue2.all_items):
            assert a.request_id == b.request_id
            assert a.user_id == b.user_id
            assert a.status == b.status
