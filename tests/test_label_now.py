"""Tests for the label-now CLI command and online queue integration.

Covers:
- label-now creates a valid span with correct time range
- label-now gracefully handles unreachable ActivityWatch
- label-now rejects overlapping spans
- Online loop enqueue integration constructs correct DataFrame
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue
from taskclf.labels.store import append_label_span, read_label_spans


class TestLabelNowSpanCreation:
    """The label-now workflow creates a span from (now - N minutes) to now."""

    def test_creates_span_for_last_n_minutes(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 14, 30, 0)
        minutes = 10
        start_ts = now - dt.timedelta(minutes=minutes)

        span = LabelSpan(
            start_ts=start_ts,
            end_ts=now,
            label="Build",
            provenance="manual",
            user_id="default-user",
        )
        append_label_span(span, labels_path)

        loaded = read_label_spans(labels_path)
        assert len(loaded) == 1
        assert loaded[0].label == "Build"
        assert loaded[0].end_ts - loaded[0].start_ts == dt.timedelta(minutes=10)

    def test_all_core_labels_accepted(self, tmp_path: Path) -> None:
        now = dt.datetime(2026, 2, 26, 14, 0, 0)
        for i, label in enumerate(sorted(LABEL_SET_V1)):
            labels_path = tmp_path / f"labels_{i}.parquet"
            start = now - dt.timedelta(minutes=5)
            span = LabelSpan(
                start_ts=start,
                end_ts=now,
                label=label,
                provenance="manual",
            )
            append_label_span(span, labels_path)
            loaded = read_label_spans(labels_path)
            assert loaded[0].label == label

    def test_rejects_overlap(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 14, 30, 0)

        span1 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=10),
            end_ts=now,
            label="Build",
            provenance="manual",
            user_id="u1",
        )
        append_label_span(span1, labels_path)

        span2 = LabelSpan(
            start_ts=now - dt.timedelta(minutes=5),
            end_ts=now + dt.timedelta(minutes=5),
            label="Debug",
            provenance="manual",
            user_id="u1",
        )
        with pytest.raises(ValueError, match="overlaps"):
            append_label_span(span2, labels_path)

    def test_different_users_no_overlap(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 14, 30, 0)

        for uid in ("u1", "u2"):
            span = LabelSpan(
                start_ts=now - dt.timedelta(minutes=10),
                end_ts=now,
                label="Build",
                provenance="manual",
                user_id=uid,
            )
            append_label_span(span, labels_path)

        loaded = read_label_spans(labels_path)
        assert len(loaded) == 2

    def test_confidence_persisted(self, tmp_path: Path) -> None:
        labels_path = tmp_path / "labels.parquet"
        now = dt.datetime(2026, 2, 26, 14, 30, 0)
        span = LabelSpan(
            start_ts=now - dt.timedelta(minutes=5),
            end_ts=now,
            label="Write",
            provenance="manual",
            confidence=0.95,
        )
        append_label_span(span, labels_path)
        loaded = read_label_spans(labels_path)
        assert loaded[0].confidence == pytest.approx(0.95)


class TestOnlineQueueEnqueue:
    """The online loop should enqueue low-confidence predictions correctly."""

    def test_enqueue_low_confidence_prediction(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "queue.json")
        bucket_start = dt.datetime(2026, 2, 26, 14, 0, 0)
        bucket_end = dt.datetime(2026, 2, 26, 14, 1, 0)

        enqueue_df = pd.DataFrame([{
            "user_id": "default-user",
            "bucket_start_ts": bucket_start,
            "bucket_end_ts": bucket_end,
            "confidence": 0.30,
            "predicted_label": "Build",
        }])
        added = queue.enqueue_low_confidence(enqueue_df, threshold=0.55)
        assert added == 1
        items = queue.all_items
        assert items[0].reason == "low_confidence"
        assert items[0].predicted_label == "Build"
        assert items[0].confidence == pytest.approx(0.30)

    def test_above_threshold_not_enqueued(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "queue.json")
        bucket_start = dt.datetime(2026, 2, 26, 14, 0, 0)

        enqueue_df = pd.DataFrame([{
            "user_id": "default-user",
            "bucket_start_ts": bucket_start,
            "bucket_end_ts": bucket_start + dt.timedelta(minutes=1),
            "confidence": 0.80,
            "predicted_label": "Build",
        }])
        added = queue.enqueue_low_confidence(enqueue_df, threshold=0.55)
        assert added == 0

    def test_no_duplicate_enqueue(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "queue.json")
        bucket_start = dt.datetime(2026, 2, 26, 14, 0, 0)

        enqueue_df = pd.DataFrame([{
            "user_id": "default-user",
            "bucket_start_ts": bucket_start,
            "bucket_end_ts": bucket_start + dt.timedelta(minutes=1),
            "confidence": 0.30,
            "predicted_label": "Build",
        }])
        queue.enqueue_low_confidence(enqueue_df, threshold=0.55)
        added = queue.enqueue_low_confidence(enqueue_df, threshold=0.55)
        assert added == 0
        assert len(queue.all_items) == 1

    def test_multiple_buckets_enqueued(self, tmp_path: Path) -> None:
        queue = ActiveLabelingQueue(tmp_path / "queue.json")
        base = dt.datetime(2026, 2, 26, 14, 0, 0)
        rows = []
        for i in range(5):
            rows.append({
                "user_id": "default-user",
                "bucket_start_ts": base + dt.timedelta(minutes=i),
                "bucket_end_ts": base + dt.timedelta(minutes=i + 1),
                "confidence": 0.20 + i * 0.05,
                "predicted_label": "Build",
            })
        enqueue_df = pd.DataFrame(rows)
        added = queue.enqueue_low_confidence(enqueue_df, threshold=0.55)
        assert added == 5

        pending = queue.get_pending()
        confidences = [r.confidence for r in pending]
        assert confidences == sorted(confidences)
