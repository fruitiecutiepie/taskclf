"""Tests for label span I/O, synthetic generation, and new store helpers.

Covers: write/read round-trip, generate_dummy_labels, append_label_span,
generate_label_summary, and CSV import with optional columns.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.labels.store import (
    append_label_span,
    generate_dummy_labels,
    generate_label_summary,
    import_labels_from_csv,
    read_label_spans,
    write_label_spans,
)


class TestLabelSpanRoundTrip:
    def test_write_read_preserves_spans(self, tmp_path: Path) -> None:
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=5)
        path = tmp_path / "labels.parquet"
        write_label_spans(spans, path)
        loaded = read_label_spans(path)

        assert len(loaded) == len(spans)
        for original, restored in zip(spans, loaded):
            assert original.start_ts == restored.start_ts
            assert original.end_ts == restored.end_ts
            assert original.label == restored.label
            assert original.provenance == restored.provenance

    def test_write_read_preserves_user_id_and_confidence(self, tmp_path: Path) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            user_id="test-user",
            confidence=0.9,
        )
        path = tmp_path / "labels.parquet"
        write_label_spans([span], path)
        loaded = read_label_spans(path)

        assert len(loaded) == 1
        assert loaded[0].user_id == "test-user"
        assert loaded[0].confidence == pytest.approx(0.9)


class TestGenerateDummyLabels:
    def test_produces_requested_count(self) -> None:
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=7)
        assert len(spans) == 7

    def test_all_labels_in_label_set(self) -> None:
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=20)
        for span in spans:
            assert span.label in LABEL_SET_V1

    def test_spans_have_positive_duration(self) -> None:
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=10)
        for span in spans:
            assert span.end_ts > span.start_ts

    def test_provenance_is_synthetic(self) -> None:
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=3)
        for span in spans:
            assert span.provenance == "synthetic"


class TestAppendLabelSpan:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            user_id="u1",
        )
        append_label_span(span, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 1

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        s1 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            user_id="u1",
        )
        s2 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 11, 0),
            end_ts=dt.datetime(2025, 6, 15, 11, 5),
            label="Debug",
            provenance="manual",
            user_id="u1",
        )
        append_label_span(s1, path)
        append_label_span(s2, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 2

    def test_rejects_overlapping_same_user(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        s1 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 10),
            label="Build",
            provenance="manual",
            user_id="u1",
        )
        s2 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 5),
            end_ts=dt.datetime(2025, 6, 15, 10, 15),
            label="Debug",
            provenance="manual",
            user_id="u1",
        )
        append_label_span(s1, path)
        with pytest.raises(ValueError, match="overlaps"):
            append_label_span(s2, path)

    def test_allows_different_user_same_time(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        s1 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 10),
            label="Build",
            provenance="manual",
            user_id="u1",
        )
        s2 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 10),
            label="Debug",
            provenance="manual",
            user_id="u2",
        )
        append_label_span(s1, path)
        append_label_span(s2, path)
        assert len(read_label_spans(path)) == 2


class TestGenerateLabelSummary:
    def test_empty_window(self) -> None:
        df = pd.DataFrame({
            "bucket_start_ts": pd.to_datetime(["2025-06-15T10:00:00"]),
            "app_id": ["com.apple.Terminal"],
            "session_id": ["s1"],
        })
        summary = generate_label_summary(
            df,
            dt.datetime(2025, 6, 15, 12, 0),
            dt.datetime(2025, 6, 15, 13, 0),
        )
        assert summary["total_buckets"] == 0

    def test_summary_fields(self) -> None:
        base = dt.datetime(2025, 6, 15, 10, 0)
        df = pd.DataFrame({
            "bucket_start_ts": [base + dt.timedelta(minutes=i) for i in range(5)],
            "app_id": ["com.apple.Terminal"] * 3 + ["org.mozilla.firefox"] * 2,
            "session_id": ["s1"] * 5,
            "keys_per_min": [50.0, 60.0, 70.0, 10.0, 5.0],
            "clicks_per_min": [3.0, 4.0, 5.0, 6.0, 7.0],
            "scroll_events_per_min": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        summary = generate_label_summary(df, base, base + dt.timedelta(minutes=5))
        assert summary["total_buckets"] == 5
        assert summary["session_count"] == 1
        assert len(summary["top_apps"]) == 2
        assert summary["mean_keys_per_min"] is not None


class TestImportLabelsFromCsvWithOptionalColumns:
    def test_csv_with_user_id_and_confidence(self, tmp_path: Path) -> None:
        csv = tmp_path / "labels.csv"
        csv.write_text(
            "start_ts,end_ts,label,provenance,user_id,confidence\n"
            "2025-06-15T10:00:00,2025-06-15T10:05:00,Build,manual,u1,0.9\n"
            "2025-06-15T11:00:00,2025-06-15T11:05:00,Debug,manual,u2,\n"
        )
        spans = import_labels_from_csv(csv)
        assert len(spans) == 2
        assert spans[0].user_id == "u1"
        assert spans[0].confidence == pytest.approx(0.9)
        assert spans[1].user_id == "u2"
        assert spans[1].confidence is None

    def test_csv_without_optional_columns(self, tmp_path: Path) -> None:
        csv = tmp_path / "labels.csv"
        csv.write_text(
            "start_ts,end_ts,label,provenance\n"
            "2025-06-15T10:00:00,2025-06-15T10:05:00,Build,manual\n"
        )
        spans = import_labels_from_csv(csv)
        assert len(spans) == 1
        assert spans[0].user_id is None
        assert spans[0].confidence is None
