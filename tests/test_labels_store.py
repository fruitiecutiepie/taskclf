"""Tests for label span I/O and synthetic label generation.

Covers: write/read round-trip, generate_dummy_labels validity and count.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from taskclf.core.types import LABEL_SET_V1
from taskclf.labels.store import (
    generate_dummy_labels,
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
