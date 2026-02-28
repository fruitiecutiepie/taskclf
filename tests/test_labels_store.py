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
    _same_user,
    append_label_span,
    extend_and_append_label_span,
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


def _span(
    start: tuple[int, int],
    end: tuple[int, int],
    label: str = "Build",
    user_id: str | None = "u1",
) -> LabelSpan:
    """Shorthand: times are (hour, minute) on a fixed date."""
    base = dt.date(2025, 6, 15)
    return LabelSpan(
        start_ts=dt.datetime(base.year, base.month, base.day, *start),
        end_ts=dt.datetime(base.year, base.month, base.day, *end),
        label=label,
        provenance="manual",
        user_id=user_id,
    )


class TestSameUser:
    def test_both_none(self) -> None:
        a = _span((9, 0), (9, 5), user_id=None)
        b = _span((10, 0), (10, 5), user_id=None)
        assert _same_user(a, b) is True

    def test_both_equal(self) -> None:
        a = _span((9, 0), (9, 5), user_id="u1")
        b = _span((10, 0), (10, 5), user_id="u1")
        assert _same_user(a, b) is True

    def test_both_set_different(self) -> None:
        a = _span((9, 0), (9, 5), user_id="u1")
        b = _span((10, 0), (10, 5), user_id="u2")
        assert _same_user(a, b) is False

    def test_one_none_one_set(self) -> None:
        a = _span((9, 0), (9, 5), user_id=None)
        b = _span((10, 0), (10, 5), user_id="u1")
        assert _same_user(a, b) is False
        assert _same_user(b, a) is False


class TestExtendAndAppendLabelSpan:
    """Tests for extend_and_append_label_span."""

    # -- happy path / gap-filling -----------------------------------------------

    def test_first_label_no_file(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        span = _span((10, 0), (10, 5))
        extend_and_append_label_span(span, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].start_ts == span.start_ts
        assert loaded[0].end_ts == span.end_ts

    def test_first_label_for_user_other_users_exist(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        other = _span((9, 0), (9, 5), user_id="u2")
        write_label_spans([other], path)

        span = _span((10, 0), (10, 5), user_id="u1")
        extend_and_append_label_span(span, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 2
        u2 = [s for s in loaded if s.user_id == "u2"][0]
        assert u2.end_ts == other.end_ts

    def test_normal_gap_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 0))
        write_label_spans([prev], path)

        new = _span((10, 29), (10, 30), label="ReadResearch")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        extended = [s for s in loaded if s.label == "Build"][0]
        assert extended.start_ts == prev.start_ts
        assert extended.end_ts == new.start_ts

        appended = [s for s in loaded if s.label == "ReadResearch"][0]
        assert appended.start_ts == new.start_ts
        assert appended.end_ts == new.end_ts

    def test_chained_extension_three_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        a = _span((9, 0), (9, 5), label="Build")
        extend_and_append_label_span(a, path)

        b = _span((9, 30), (9, 35), label="Debug")
        extend_and_append_label_span(b, path)

        loaded = read_label_spans(path)
        a_loaded = [s for s in loaded if s.label == "Build"][0]
        assert a_loaded.end_ts == b.start_ts

        c = _span((10, 0), (10, 5), label="Write")
        extend_and_append_label_span(c, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 3
        b_loaded = [s for s in loaded if s.label == "Debug"][0]
        assert b_loaded.end_ts == c.start_ts
        a_loaded = [s for s in loaded if s.label == "Build"][0]
        assert a_loaded.end_ts == b.start_ts

    # -- truncation / overlap handling ------------------------------------------

    def test_overlap_truncates_previous(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 5))
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 10), label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        truncated = [s for s in loaded if s.label == "Build"][0]
        assert truncated.end_ts == new.start_ts

    def test_adjacent_spans_no_change(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 0))
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        b = [s for s in loaded if s.label == "Build"][0]
        assert b.end_ts == new.start_ts

    # -- guard conditions -------------------------------------------------------

    def test_same_start_as_previous_raises_overlap(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((10, 0), (10, 5))
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 10), label="Debug")
        with pytest.raises(ValueError, match="overlaps"):
            extend_and_append_label_span(new, path)

    def test_new_before_previous_no_overlap_succeeds(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((11, 0), (11, 5))
        write_label_spans([prev], path)

        new = _span((9, 0), (9, 5), label="Debug")
        extend_and_append_label_span(new, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 2

    def test_new_inside_previous_truncates(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 10))
        write_label_spans([prev], path)

        new = _span((9, 5), (9, 8), label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts

    # -- multi-user isolation ---------------------------------------------------

    def test_different_user_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        u1_span = _span((9, 0), (9, 5), user_id="u1")
        write_label_spans([u1_span], path)

        u2_span = _span((10, 0), (10, 5), user_id="u2", label="Debug")
        extend_and_append_label_span(u2_span, path)

        loaded = read_label_spans(path)
        u1 = [s for s in loaded if s.user_id == "u1"][0]
        assert u1.end_ts == u1_span.end_ts

    def test_same_user_extended_other_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        u1_prev = _span((9, 0), (9, 5), user_id="u1")
        u2_span = _span((9, 0), (9, 5), user_id="u2")
        write_label_spans([u1_prev, u2_span], path)

        u1_new = _span((10, 0), (10, 5), user_id="u1", label="Debug")
        extend_and_append_label_span(u1_new, path)

        loaded = read_label_spans(path)
        u1_old = [s for s in loaded if s.user_id == "u1" and s.label == "Build"][0]
        assert u1_old.end_ts == u1_new.start_ts
        u2 = [s for s in loaded if s.user_id == "u2"][0]
        assert u2.end_ts == u2_span.end_ts

    # -- user_id None handling --------------------------------------------------

    def test_both_none_user_extends(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5), user_id=None)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), user_id=None, label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts

    def test_different_user_no_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5), user_id="other")
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), user_id="u1", label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == prev.end_ts

    # -- picks correct most-recent label ----------------------------------------

    def test_picks_latest_by_start_ts(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        early = _span((8, 0), (8, 5), label="Debug")
        middle = _span((9, 0), (9, 5), label="Review")
        late = _span((9, 55), (10, 0), label="Write")
        write_label_spans([early, middle, late], path)

        new = _span((10, 29), (10, 30), label="Meet")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        by_label = {s.label: s for s in loaded}
        assert by_label["Debug"].end_ts == early.end_ts
        assert by_label["Review"].end_ts == middle.end_ts
        assert by_label["Write"].end_ts == new.start_ts

    # -- post-extension overlap validation --------------------------------------

    def test_extension_truncate_avoids_third_span_overlap(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        first = _span((9, 0), (9, 30), label="Build")
        third = _span((9, 55), (10, 0), label="Review")
        write_label_spans([first, third], path)

        new = _span((9, 20), (9, 25), label="Debug")
        extend_and_append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 3
        b = [s for s in loaded if s.label == "Build"][0]
        assert b.end_ts == new.start_ts

    # -- persistence round-trip -------------------------------------------------

    def test_extension_persisted_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5))
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), label="Debug")
        extend_and_append_label_span(new, path)

        reloaded = read_label_spans(path)
        old = [s for s in reloaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts
