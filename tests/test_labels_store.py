"""Tests for label span I/O, synthetic generation, and new store helpers.

Covers: write/read round-trip, generate_dummy_labels, append_label_span,
generate_label_summary, CSV import with optional columns,
TC-LABEL-UPD-001..005 (update_label_span),
TC-LABEL-DEL-001..005 (delete_label_span).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from taskclf.core.types import LABEL_SET_V1, LabelSpan
from taskclf.labels.store import (
    _same_user,
    append_label_span,
    delete_label_span,
    export_labels_to_csv,
    generate_dummy_labels,
    generate_label_summary,
    import_labels_from_csv,
    read_label_spans,
    update_label_span,
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


class TestGenerateLabelSummaryEdgeCases:
    """TC-LABEL-SUM-001..004: edge cases for generate_label_summary."""

    def _base_df(self, **extra_cols: object) -> pd.DataFrame:
        base = dt.datetime(2025, 6, 15, 10, 0)
        data: dict[str, object] = {
            "bucket_start_ts": [base + dt.timedelta(minutes=i) for i in range(5)],
        }
        data.update(extra_cols)
        return pd.DataFrame(data)

    def test_no_app_id_column(self) -> None:
        """TC-LABEL-SUM-001: missing app_id → top_apps == []."""
        df = self._base_df(session_id=["s1"] * 5)
        base = dt.datetime(2025, 6, 15, 10, 0)
        summary = generate_label_summary(df, base, base + dt.timedelta(minutes=5))
        assert summary["top_apps"] == []
        assert summary["total_buckets"] == 5

    def test_no_session_id_column(self) -> None:
        """TC-LABEL-SUM-002: missing session_id → session_count == 0."""
        df = self._base_df(app_id=["com.apple.Terminal"] * 5)
        base = dt.datetime(2025, 6, 15, 10, 0)
        summary = generate_label_summary(df, base, base + dt.timedelta(minutes=5))
        assert summary["session_count"] == 0

    def test_no_input_rate_columns(self) -> None:
        """TC-LABEL-SUM-003: missing rate columns → all means are None."""
        df = self._base_df(app_id=["com.apple.Terminal"] * 5)
        base = dt.datetime(2025, 6, 15, 10, 0)
        summary = generate_label_summary(df, base, base + dt.timedelta(minutes=5))
        assert summary["mean_keys_per_min"] is None
        assert summary["mean_clicks_per_min"] is None
        assert summary["mean_scroll_per_min"] is None

    def test_rate_columns_all_nan(self) -> None:
        """TC-LABEL-SUM-004: rate columns all NaN → means are None."""
        df = self._base_df(
            keys_per_min=[float("nan")] * 5,
            clicks_per_min=[float("nan")] * 5,
            scroll_events_per_min=[float("nan")] * 5,
        )
        base = dt.datetime(2025, 6, 15, 10, 0)
        summary = generate_label_summary(df, base, base + dt.timedelta(minutes=5))
        assert summary["mean_keys_per_min"] is None
        assert summary["mean_clicks_per_min"] is None
        assert summary["mean_scroll_per_min"] is None


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
    extend_forward: bool = False,
) -> LabelSpan:
    """Shorthand: times are (hour, minute) on a fixed date."""
    base = dt.date(2025, 6, 15)
    return LabelSpan(
        start_ts=dt.datetime(base.year, base.month, base.day, *start),
        end_ts=dt.datetime(base.year, base.month, base.day, *end),
        label=label,
        provenance="manual",
        user_id=user_id,
        extend_forward=extend_forward,
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
        """None user_id acts as wildcard — matches any concrete user."""
        a = _span((9, 0), (9, 5), user_id=None)
        b = _span((10, 0), (10, 5), user_id="u1")
        assert _same_user(a, b) is True
        assert _same_user(b, a) is True


class TestExtendForward:
    """Tests for extend_forward flag on append_label_span."""

    # -- happy path / gap-filling -----------------------------------------------

    def test_first_label_with_extend_forward(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        span = _span((10, 0), (10, 5), extend_forward=True)
        append_label_span(span, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].extend_forward is True

    def test_first_label_for_user_other_users_exist(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        other = _span((9, 0), (9, 5), user_id="u2", extend_forward=True)
        write_label_spans([other], path)

        span = _span((10, 0), (10, 5), user_id="u1")
        append_label_span(span, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 2
        u2 = [s for s in loaded if s.user_id == "u2"][0]
        assert u2.end_ts == other.end_ts

    def test_normal_gap_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 0), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 29), (10, 30), label="ReadResearch")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        extended = [s for s in loaded if s.label == "Build"][0]
        assert extended.start_ts == prev.start_ts
        assert extended.end_ts == new.start_ts

        appended = [s for s in loaded if s.label == "ReadResearch"][0]
        assert appended.start_ts == new.start_ts
        assert appended.end_ts == new.end_ts

    def test_no_extension_without_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 0))
        write_label_spans([prev], path)

        new = _span((10, 29), (10, 30), label="ReadResearch")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        build = [s for s in loaded if s.label == "Build"][0]
        assert build.end_ts == prev.end_ts

    def test_chained_extension_three_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        a = _span((9, 0), (9, 5), label="Build", extend_forward=True)
        append_label_span(a, path)

        b = _span((9, 30), (9, 35), label="Debug", extend_forward=True)
        append_label_span(b, path)

        loaded = read_label_spans(path)
        a_loaded = [s for s in loaded if s.label == "Build"][0]
        assert a_loaded.end_ts == b.start_ts

        c = _span((10, 0), (10, 5), label="Write")
        append_label_span(c, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 3
        b_loaded = [s for s in loaded if s.label == "Debug"][0]
        assert b_loaded.end_ts == c.start_ts
        a_loaded = [s for s in loaded if s.label == "Build"][0]
        assert a_loaded.end_ts == b.start_ts

    # -- truncation / overlap handling ------------------------------------------

    def test_overlap_truncates_previous(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 5), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 10), label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        truncated = [s for s in loaded if s.label == "Build"][0]
        assert truncated.end_ts == new.start_ts

    def test_adjacent_spans_no_change(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 55), (10, 0), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        b = [s for s in loaded if s.label == "Build"][0]
        assert b.end_ts == new.start_ts

    # -- guard conditions -------------------------------------------------------

    def test_same_start_as_previous_raises_overlap(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((10, 0), (10, 5), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 10), label="Debug")
        with pytest.raises(ValueError, match="overlaps"):
            append_label_span(new, path)

    def test_new_before_previous_no_overlap_succeeds(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((11, 0), (11, 5), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((9, 0), (9, 5), label="Debug")
        append_label_span(new, path)
        loaded = read_label_spans(path)
        assert len(loaded) == 2

    def test_new_inside_previous_truncates(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 10), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((9, 5), (9, 8), label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 2
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts

    # -- multi-user isolation ---------------------------------------------------

    def test_different_user_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        u1_span = _span((9, 0), (9, 5), user_id="u1", extend_forward=True)
        write_label_spans([u1_span], path)

        u2_span = _span((10, 0), (10, 5), user_id="u2", label="Debug")
        append_label_span(u2_span, path)

        loaded = read_label_spans(path)
        u1 = [s for s in loaded if s.user_id == "u1"][0]
        assert u1.end_ts == u1_span.end_ts

    def test_same_user_extended_other_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        u1_prev = _span((9, 0), (9, 5), user_id="u1", extend_forward=True)
        u2_span = _span((9, 0), (9, 5), user_id="u2")
        write_label_spans([u1_prev, u2_span], path)

        u1_new = _span((10, 0), (10, 5), user_id="u1", label="Debug")
        append_label_span(u1_new, path)

        loaded = read_label_spans(path)
        u1_old = [s for s in loaded if s.user_id == "u1" and s.label == "Build"][0]
        assert u1_old.end_ts == u1_new.start_ts
        u2 = [s for s in loaded if s.user_id == "u2"][0]
        assert u2.end_ts == u2_span.end_ts

    # -- user_id None handling --------------------------------------------------

    def test_both_none_user_extends(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5), user_id=None, extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), user_id=None, label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts

    def test_different_user_no_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5), user_id="other", extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), user_id="u1", label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        old = [s for s in loaded if s.label == "Build"][0]
        assert old.end_ts == prev.end_ts

    # -- picks correct most-recent label ----------------------------------------

    def test_picks_latest_by_start_ts(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        early = _span((8, 0), (8, 5), label="Debug")
        middle = _span((9, 0), (9, 5), label="Review")
        late = _span((9, 55), (10, 0), label="Write", extend_forward=True)
        write_label_spans([early, middle, late], path)

        new = _span((10, 29), (10, 30), label="Meet")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        by_label = {s.label: s for s in loaded}
        assert by_label["Debug"].end_ts == early.end_ts
        assert by_label["Review"].end_ts == middle.end_ts
        assert by_label["Write"].end_ts == new.start_ts

    # -- post-extension overlap validation --------------------------------------

    def test_extension_truncate_avoids_third_span_overlap(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        first = _span((9, 0), (9, 30), label="Build", extend_forward=True)
        third = _span((9, 55), (10, 0), label="Review")
        write_label_spans([first, third], path)

        new = _span((9, 20), (9, 25), label="Debug")
        append_label_span(new, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 3
        b = [s for s in loaded if s.label == "Build"][0]
        assert b.end_ts == new.start_ts

    # -- persistence round-trip -------------------------------------------------

    def test_extension_persisted_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        prev = _span((9, 0), (9, 5), extend_forward=True)
        write_label_spans([prev], path)

        new = _span((10, 0), (10, 5), label="Debug")
        append_label_span(new, path)

        reloaded = read_label_spans(path)
        old = [s for s in reloaded if s.label == "Build"][0]
        assert old.end_ts == new.start_ts

    def test_extend_forward_flag_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        span = _span((9, 0), (9, 5), extend_forward=True)
        write_label_spans([span], path)
        loaded = read_label_spans(path)
        assert loaded[0].extend_forward is True


# ---------------------------------------------------------------------------
# update_label_span
# ---------------------------------------------------------------------------


class TestUpdateLabelSpan:
    """TC-LABEL-UPD-001..005: update_label_span replaces a span's label."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-001: update an existing span's label."""
        path = tmp_path / "labels.parquet"
        s1 = _span((10, 0), (10, 5), label="Build")
        s2 = _span((11, 0), (11, 5), label="Write")
        write_label_spans([s1, s2], path)

        updated = update_label_span(s1.start_ts, s1.end_ts, "Debug", path)
        assert updated.label == "Debug"

        loaded = read_label_spans(path)
        labels = {s.label for s in loaded}
        assert "Debug" in labels
        assert "Build" not in labels
        assert "Write" in labels

    def test_file_not_found(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-002: missing file raises ValueError."""
        path = tmp_path / "no_such_file.parquet"
        with pytest.raises(ValueError, match="No labels file found"):
            update_label_span(
                dt.datetime(2025, 6, 15, 10, 0),
                dt.datetime(2025, 6, 15, 10, 5),
                "Debug",
                path,
            )

    def test_no_matching_span(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-003: no span with given timestamps raises ValueError."""
        path = tmp_path / "labels.parquet"
        write_label_spans([_span((10, 0), (10, 5))], path)

        with pytest.raises(ValueError, match="No label found"):
            update_label_span(
                dt.datetime(2025, 6, 15, 12, 0),
                dt.datetime(2025, 6, 15, 12, 5),
                "Debug",
                path,
            )

    def test_invalid_new_label(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-004: invalid label raises ValidationError."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 5))
        write_label_spans([s], path)

        with pytest.raises(ValidationError, match="Unknown label"):
            update_label_span(s.start_ts, s.end_ts, "InvalidLabel", path)

    def test_preserves_other_fields(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-005: all fields except label are preserved."""
        path = tmp_path / "labels.parquet"
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            user_id="test-user",
            confidence=0.85,
            extend_forward=True,
        )
        write_label_spans([span], path)

        updated = update_label_span(span.start_ts, span.end_ts, "Debug", path)
        assert updated.label == "Debug"
        assert updated.provenance == "manual"
        assert updated.user_id == "test-user"
        assert updated.confidence == pytest.approx(0.85)
        assert updated.extend_forward is True

    def test_update_start_ts(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-006: update only the start timestamp."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 30), label="Build")
        write_label_spans([s], path)

        updated = update_label_span(
            s.start_ts, s.end_ts, "Build", path,
            new_start_ts=dt.datetime(2025, 6, 15, 10, 5),
        )
        assert updated.start_ts == dt.datetime(2025, 6, 15, 10, 5)
        assert updated.end_ts == s.end_ts
        assert updated.label == "Build"

        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].start_ts == dt.datetime(2025, 6, 15, 10, 5)

    def test_update_end_ts(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-007: update only the end timestamp."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 30), label="Build")
        write_label_spans([s], path)

        updated = update_label_span(
            s.start_ts, s.end_ts, "Build", path,
            new_end_ts=dt.datetime(2025, 6, 15, 11, 0),
        )
        assert updated.start_ts == s.start_ts
        assert updated.end_ts == dt.datetime(2025, 6, 15, 11, 0)

    def test_update_both_timestamps(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-008: update both timestamps and label together."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 30), label="Build")
        write_label_spans([s], path)

        new_start = dt.datetime(2025, 6, 15, 9, 45)
        new_end = dt.datetime(2025, 6, 15, 11, 0)
        updated = update_label_span(
            s.start_ts, s.end_ts, "Debug", path,
            new_start_ts=new_start, new_end_ts=new_end,
        )
        assert updated.start_ts == new_start
        assert updated.end_ts == new_end
        assert updated.label == "Debug"

        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].start_ts == new_start
        assert loaded[0].end_ts == new_end
        assert loaded[0].label == "Debug"

    def test_update_timestamps_preserves_fields(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-009: changing timestamps preserves provenance/user/confidence/extend_forward."""
        path = tmp_path / "labels.parquet"
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 30),
            label="Build",
            provenance="manual",
            user_id="test-user",
            confidence=0.9,
            extend_forward=True,
        )
        write_label_spans([span], path)

        updated = update_label_span(
            span.start_ts, span.end_ts, "Build", path,
            new_start_ts=dt.datetime(2025, 6, 15, 9, 50),
            new_end_ts=dt.datetime(2025, 6, 15, 10, 40),
        )
        assert updated.provenance == "manual"
        assert updated.user_id == "test-user"
        assert updated.confidence == pytest.approx(0.9)
        assert updated.extend_forward is True

    def test_update_timestamps_none_keeps_original(self, tmp_path: Path) -> None:
        """TC-LABEL-UPD-010: passing None for new timestamps keeps originals."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 30), label="Build")
        write_label_spans([s], path)

        updated = update_label_span(
            s.start_ts, s.end_ts, "Debug", path,
            new_start_ts=None, new_end_ts=None,
        )
        assert updated.start_ts == s.start_ts
        assert updated.end_ts == s.end_ts
        assert updated.label == "Debug"


# ---------------------------------------------------------------------------
# delete_label_span
# ---------------------------------------------------------------------------


class TestDeleteLabelSpan:
    """TC-LABEL-DEL-001..005: delete_label_span removes a span."""

    def test_delete_one_of_two(self, tmp_path: Path) -> None:
        """TC-LABEL-DEL-001: delete one span, other remains."""
        path = tmp_path / "labels.parquet"
        s1 = _span((10, 0), (10, 5), label="Build")
        s2 = _span((11, 0), (11, 5), label="Write")
        write_label_spans([s1, s2], path)

        delete_label_span(s1.start_ts, s1.end_ts, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].label == "Write"

    def test_delete_only_span(self, tmp_path: Path) -> None:
        """TC-LABEL-DEL-002: deleting the only span leaves empty file."""
        path = tmp_path / "labels.parquet"
        s = _span((10, 0), (10, 5))
        write_label_spans([s], path)

        delete_label_span(s.start_ts, s.end_ts, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """TC-LABEL-DEL-003: missing file raises ValueError."""
        path = tmp_path / "no_such_file.parquet"
        with pytest.raises(ValueError, match="No labels file found"):
            delete_label_span(
                dt.datetime(2025, 6, 15, 10, 0),
                dt.datetime(2025, 6, 15, 10, 5),
                path,
            )

    def test_no_matching_span(self, tmp_path: Path) -> None:
        """TC-LABEL-DEL-004: no matching span raises ValueError."""
        path = tmp_path / "labels.parquet"
        write_label_spans([_span((10, 0), (10, 5))], path)

        with pytest.raises(ValueError, match="No label found"):
            delete_label_span(
                dt.datetime(2025, 6, 15, 12, 0),
                dt.datetime(2025, 6, 15, 12, 5),
                path,
            )

    def test_same_start_different_end(self, tmp_path: Path) -> None:
        """TC-LABEL-DEL-005: spans with same start but different end are distinct."""
        path = tmp_path / "labels.parquet"
        s1 = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
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
        write_label_spans([s1, s2], path)

        delete_label_span(s1.start_ts, s1.end_ts, path)

        loaded = read_label_spans(path)
        assert len(loaded) == 1
        assert loaded[0].label == "Debug"
        assert loaded[0].end_ts == dt.datetime(2025, 6, 15, 10, 10)


# ---------------------------------------------------------------------------
# import_labels_from_csv — error paths (item 30)
# ---------------------------------------------------------------------------


class TestImportLabelsFromCsvErrors:
    """TC-LABEL-CSV-001..003: import_labels_from_csv error paths."""

    def test_missing_label_column(self, tmp_path: Path) -> None:
        """TC-LABEL-CSV-001: missing 'label' column raises ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "start_ts,end_ts,provenance\n"
            "2025-06-15T10:00:00,2025-06-15T10:05:00,manual\n"
        )
        with pytest.raises(ValueError, match="label"):
            import_labels_from_csv(csv_path)

    def test_missing_multiple_columns(self, tmp_path: Path) -> None:
        """TC-LABEL-CSV-002: missing multiple columns lists all in error."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("start_ts,end_ts\n2025-06-15T10:00:00,2025-06-15T10:05:00\n")
        with pytest.raises(ValueError, match="CSV missing required columns") as exc_info:
            import_labels_from_csv(csv_path)
        msg = str(exc_info.value)
        assert "label" in msg
        assert "provenance" in msg

    def test_invalid_label_value(self, tmp_path: Path) -> None:
        """TC-LABEL-CSV-003: invalid label value in row raises ValidationError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "start_ts,end_ts,label,provenance\n"
            "2025-06-15T10:00:00,2025-06-15T10:05:00,NotALabel,manual\n"
        )
        with pytest.raises(ValidationError):
            import_labels_from_csv(csv_path)


# ---------------------------------------------------------------------------
# export_labels_to_csv
# ---------------------------------------------------------------------------


class TestExportLabelsToCsv:
    """TC-LABEL-EXP-001..003: export_labels_to_csv writes spans to CSV."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """TC-LABEL-EXP-001: exported CSV is re-importable and faithful."""
        parquet_path = tmp_path / "labels.parquet"
        spans = generate_dummy_labels(dt.date(2025, 6, 15), n_rows=5)
        write_label_spans(spans, parquet_path)

        csv_path = tmp_path / "export.csv"
        result = export_labels_to_csv(parquet_path, csv_path)
        assert result == csv_path
        assert csv_path.exists()

        reimported = import_labels_from_csv(csv_path)
        assert len(reimported) == len(spans)
        for original, restored in zip(spans, reimported):
            assert original.start_ts == restored.start_ts
            assert original.end_ts == restored.end_ts
            assert original.label == restored.label
            assert original.provenance == restored.provenance

    def test_no_labels_raises(self, tmp_path: Path) -> None:
        """TC-LABEL-EXP-002: empty parquet raises ValueError."""
        parquet_path = tmp_path / "labels.parquet"
        write_label_spans([], parquet_path)

        csv_path = tmp_path / "export.csv"
        with pytest.raises(ValueError, match="No labels to export"):
            export_labels_to_csv(parquet_path, csv_path)

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        """TC-LABEL-EXP-003: non-existent parquet raises ValueError."""
        parquet_path = tmp_path / "no_such_file.parquet"
        csv_path = tmp_path / "export.csv"
        with pytest.raises(ValueError, match="Labels file not found"):
            export_labels_to_csv(parquet_path, csv_path)

    def test_preserves_optional_fields(self, tmp_path: Path) -> None:
        """TC-LABEL-EXP-004: user_id and confidence survive export."""
        parquet_path = tmp_path / "labels.parquet"
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            user_id="test-user",
            confidence=0.85,
        )
        write_label_spans([span], parquet_path)

        csv_path = tmp_path / "export.csv"
        export_labels_to_csv(parquet_path, csv_path)

        df = pd.read_csv(csv_path)
        assert df.iloc[0]["user_id"] == "test-user"
        assert df.iloc[0]["confidence"] == pytest.approx(0.85)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TC-LABEL-EXP-005: parent directories are created automatically."""
        parquet_path = tmp_path / "labels.parquet"
        write_label_spans(generate_dummy_labels(dt.date(2025, 6, 15), n_rows=2), parquet_path)

        csv_path = tmp_path / "nested" / "deep" / "export.csv"
        result = export_labels_to_csv(parquet_path, csv_path)
        assert result.exists()
