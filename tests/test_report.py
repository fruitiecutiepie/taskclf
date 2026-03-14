"""Tests for report generation: flap rate, daily summaries, and export formats.

Also covers: TC-RPT-SENS-001..004 (_check_no_sensitive_fields),
TC-RPT-GUARD-001..003 (export functions reject sensitive data),
TC-RPT-DAILY-001..003, TC-RPT-CTX-001..004, TC-RPT-VAL-001..003,
TC-RPT-ROWS-001..003 (_breakdown_to_rows), TC-RPT-CSVVAL-001..004.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from pydantic import ValidationError

from taskclf.infer.smooth import Segment, flap_rate
from taskclf.report.daily import (
    ContextSwitchStats,
    DailyReport,
    _build_context_switch_stats,
    build_daily_report,
)
from taskclf.report.export import (
    _breakdown_to_rows,
    _check_no_sensitive_fields,
    export_report_csv,
    export_report_json,
    export_report_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2025, 6, 15, 10, 0, 0)


def _segs(specs: list[tuple[str, int]]) -> list[Segment]:
    ts = _BASE
    out: list[Segment] = []
    for label, count in specs:
        end = ts + timedelta(seconds=60 * count)
        out.append(Segment(start_ts=ts, end_ts=end, label=label, bucket_count=count))
        ts = end
    return out


def _basic_report(**kwargs) -> DailyReport:
    segs = _segs([("Build", 3), ("Write", 2), ("BreakIdle", 5)])
    return build_daily_report(segs, bucket_seconds=60, **kwargs)


# ---------------------------------------------------------------------------
# flap_rate()
# ---------------------------------------------------------------------------


class TestFlapRate:
    def test_all_same(self) -> None:
        assert flap_rate(["Build"] * 10) == 0.0

    def test_alternating(self) -> None:
        labels = ["Build", "Write"] * 5
        assert flap_rate(labels) == pytest.approx(0.9)

    def test_single_element(self) -> None:
        assert flap_rate(["Build"]) == 0.0

    def test_empty(self) -> None:
        assert flap_rate([]) == 0.0

    def test_one_change(self) -> None:
        labels = ["Build"] * 5 + ["Write"] * 5
        assert flap_rate(labels) == pytest.approx(0.1)

    def test_two_changes(self) -> None:
        labels = ["Build"] * 3 + ["Write"] * 3 + ["Build"] * 4
        assert flap_rate(labels) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# build_daily_report() — basic (core_breakdown from segments)
# ---------------------------------------------------------------------------


class TestBuildDailyReportBasic:
    def test_totals_match_active_time(self) -> None:
        report = _basic_report()
        assert abs(report.total_minutes - 10.0) < 0.01
        assert abs(sum(report.core_breakdown.values()) - report.total_minutes) < 0.01

    def test_core_breakdown_values(self) -> None:
        report = _basic_report()
        assert report.core_breakdown["Build"] == 3.0
        assert report.core_breakdown["Write"] == 2.0
        assert report.core_breakdown["BreakIdle"] == 5.0

    def test_segments_count(self) -> None:
        report = _basic_report()
        assert report.segments_count == 3

    def test_optional_fields_none_by_default(self) -> None:
        report = _basic_report()
        assert report.mapped_breakdown is None
        assert report.context_switch_stats is None
        assert report.flap_rate_raw is None
        assert report.flap_rate_smoothed is None

    def test_empty_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="zero segments"):
            build_daily_report([])


# ---------------------------------------------------------------------------
# build_daily_report() — mapped_labels
# ---------------------------------------------------------------------------


class TestBuildDailyReportMapped:
    def test_mapped_breakdown_populated(self) -> None:
        mapped = ["Deep Work"] * 5 + ["Break"] * 5
        report = _basic_report(mapped_labels=mapped)
        assert report.mapped_breakdown is not None
        assert report.mapped_breakdown["Deep Work"] == pytest.approx(5.0)
        assert report.mapped_breakdown["Break"] == pytest.approx(5.0)

    def test_mapped_breakdown_single_bucket(self) -> None:
        mapped = ["Focus"] * 10
        report = _basic_report(mapped_labels=mapped)
        assert report.mapped_breakdown == {"Focus": pytest.approx(10.0)}

    def test_mapped_labels_length_must_match_buckets(self) -> None:
        """Regression: mapped_labels list must have one entry per bucket.

        The Bug 11 fix in CLI ensures NaN values are filled rather than
        dropped, keeping the list the same length as the bucket count.
        """
        segs = _segs([("Build", 5), ("Write", 5)])
        mapped = ["Deep Work"] * 10
        report = build_daily_report(segs, bucket_seconds=60, mapped_labels=mapped)
        assert report.mapped_breakdown is not None
        total_mapped = sum(report.mapped_breakdown.values())
        assert total_mapped == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# build_daily_report() — flap rates
# ---------------------------------------------------------------------------


class TestBuildDailyReportFlapRate:
    def test_flap_rates_populated(self) -> None:
        raw = ["Build"] * 3 + ["Write"] * 2 + ["BreakIdle"] * 5
        smoothed = ["Build"] * 5 + ["BreakIdle"] * 5
        report = _basic_report(raw_labels=raw, smoothed_labels=smoothed)
        assert report.flap_rate_raw is not None
        assert report.flap_rate_smoothed is not None
        assert report.flap_rate_raw == pytest.approx(flap_rate(raw), abs=1e-4)
        assert report.flap_rate_smoothed == pytest.approx(flap_rate(smoothed), abs=1e-4)

    def test_flap_rate_only_raw(self) -> None:
        raw = ["Build"] * 10
        report = _basic_report(raw_labels=raw)
        assert report.flap_rate_raw == 0.0
        assert report.flap_rate_smoothed is None


# ---------------------------------------------------------------------------
# build_daily_report() — context switching stats
# ---------------------------------------------------------------------------


class TestBuildDailyReportContextSwitch:
    def test_context_switch_stats_populated(self) -> None:
        counts = [2, 4, 6, 8, 10, 3, 5, 7, 9, 1]
        report = _basic_report(app_switch_counts=counts)
        ctx = report.context_switch_stats
        assert ctx is not None
        assert ctx.buckets_counted == 10
        assert ctx.total_switches == sum(counts)
        assert ctx.max_value == 10
        assert ctx.mean == pytest.approx(5.5)

    def test_context_switch_with_nones(self) -> None:
        counts: list[float | int | None] = [
            2,
            None,
            4,
            None,
            6,
            None,
            8,
            None,
            10,
            None,
        ]
        report = _basic_report(app_switch_counts=counts)
        ctx = report.context_switch_stats
        assert ctx is not None
        assert ctx.buckets_counted == 5
        assert ctx.total_switches == 30

    def test_all_nones_gives_none(self) -> None:
        counts: list[None] = [None] * 10
        report = _basic_report(app_switch_counts=counts)
        assert report.context_switch_stats is None


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_roundtrip(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.json"
        export_report_json(report, out)
        data = json.loads(out.read_text())
        assert data["date"] == "2025-06-15"
        assert "core_breakdown" in data
        assert data["total_minutes"] == pytest.approx(10.0)

    def test_no_sensitive_fields(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.json"
        export_report_json(report, out)
        raw = out.read_text()
        for forbidden in (
            "raw_keystrokes",
            "window_title_raw",
            "clipboard_content",
            "clipboard",
        ):
            assert forbidden not in raw

    def test_excludes_none_fields(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.json"
        export_report_json(report, out)
        data = json.loads(out.read_text())
        assert "mapped_breakdown" not in data
        assert "context_switch_stats" not in data

    def test_includes_full_data(self, tmp_path: Path) -> None:
        report = _basic_report(
            raw_labels=["Build"] * 10,
            smoothed_labels=["Build"] * 10,
            mapped_labels=["Deep Work"] * 10,
            app_switch_counts=[3] * 10,
        )
        out = tmp_path / "report.json"
        export_report_json(report, out)
        data = json.loads(out.read_text())
        assert "mapped_breakdown" in data
        assert "context_switch_stats" in data
        assert "flap_rate_raw" in data
        assert "flap_rate_smoothed" in data


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_core_only(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.csv"
        export_report_csv(report, out)

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        labels = {r["label"] for r in rows}
        assert labels == {"Build", "Write", "BreakIdle"}
        assert all(r["label_type"] == "core" for r in rows)

    def test_core_and_mapped(self, tmp_path: Path) -> None:
        report = _basic_report(mapped_labels=["Deep Work"] * 5 + ["Break"] * 5)
        out = tmp_path / "report.csv"
        export_report_csv(report, out)

        with open(out) as f:
            rows = list(csv.DictReader(f))

        core_rows = [r for r in rows if r["label_type"] == "core"]
        mapped_rows = [r for r in rows if r["label_type"] == "mapped"]
        assert len(core_rows) == 3
        assert len(mapped_rows) == 2

    def test_columns(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.csv"
        export_report_csv(report, out)

        with open(out) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["date", "label_type", "label", "minutes"]


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------


class TestExportParquet:
    def test_roundtrip(self, tmp_path: Path) -> None:
        report = _basic_report()
        out = tmp_path / "report.parquet"
        export_report_parquet(report, out)

        df = pd.read_parquet(out)
        assert set(df.columns) == {"date", "label_type", "label", "minutes"}
        assert len(df) == 3

    def test_core_and_mapped(self, tmp_path: Path) -> None:
        report = _basic_report(mapped_labels=["Focus"] * 10)
        out = tmp_path / "report.parquet"
        export_report_parquet(report, out)

        df = pd.read_parquet(out)
        assert len(df[df["label_type"] == "core"]) == 3
        assert len(df[df["label_type"] == "mapped"]) == 1


# ---------------------------------------------------------------------------
# _check_no_sensitive_fields — direct tests
# ---------------------------------------------------------------------------


class TestCheckNoSensitiveFields:
    """TC-RPT-SENS-001..004: _check_no_sensitive_fields rejects forbidden keys."""

    def test_top_level_sensitive_key(self) -> None:
        with pytest.raises(ValueError, match="raw_keystrokes"):
            _check_no_sensitive_fields({"raw_keystrokes": "secret"})

    def test_nested_sensitive_key(self) -> None:
        with pytest.raises(ValueError, match="clipboard_content"):
            _check_no_sensitive_fields({"meta": {"clipboard_content": "paste"}})

    def test_all_four_sensitive_keys_rejected(self) -> None:
        for key in (
            "raw_keystrokes",
            "window_title_raw",
            "clipboard_content",
            "clipboard",
        ):
            with pytest.raises(ValueError, match=key):
                _check_no_sensitive_fields({key: "value"})

    def test_clean_dict_passes(self) -> None:
        _check_no_sensitive_fields(
            {
                "total_minutes": 10.0,
                "core_breakdown": {"Build": 5.0, "Write": 5.0},
            }
        )

    def test_sensitive_key_inside_list(self) -> None:
        """Regression: sensitive keys nested in lists must be detected."""
        with pytest.raises(ValueError, match="raw_keystrokes"):
            _check_no_sensitive_fields(
                {
                    "items": [{"raw_keystrokes": "secret"}],
                }
            )

    def test_clean_list_passes(self) -> None:
        _check_no_sensitive_fields(
            {
                "items": [{"app_id": "com.apple.Terminal", "count": 5}],
            }
        )


# ---------------------------------------------------------------------------
# Export functions reject sensitive data
# ---------------------------------------------------------------------------


class TestExportSensitiveGuard:
    """TC-RPT-GUARD-001..003: export functions raise ValueError on sensitive keys."""

    def test_json_rejects_sensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _basic_report()
        out = tmp_path / "report.json"

        original_dump = DailyReport.model_dump

        def _injected_dump(self, **kwargs):  # noqa: ANN001, ANN003
            data = original_dump(self, **kwargs)
            data["raw_keystrokes"] = "leaked"
            return data

        monkeypatch.setattr(DailyReport, "model_dump", _injected_dump)

        with pytest.raises(ValueError, match="raw_keystrokes"):
            export_report_json(report, out)

    def test_csv_rejects_sensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _basic_report()
        out = tmp_path / "report.csv"

        original_dump = DailyReport.model_dump

        def _injected_dump(self, **kwargs):  # noqa: ANN001, ANN003
            data = original_dump(self, **kwargs)
            data["clipboard_content"] = "leaked"
            return data

        monkeypatch.setattr(DailyReport, "model_dump", _injected_dump)

        with pytest.raises(ValueError, match="clipboard_content"):
            export_report_csv(report, out)

    def test_parquet_rejects_sensitive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _basic_report()
        out = tmp_path / "report.parquet"

        original_dump = DailyReport.model_dump

        def _injected_dump(self, **kwargs):  # noqa: ANN001, ANN003
            data = original_dump(self, **kwargs)
            data["window_title_raw"] = "leaked"
            return data

        monkeypatch.setattr(DailyReport, "model_dump", _injected_dump)

        with pytest.raises(ValueError, match="window_title_raw"):
            export_report_parquet(report, out)


# ---------------------------------------------------------------------------
# build_daily_report() — non-default bucket_seconds (item 21)
# ---------------------------------------------------------------------------


class TestBuildDailyReportBucketSeconds:
    """TC-RPT-DAILY-001..002: non-default bucket_seconds scales minutes."""

    def test_bucket_seconds_300(self) -> None:
        """TC-RPT-DAILY-001: 5-min buckets scale total_minutes correctly."""
        segs = _segs([("Build", 3), ("Write", 2)])
        report = build_daily_report(segs, bucket_seconds=300)
        assert report.core_breakdown["Build"] == pytest.approx(3 * 300 / 60)
        assert report.core_breakdown["Write"] == pytest.approx(2 * 300 / 60)
        assert report.total_minutes == pytest.approx(5 * 300 / 60)

    def test_bucket_seconds_120(self) -> None:
        """TC-RPT-DAILY-002: 2-min buckets scale core_breakdown correctly."""
        segs = _segs([("Build", 4), ("BreakIdle", 6)])
        report = build_daily_report(segs, bucket_seconds=120)
        assert report.core_breakdown["Build"] == pytest.approx(4 * 120 / 60)
        assert report.core_breakdown["BreakIdle"] == pytest.approx(6 * 120 / 60)


# ---------------------------------------------------------------------------
# build_daily_report() — smoothed_labels without raw_labels (item 22)
# ---------------------------------------------------------------------------


class TestBuildDailyReportSmoothedOnly:
    """TC-RPT-DAILY-003: smoothed without raw."""

    def test_smoothed_without_raw(self) -> None:
        smoothed = ["Build"] * 5 + ["BreakIdle"] * 5
        report = _basic_report(smoothed_labels=smoothed)
        assert report.flap_rate_smoothed is not None
        assert report.flap_rate_raw is None


# ---------------------------------------------------------------------------
# _build_context_switch_stats() — edge cases (item 23)
# ---------------------------------------------------------------------------


class TestBuildContextSwitchStatsEdgeCases:
    """TC-RPT-CTX-001..004: edge cases for _build_context_switch_stats."""

    def test_empty_list(self) -> None:
        """TC-RPT-CTX-001: empty list returns None."""
        assert _build_context_switch_stats([]) is None

    def test_single_element(self) -> None:
        """TC-RPT-CTX-002: single-element list."""
        stats = _build_context_switch_stats([5])
        assert stats is not None
        assert stats.mean == 5.0
        assert stats.median == 5.0
        assert stats.max_value == 5
        assert stats.total_switches == 5
        assert stats.buckets_counted == 1

    def test_float_values_truncated(self) -> None:
        """TC-RPT-CTX-003: float values truncated to int."""
        stats = _build_context_switch_stats([2.7, 3.1])
        assert stats is not None
        assert stats.total_switches == 2 + 3
        assert stats.max_value == 3

    def test_median_even_count(self) -> None:
        """TC-RPT-CTX-004: median of even-count list."""
        stats = _build_context_switch_stats([1, 2, 3, 4])
        assert stats is not None
        assert stats.median == 2.5

    def test_nan_values_filtered(self) -> None:
        """Regression: NaN values must not crash int() conversion."""
        counts: list[float | int | None] = [2, float("nan"), 4, None, float("nan")]
        stats = _build_context_switch_stats(counts)
        assert stats is not None
        assert stats.buckets_counted == 2
        assert stats.total_switches == 6


# ---------------------------------------------------------------------------
# Pydantic validation on report models (item 24)
# ---------------------------------------------------------------------------


class TestReportModelValidation:
    """TC-RPT-VAL-001..003: Field(ge=0) rejection."""

    def test_negative_context_switch_mean(self) -> None:
        """TC-RPT-VAL-001: ContextSwitchStats rejects negative mean."""
        with pytest.raises(ValidationError):
            ContextSwitchStats(
                mean=-1,
                median=1.0,
                max_value=1,
                total_switches=1,
                buckets_counted=1,
            )

    def test_negative_total_minutes(self) -> None:
        """TC-RPT-VAL-002: DailyReport rejects negative total_minutes."""
        with pytest.raises(ValidationError):
            DailyReport(
                date="2025-06-15",
                total_minutes=-1,
                core_breakdown={"Build": 5.0},
                segments_count=1,
            )

    def test_negative_segments_count(self) -> None:
        """TC-RPT-VAL-003: DailyReport rejects negative segments_count."""
        with pytest.raises(ValidationError):
            DailyReport(
                date="2025-06-15",
                total_minutes=5.0,
                core_breakdown={"Build": 5.0},
                segments_count=-1,
            )


# ---------------------------------------------------------------------------
# Export functions — parent directory creation (item 25)
# ---------------------------------------------------------------------------


class TestExportParentDirCreation:
    """TC-RPT-MKDIR-001..003: export functions create nested parent dirs."""

    def test_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TC-RPT-MKDIR-001: nested non-existent parents created for JSON."""
        report = _basic_report()
        out = tmp_path / "a" / "b" / "report.json"
        export_report_json(report, out)
        assert out.exists()

    def test_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TC-RPT-MKDIR-002: nested non-existent parents created for CSV."""
        report = _basic_report()
        out = tmp_path / "x" / "y" / "report.csv"
        export_report_csv(report, out)
        assert out.exists()

    def test_parquet_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TC-RPT-MKDIR-003: nested non-existent parents created for Parquet."""
        report = _basic_report()
        out = tmp_path / "p" / "q" / "report.parquet"
        export_report_parquet(report, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# _breakdown_to_rows() — row content correctness (item 26)
# ---------------------------------------------------------------------------


class TestBreakdownToRows:
    """TC-RPT-ROWS-001..003: _breakdown_to_rows content correctness."""

    def test_core_rows_sorted_alphabetically(self) -> None:
        """TC-RPT-ROWS-001: core rows are sorted by label name."""
        report = _basic_report()
        rows = _breakdown_to_rows(report)
        core_labels = [str(r["label"]) for r in rows if r["label_type"] == "core"]
        assert core_labels == sorted(core_labels)

    def test_minutes_match_core_breakdown(self) -> None:
        """TC-RPT-ROWS-002: each row's minutes equals the rounded breakdown value."""
        report = _basic_report()
        rows = _breakdown_to_rows(report)
        for row in rows:
            label = str(row["label"])
            assert row["minutes"] == round(report.core_breakdown[label], 2)

    def test_date_propagated(self) -> None:
        """TC-RPT-ROWS-003: report.date appears in every row."""
        report = _basic_report()
        rows = _breakdown_to_rows(report)
        assert len(rows) > 0
        assert all(r["date"] == report.date for r in rows)


# ---------------------------------------------------------------------------
# Export value correctness — CSV and Parquet (item 27)
# ---------------------------------------------------------------------------


class TestExportValueCorrectness:
    """TC-RPT-CSVVAL-001..004: exported minutes and date values match the report."""

    def test_csv_minutes_match_breakdown(self, tmp_path: Path) -> None:
        """TC-RPT-CSVVAL-001: CSV minutes column matches core_breakdown."""
        report = _basic_report()
        out = tmp_path / "report.csv"
        export_report_csv(report, out)

        with open(out) as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            expected = round(report.core_breakdown[row["label"]], 2)
            assert float(row["minutes"]) == pytest.approx(expected)

    def test_csv_date_matches_report(self, tmp_path: Path) -> None:
        """TC-RPT-CSVVAL-002: CSV date column matches report.date in all rows."""
        report = _basic_report()
        out = tmp_path / "report.csv"
        export_report_csv(report, out)

        with open(out) as f:
            rows = list(csv.DictReader(f))

        assert all(row["date"] == report.date for row in rows)

    def test_parquet_minutes_match_breakdown(self, tmp_path: Path) -> None:
        """TC-RPT-CSVVAL-003: Parquet minutes column matches core_breakdown."""
        report = _basic_report()
        out = tmp_path / "report.parquet"
        export_report_parquet(report, out)

        df = pd.read_parquet(out)
        for _, row in df.iterrows():
            expected = round(report.core_breakdown[row["label"]], 2)
            assert row["minutes"] == pytest.approx(expected)

    def test_parquet_date_matches_report(self, tmp_path: Path) -> None:
        """TC-RPT-CSVVAL-004: Parquet date column matches report.date in all rows."""
        report = _basic_report()
        out = tmp_path / "report.parquet"
        export_report_parquet(report, out)

        df = pd.read_parquet(out)
        assert all(d == report.date for d in df["date"])
