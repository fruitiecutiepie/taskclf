"""Tests for report generation: flap rate, daily summaries, and export formats."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from taskclf.infer.smooth import Segment, flap_rate
from taskclf.report.daily import DailyReport, build_daily_report
from taskclf.report.export import (
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
        counts: list[float | int | None] = [2, None, 4, None, 6, None, 8, None, 10, None]
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
        for forbidden in ("raw_keystrokes", "window_title_raw", "clipboard_content", "clipboard"):
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
