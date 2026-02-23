"""Tests for AWInputEvent type, parsing, and bucket discovery.

Covers:
- AWInputEvent construction and field validation
- parse_aw_input_export() with synthetic JSON exports
- parse_aw_input_export() returns empty list when no input bucket exists
- find_input_bucket_id() returns None when no input bucket exists
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskclf.adapters.activitywatch.types import AWInputEvent


class TestAWInputEvent:
    def test_construction(self) -> None:
        from datetime import datetime

        ev = AWInputEvent(
            timestamp=datetime(2026, 2, 23, 10, 0, 0),
            duration_seconds=5.0,
            presses=12,
            clicks=3,
            delta_x=150,
            delta_y=80,
            scroll_x=0,
            scroll_y=5,
        )
        assert ev.presses == 12
        assert ev.clicks == 3
        assert ev.delta_x == 150
        assert ev.delta_y == 80
        assert ev.scroll_x == 0
        assert ev.scroll_y == 5

    def test_frozen(self) -> None:
        from datetime import datetime

        ev = AWInputEvent(
            timestamp=datetime(2026, 2, 23, 10, 0, 0),
            duration_seconds=5.0,
            presses=0,
            clicks=0,
            delta_x=0,
            delta_y=0,
            scroll_x=0,
            scroll_y=0,
        )
        with pytest.raises(Exception):
            ev.presses = 10  # type: ignore[misc]

    def test_rejects_negative_values(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError):
            AWInputEvent(
                timestamp=datetime(2026, 2, 23, 10, 0, 0),
                duration_seconds=5.0,
                presses=-1,
                clicks=0,
                delta_x=0,
                delta_y=0,
                scroll_x=0,
                scroll_y=0,
            )


def _make_export_json(tmp_path: Path, *, include_input: bool) -> Path:
    """Create a synthetic AW export JSON file."""
    export: dict = {
        "buckets": {
            "aw-watcher-window_testhost": {
                "id": "aw-watcher-window_testhost",
                "type": "currentwindow",
                "client": "aw-watcher-window",
                "hostname": "testhost",
                "created": "2026-01-01T00:00:00.000000",
                "events": [
                    {
                        "timestamp": "2026-02-23T10:00:00Z",
                        "duration": 30.0,
                        "data": {"app": "Firefox", "title": "GitHub"},
                    },
                ],
            },
        }
    }

    if include_input:
        export["buckets"]["aw-watcher-input_testhost"] = {
            "id": "aw-watcher-input_testhost",
            "type": "os.hid.input",
            "client": "aw-watcher-input",
            "hostname": "testhost",
            "created": "2026-01-01T00:00:00.000000",
            "events": [
                {
                    "timestamp": "2026-02-23T10:00:00Z",
                    "duration": 5.0,
                    "data": {
                        "presses": 10,
                        "clicks": 2,
                        "deltaX": 100,
                        "deltaY": 50,
                        "scrollX": 0,
                        "scrollY": 3,
                    },
                },
                {
                    "timestamp": "2026-02-23T10:00:05Z",
                    "duration": 5.0,
                    "data": {
                        "presses": 8,
                        "clicks": 1,
                        "deltaX": 200,
                        "deltaY": 30,
                        "scrollX": 0,
                        "scrollY": 0,
                    },
                },
                {
                    "timestamp": "2026-02-23T10:01:00Z",
                    "duration": 5.0,
                    "data": {
                        "presses": 0,
                        "clicks": 0,
                        "deltaX": 0,
                        "deltaY": 0,
                        "scrollX": 0,
                        "scrollY": 0,
                    },
                },
            ],
        }

    f = tmp_path / "aw-export.json"
    f.write_text(json.dumps(export))
    return f


class TestParseAWInputExport:
    def test_parses_input_events(self, tmp_path: Path) -> None:
        from taskclf.adapters.activitywatch.client import parse_aw_input_export

        path = _make_export_json(tmp_path, include_input=True)
        events = parse_aw_input_export(path)
        assert len(events) == 3
        assert events[0].presses == 10
        assert events[1].presses == 8
        assert events[2].presses == 0

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        from taskclf.adapters.activitywatch.client import parse_aw_input_export

        path = _make_export_json(tmp_path, include_input=True)
        events = parse_aw_input_export(path)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_empty_when_no_input_bucket(self, tmp_path: Path) -> None:
        from taskclf.adapters.activitywatch.client import parse_aw_input_export

        path = _make_export_json(tmp_path, include_input=False)
        events = parse_aw_input_export(path)
        assert events == []

    def test_maps_camelcase_fields(self, tmp_path: Path) -> None:
        from taskclf.adapters.activitywatch.client import parse_aw_input_export

        path = _make_export_json(tmp_path, include_input=True)
        events = parse_aw_input_export(path)
        ev = events[0]
        assert ev.delta_x == 100
        assert ev.delta_y == 50
        assert ev.scroll_x == 0
        assert ev.scroll_y == 3

    def test_duration_parsed(self, tmp_path: Path) -> None:
        from taskclf.adapters.activitywatch.client import parse_aw_input_export

        path = _make_export_json(tmp_path, include_input=True)
        events = parse_aw_input_export(path)
        assert events[0].duration_seconds == 5.0
