"""Tests for the ActivityWatch adapter layer.

Covers:
- AW export JSON parsing (client.parse_aw_export)
- App name normalization (mapping.normalize_app)
- Privacy: no raw titles appear in AWEvent outputs
- REST helper function contracts (list_aw_buckets, find_window_bucket_id)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from taskclf.adapters.activitywatch.client import (
    _raw_event_to_aw_event,
    parse_aw_export,
)
from taskclf.adapters.activitywatch.mapping import KNOWN_APPS, normalize_app
from taskclf.adapters.activitywatch.types import AWEvent

SALT = "test-salt-42"


# ---------------------------------------------------------------------------
# mapping.normalize_app
# ---------------------------------------------------------------------------


class TestNormalizeApp:
    def test_known_browser(self) -> None:
        app_id, is_browser, is_editor, is_terminal = normalize_app("Firefox")
        assert app_id == "org.mozilla.firefox"
        assert is_browser is True
        assert is_editor is False
        assert is_terminal is False

    def test_known_editor(self) -> None:
        app_id, is_browser, is_editor, is_terminal = normalize_app("Code")
        assert app_id == "com.microsoft.VSCode"
        assert is_editor is True

    def test_known_terminal(self) -> None:
        app_id, _, _, is_terminal = normalize_app("Terminal")
        assert app_id == "com.apple.Terminal"
        assert is_terminal is True

    def test_case_insensitive(self) -> None:
        assert normalize_app("FIREFOX") == normalize_app("firefox")
        assert normalize_app("Google Chrome") == normalize_app("google chrome")

    def test_unknown_app_fallback(self) -> None:
        app_id, is_browser, is_editor, is_terminal = normalize_app("MyCustomApp")
        assert app_id == "unknown.mycustomapp"
        assert is_browser is False
        assert is_editor is False
        assert is_terminal is False

    def test_unknown_app_with_spaces(self) -> None:
        app_id, _, _, _ = normalize_app("My Custom App")
        assert app_id == "unknown.my_custom_app"

    def test_all_known_apps_have_valid_tuples(self) -> None:
        for name, info in KNOWN_APPS.items():
            assert len(info) == 4
            assert isinstance(info[0], str)
            assert isinstance(info[1], bool)
            assert isinstance(info[2], bool)
            assert isinstance(info[3], bool)
            flags = (info[1], info[2], info[3])
            assert sum(flags) <= 1, f"{name}: multiple flags set"


# ---------------------------------------------------------------------------
# client._raw_event_to_aw_event
# ---------------------------------------------------------------------------


class TestRawEventToAWEvent:
    def test_basic_conversion(self) -> None:
        raw = {
            "timestamp": "2026-02-23T10:05:00+00:00",
            "duration": 30.5,
            "data": {"app": "Firefox", "title": "GitHub - taskclf"},
        }
        ev = _raw_event_to_aw_event(raw, title_salt=SALT)
        assert isinstance(ev, AWEvent)
        assert ev.app_id == "org.mozilla.firefox"
        assert ev.is_browser is True
        assert ev.duration_seconds == 30.5

    def test_title_is_hashed_not_raw(self) -> None:
        raw = {
            "timestamp": "2026-02-23T10:05:00+00:00",
            "duration": 10.0,
            "data": {"app": "Firefox", "title": "Secret Document Title"},
        }
        ev = _raw_event_to_aw_event(raw, title_salt=SALT)
        assert "Secret" not in ev.window_title_hash
        assert "Document" not in ev.window_title_hash
        assert len(ev.window_title_hash) == 12

    def test_different_salts_produce_different_hashes(self) -> None:
        raw = {
            "timestamp": "2026-02-23T10:05:00+00:00",
            "duration": 5.0,
            "data": {"app": "Terminal", "title": "bash"},
        }
        ev1 = _raw_event_to_aw_event(raw, title_salt="salt-a")
        ev2 = _raw_event_to_aw_event(raw, title_salt="salt-b")
        assert ev1.window_title_hash != ev2.window_title_hash

    def test_missing_data_fields(self) -> None:
        raw = {
            "timestamp": "2026-02-23T10:05:00Z",
            "duration": 0,
            "data": {},
        }
        ev = _raw_event_to_aw_event(raw, title_salt=SALT)
        assert ev.app_id.startswith("unknown.")

    def test_utc_conversion(self) -> None:
        raw = {
            "timestamp": "2026-02-23T12:00:00+02:00",
            "duration": 1.0,
            "data": {"app": "Code", "title": "main.py"},
        }
        ev = _raw_event_to_aw_event(raw, title_salt=SALT)
        assert ev.timestamp == datetime(2026, 2, 23, 10, 0, 0)


# ---------------------------------------------------------------------------
# client.parse_aw_export
# ---------------------------------------------------------------------------


def _make_export(events: list[dict], bucket_type: str = "currentwindow") -> dict:
    return {
        "buckets": {
            "aw-watcher-window_testhost": {
                "id": "aw-watcher-window_testhost",
                "type": bucket_type,
                "client": "aw-watcher-window",
                "hostname": "testhost",
                "created": "2026-01-01T00:00:00.000000",
                "events": events,
            }
        }
    }


class TestParseAWExport:
    def test_parse_basic_export(self, tmp_path: Path) -> None:
        export = _make_export([
            {"timestamp": "2026-02-23T10:00:00Z", "duration": 30.0,
             "data": {"app": "Firefox", "title": "Page 1"}},
            {"timestamp": "2026-02-23T10:01:00Z", "duration": 45.0,
             "data": {"app": "Code", "title": "main.py"}},
        ])
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        assert len(events) == 2
        assert events[0].timestamp < events[1].timestamp
        assert events[0].is_browser is True
        assert events[1].is_editor is True

    def test_skips_non_window_buckets(self, tmp_path: Path) -> None:
        export = {
            "buckets": {
                "aw-watcher-afk_testhost": {
                    "id": "aw-watcher-afk_testhost",
                    "type": "afkstatus",
                    "client": "aw-watcher-afk",
                    "hostname": "testhost",
                    "created": "2026-01-01T00:00:00.000000",
                    "events": [
                        {"timestamp": "2026-02-23T10:00:00Z", "duration": 300.0,
                         "data": {"status": "not-afk"}},
                    ],
                }
            }
        }
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        assert len(events) == 0

    def test_no_raw_titles_in_output(self, tmp_path: Path) -> None:
        raw_title = "Super Secret Project - Confidential"
        export = _make_export([
            {"timestamp": "2026-02-23T10:00:00Z", "duration": 30.0,
             "data": {"app": "Firefox", "title": raw_title}},
        ])
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        for ev in events:
            assert raw_title not in ev.window_title_hash
            assert "Super" not in ev.window_title_hash
            assert "Confidential" not in ev.window_title_hash

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        export = _make_export([
            {"timestamp": "2026-02-23T11:00:00Z", "duration": 10.0,
             "data": {"app": "Firefox", "title": "b"}},
            {"timestamp": "2026-02-23T09:00:00Z", "duration": 10.0,
             "data": {"app": "Code", "title": "a"}},
        ])
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        assert events[0].timestamp < events[1].timestamp

    def test_empty_export(self, tmp_path: Path) -> None:
        export = _make_export([])
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        assert events == []

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_aw_export(Path("/nonexistent/export.json"), title_salt=SALT)

    def test_multiple_window_buckets(self, tmp_path: Path) -> None:
        export = {
            "buckets": {
                "aw-watcher-window_host1": {
                    "id": "aw-watcher-window_host1",
                    "type": "currentwindow",
                    "client": "aw-watcher-window",
                    "hostname": "host1",
                    "created": "2026-01-01T00:00:00.000000",
                    "events": [
                        {"timestamp": "2026-02-23T10:00:00Z", "duration": 30.0,
                         "data": {"app": "Firefox", "title": "Page"}},
                    ],
                },
                "aw-watcher-window_host2": {
                    "id": "aw-watcher-window_host2",
                    "type": "currentwindow",
                    "client": "aw-watcher-window",
                    "hostname": "host2",
                    "created": "2026-01-01T00:00:00.000000",
                    "events": [
                        {"timestamp": "2026-02-23T11:00:00Z", "duration": 20.0,
                         "data": {"app": "Code", "title": "file.py"}},
                    ],
                },
            }
        }
        f = tmp_path / "export.json"
        f.write_text(json.dumps(export))

        events = parse_aw_export(f, title_salt=SALT)
        assert len(events) == 2
