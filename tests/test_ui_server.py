"""Tests for the FastAPI labeling web UI server.

Covers REST endpoints for labels, queue, config, and feature summary.
WebSocket broadcast is tested via the EventBus integration.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import json
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from taskclf.core.types import LabelSpan
from taskclf.labels.store import write_label_spans
from taskclf.ui.copy import transition_suggestion_text
from taskclf.ui.events import EventBus
from taskclf.ui.server import _create_dev_app_from_env, create_app


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "labels_v1").mkdir()
    return tmp_path


@pytest.fixture()
def client(data_dir: Path) -> TestClient:
    app = create_app(data_dir=data_dir, event_bus=EventBus())
    return TestClient(app)


class TestConfigLabels:
    def test_returns_all_core_labels(self, client: TestClient) -> None:
        resp = client.get("/api/config/labels")
        assert resp.status_code == 200
        labels = resp.json()
        assert "Build" in labels
        assert "Meet" in labels
        assert len(labels) == 8


class TestLabelsCRUD:
    def test_empty_labels(self, client: TestClient) -> None:
        resp = client.get("/api/labels")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_list(self, client: TestClient) -> None:
        body = {
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
            "user_id": "test-user",
            "confidence": 0.9,
        }
        resp = client.post("/api/labels", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["label"] == "Build"
        assert data["provenance"] == "manual"

        resp = client.get("/api/labels")
        assert len(resp.json()) == 1

    def test_create_invalid_label_rejected(self, client: TestClient) -> None:
        body = {
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "NotALabel",
        }
        resp = client.post("/api/labels", json=body)
        assert resp.status_code >= 400

    def test_create_overlapping_rejected(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        }
        resp = client.post("/api/labels", json=body1)
        assert resp.status_code == 201

        body2 = {
            "start_ts": "2026-02-27T09:30:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "Meet",
        }
        resp = client.post("/api/labels", json=body2)
        assert resp.status_code >= 400

    def test_labels_ordered_most_recent_first(self, client: TestClient) -> None:
        for hour in (8, 10, 9):
            body = {
                "start_ts": f"2026-02-27T{hour:02d}:00:00",
                "end_ts": f"2026-02-27T{hour:02d}:30:00",
                "label": "Build",
            }
            client.post("/api/labels", json=body)

        resp = client.get("/api/labels")
        labels = resp.json()
        assert len(labels) == 3
        assert labels[0]["end_ts"] > labels[1]["end_ts"]

    def test_current_label_empty_returns_null(self, client: TestClient) -> None:
        resp = client.get("/api/labels/current")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_current_label_returns_open_ended_span_even_when_latest_end_differs(
        self, client: TestClient
    ) -> None:
        create_current = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T09:00:00",
                "label": "Build",
                "extend_forward": True,
            },
        )
        assert create_current.status_code == 201

        create_overlap = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T08:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Write",
                "allow_overlap": True,
            },
        )
        assert create_overlap.status_code == 201

        latest_ended = client.get("/api/labels", params={"limit": 1})
        assert latest_ended.status_code == 200
        assert latest_ended.json()[0]["label"] == "Write"

        current = client.get("/api/labels/current")
        assert current.status_code == 200
        data = current.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["label"] == "Build"
        assert data["provenance"] == "manual"
        assert data["extend_forward"] is True


class TestQueue:
    def test_empty_queue(self, client: TestClient) -> None:
        resp = client.get("/api/queue")
        assert resp.status_code == 200
        assert resp.json() == []


class TestFeatureSummary:
    def test_no_features_returns_empty(self, client: TestClient) -> None:
        resp = client.get(
            "/api/features/summary",
            params={"start": "2026-02-27T09:00:00", "end": "2026-02-27T10:00:00"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_buckets"] == 0


class TestAWLive:
    def test_aw_unavailable_returns_empty(self, client: TestClient) -> None:
        resp = client.get(
            "/api/aw/live",
            params={"start": "2026-02-27T09:00:00", "end": "2026-02-27T10:00:00"},
        )
        assert resp.status_code == 200
        assert resp.json() == []


class TestWindowControl:
    def test_toggle_without_window_api(self, client: TestClient) -> None:
        resp = client.post("/api/window/toggle")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_window"

    def test_state_without_window_api(self, client: TestClient) -> None:
        resp = client.get("/api/window/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False

    def test_toggle_with_window_api(self, data_dir: Path) -> None:
        from taskclf.ui.window import WindowAPI

        win_api = WindowAPI()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), window_api=win_api)
        client = TestClient(app)

        resp = client.get("/api/window/state")
        assert resp.json()["available"] is True
        assert resp.json()["visible"] is True

        resp = client.post("/api/window/toggle")
        assert resp.json()["status"] == "ok"
        assert resp.json()["visible"] is False

        resp = client.post("/api/window/toggle")
        assert resp.json()["visible"] is True


class TestExtendForwardAPI:
    """API-level tests for the per-label extend_forward flag on POST /api/labels."""

    def test_extend_forward_on_previous_extends_gap(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
            "extend_forward": True,
        }
        resp = client.post("/api/labels", json=body1)
        assert resp.status_code == 201

        body2 = {
            "start_ts": "2026-02-27T10:29:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "ReadResearch",
        }
        resp = client.post("/api/labels", json=body2)
        assert resp.status_code == 201

        labels = client.get("/api/labels").json()
        assert len(labels) == 2
        build = [rec for rec in labels if rec["label"] == "Build"][0]
        assert build["end_ts"] == "2026-02-27T10:29:00+00:00"

    def test_extend_forward_off_no_extension(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        }
        resp = client.post("/api/labels", json=body1)
        assert resp.status_code == 201

        body2 = {
            "start_ts": "2026-02-27T10:29:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "ReadResearch",
        }
        resp = client.post("/api/labels", json=body2)
        assert resp.status_code == 201

        labels = client.get("/api/labels").json()
        build = [rec for rec in labels if rec["label"] == "Build"][0]
        assert build["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_extend_forward_flag_on_previous_not_request(
        self, client: TestClient
    ) -> None:
        """Extension is driven by the *previous* label's flag, not the current request."""
        body1 = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
            "extend_forward": True,
        }
        client.post("/api/labels", json=body1)

        body2 = {
            "start_ts": "2026-02-27T10:29:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "ReadResearch",
            "extend_forward": False,
        }
        client.post("/api/labels", json=body2)

        labels = client.get("/api/labels").json()
        build = [rec for rec in labels if rec["label"] == "Build"][0]
        assert build["end_ts"] == "2026-02-27T10:29:00+00:00"

    def test_extend_forward_defaults_to_false(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        }
        client.post("/api/labels", json=body1)

        body2 = {
            "start_ts": "2026-02-27T10:29:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "ReadResearch",
        }
        client.post("/api/labels", json=body2)

        labels = client.get("/api/labels").json()
        build = [rec for rec in labels if rec["label"] == "Build"][0]
        assert build["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_extend_forward_201_response_correct(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
            "extend_forward": True,
        }
        client.post("/api/labels", json=body1)

        body2 = {
            "start_ts": "2026-02-27T10:29:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "Debug",
            "user_id": "default-user",
            "confidence": 0.8,
            "extend_forward": True,
        }
        resp = client.post("/api/labels", json=body2)
        assert resp.status_code == 201
        data = resp.json()
        assert data["label"] == "Debug"
        assert data["provenance"] == "manual"
        assert data["start_ts"] == "2026-02-27T10:29:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:30:00+00:00"
        assert data["extend_forward"] is True

    def test_extend_forward_response_flag_persisted(self, client: TestClient) -> None:
        body = {
            "start_ts": "2026-02-27T09:55:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
            "extend_forward": True,
        }
        client.post("/api/labels", json=body)

        labels = client.get("/api/labels").json()
        assert labels[0]["extend_forward"] is True

    def test_extend_forward_overlap_409(self, client: TestClient) -> None:
        body1 = {
            "start_ts": "2026-02-27T08:00:00",
            "end_ts": "2026-02-27T08:30:00",
            "label": "Build",
            "extend_forward": True,
        }
        client.post("/api/labels", json=body1)

        body2 = {
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T09:10:00",
            "label": "Write",
        }
        client.post("/api/labels", json=body2)

        body3 = {
            "start_ts": "2026-02-27T08:50:00",
            "end_ts": "2026-02-27T09:05:00",
            "label": "Debug",
        }
        resp = client.post("/api/labels", json=body3)
        assert resp.status_code == 409

    def test_invalid_label_still_422(self, client: TestClient) -> None:
        body = {
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "NotALabel",
            "extend_forward": True,
        }
        resp = client.post("/api/labels", json=body)
        assert resp.status_code == 422

    def test_now_label_handoff_truncates_active_previous(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T09:30:00",
                "label": "Debug",
                "extend_forward": True,
            },
        )
        assert resp.status_code == 201

        labels = client.get("/api/labels").json()
        assert len(labels) == 2
        by_label = {rec["label"]: rec for rec in labels}
        assert by_label["Build"]["end_ts"] == "2026-02-27T09:30:00+00:00"
        assert by_label["Debug"]["start_ts"] == "2026-02-27T09:30:00+00:00"
        assert by_label["Debug"]["end_ts"] == "2026-02-27T09:30:00+00:00"


class TestWebSocket:
    def test_ws_connects(self, data_dir: Path) -> None:
        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        client = TestClient(app)

        with client.websocket_connect("/ws/predictions") as ws:
            bus.publish_threadsafe(
                {"type": "status", "state": "idle", "current_app": "test"}
            )
            import time

            time.sleep(0.1)
            ws.close()

    def test_event_delivery_roundtrip(self, data_dir: Path) -> None:
        """TC-UI-WS-003: publish via EventBus, read from WS, verify match."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        event = {"type": "status", "state": "collecting", "current_app": "Terminal"}

        with (
            TestClient(app) as client,
            client.websocket_connect("/ws/predictions") as ws,
        ):

            def _send() -> None:
                __import__("time").sleep(0.05)
                bus.publish_threadsafe(event)

            threading.Thread(target=_send).start()
            received = ws.receive_json()
            assert received == event

    def test_multiple_event_types_in_order(self, data_dir: Path) -> None:
        """TC-UI-WS-004: multiple event types received in publish order."""
        import threading
        import time
        from typing import Any

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        events: list[dict[str, Any]] = [
            {"type": "status", "state": "idle"},
            {"type": "prediction", "label": "Build", "confidence": 0.9},
            {"type": "tray_state", "model_loaded": True},
            {"type": "suggest_label", "reason": "app_switch"},
        ]

        def _publish_all() -> None:
            time.sleep(0.05)
            for ev in events:
                bus.publish_threadsafe(ev)

        with (
            TestClient(app) as client,
            client.websocket_connect("/ws/predictions") as ws,
        ):
            threading.Thread(target=_publish_all).start()
            received = [ws.receive_json() for _ in events]
            assert received == events

    def test_multiple_subscribers(self, data_dir: Path) -> None:
        """TC-UI-WS-005: two WS clients both receive the same event."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        event = {"type": "prediction", "label": "Write"}

        with TestClient(app) as client:
            with (
                client.websocket_connect("/ws/predictions") as ws1,
                client.websocket_connect("/ws/predictions") as ws2,
            ):

                def _send() -> None:
                    __import__("time").sleep(0.05)
                    bus.publish_threadsafe(event)

                threading.Thread(target=_send).start()
                r1 = ws1.receive_json()
                r2 = ws2.receive_json()
                assert r1 == event
                assert r2 == event

    def test_disconnect_no_server_error(self, data_dir: Path) -> None:
        """TC-UI-WS-006: disconnect then publish causes no server error."""
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/predictions") as ws:
                ws.close()

            bus.publish_threadsafe({"type": "status", "state": "idle"})
            time.sleep(0.1)

            resp = client.get("/api/config/labels")
            assert resp.status_code == 200

    def test_ws_snapshot_empty(self, data_dir: Path) -> None:
        """TC-UI-WS-SNAP-001: snapshot returns empty dict when no events published."""
        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        with TestClient(app) as client:
            resp = client.get("/api/ws/snapshot")
            assert resp.status_code == 200
            assert resp.json() == {}

    def test_ws_snapshot_after_publish(self, data_dir: Path) -> None:
        """TC-UI-WS-SNAP-002: snapshot reflects latest published events."""
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        with TestClient(app) as client:
            bus.publish_threadsafe({"type": "status", "state": "idle"})
            bus.publish_threadsafe({"type": "status", "state": "collecting"})
            bus.publish_threadsafe({"type": "tray_state", "paused": False})
            time.sleep(0.1)

            resp = client.get("/api/ws/snapshot")
            assert resp.status_code == 200
            snap = resp.json()
            assert snap["status"]["state"] == "collecting"
            assert snap["tray_state"]["paused"] is False


# ---------------------------------------------------------------------------
# PUT /api/labels  (Item 34)
# ---------------------------------------------------------------------------


class TestUpdateLabel:
    """TC-UI-UPD-001 through TC-UI-UPD-004."""

    def _create_label(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

    def test_happy_path_update(self, client: TestClient) -> None:
        """TC-UI-UPD-001: POST then PUT with new label -> 200, label changed."""
        self._create_label(client)
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "Write"

    def test_404_no_matching_span(self, client: TestClient) -> None:
        """TC-UI-UPD-002: PUT with non-existent timestamps -> 404."""
        self._create_label(client)
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T11:00:00",
                "end_ts": "2026-02-27T12:00:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 404

    def test_422_invalid_timestamp(self, client: TestClient) -> None:
        """TC-UI-UPD-003: Malformed start_ts -> 422."""
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "not-a-date",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 422

    def test_updated_label_persisted(self, client: TestClient) -> None:
        """TC-UI-UPD-004: PUT then GET -> updated label visible."""
        self._create_label(client)
        client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Debug",
            },
        )
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["label"] == "Debug"

    def test_update_timestamps(self, client: TestClient) -> None:
        """TC-UI-UPD-005: PUT with new_start_ts/new_end_ts changes the time range."""
        self._create_label(client)
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
                "new_start_ts": "2026-02-27T08:30:00",
                "new_end_ts": "2026-02-27T10:30:00",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T08:30:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:30:00+00:00"
        assert data["label"] == "Build"

    def test_update_timestamps_and_label(self, client: TestClient) -> None:
        """TC-UI-UPD-006: PUT with new timestamps and new label together."""
        self._create_label(client)
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Debug",
                "new_start_ts": "2026-02-27T09:15:00",
                "new_end_ts": "2026-02-27T09:45:00",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Debug"
        assert data["start_ts"] == "2026-02-27T09:15:00+00:00"
        assert data["end_ts"] == "2026-02-27T09:45:00+00:00"

    def test_update_timestamps_persisted(self, client: TestClient) -> None:
        """TC-UI-UPD-007: PUT with new timestamps then GET shows updated times."""
        self._create_label(client)
        client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
                "new_start_ts": "2026-02-27T08:00:00",
                "new_end_ts": "2026-02-27T11:00:00",
            },
        )
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["start_ts"] == "2026-02-27T08:00:00+00:00"
        assert labels[0]["end_ts"] == "2026-02-27T11:00:00+00:00"

    def test_update_without_new_timestamps(self, client: TestClient) -> None:
        """TC-UI-UPD-008: PUT without new_start_ts/new_end_ts keeps original times."""
        self._create_label(client)
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"
        assert data["label"] == "Write"

    def test_stop_current_open_ended_label(self, client: TestClient) -> None:
        """TC-UI-UPD-009: PUT can stop a running extend-forward label at a concrete end time."""
        create_resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T09:00:00",
                "label": "Build",
                "extend_forward": True,
            },
        )
        assert create_resp.status_code == 201

        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T09:00:00",
                "label": "Build",
                "new_end_ts": "2026-02-27T09:30:00",
                "extend_forward": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T09:30:00+00:00"
        assert data["extend_forward"] is False

        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["end_ts"] == "2026-02-27T09:30:00+00:00"
        assert labels[0]["extend_forward"] is False


# ---------------------------------------------------------------------------
# DELETE /api/labels  (Item 35)
# ---------------------------------------------------------------------------


class TestDeleteLabel:
    """TC-UI-DEL-001 through TC-UI-DEL-004."""

    def _create_label(
        self, client: TestClient, start: str, end: str, label: str = "Build"
    ) -> None:
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": start,
                "end_ts": end,
                "label": label,
            },
        )
        assert resp.status_code == 201

    def test_happy_path_delete(self, client: TestClient) -> None:
        """TC-UI-DEL-001: POST then DELETE -> 200, status deleted."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        resp = client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}

    def test_404_no_matching_span(self, client: TestClient) -> None:
        """TC-UI-DEL-002: DELETE with non-existent timestamps -> 404."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        resp = client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "2026-02-27T11:00:00",
                "end_ts": "2026-02-27T12:00:00",
            },
        )
        assert resp.status_code == 404

    def test_span_removed_from_storage(self, client: TestClient) -> None:
        """TC-UI-DEL-003: POST, DELETE, then GET -> empty list."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
            },
        )
        labels = client.get("/api/labels").json()
        assert labels == []

    def test_422_invalid_timestamp(self, client: TestClient) -> None:
        """TC-UI-DEL-004: Malformed timestamps -> 422."""
        resp = client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "garbage",
                "end_ts": "2026-02-27T10:00:00",
            },
        )
        assert resp.status_code == 422


class TestLabelsChangedEvents:
    def test_create_publishes_labels_changed(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

        snap = client.get("/api/ws/snapshot").json()
        assert snap["labels_changed"]["reason"] == "created"

    def test_update_publishes_labels_changed(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

        client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Write",
            },
        )

        snap = client.get("/api/ws/snapshot").json()
        assert snap["labels_changed"]["reason"] == "updated"

    def test_delete_publishes_labels_changed(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

        client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
            },
        )

        snap = client.get("/api/ws/snapshot").json()
        assert snap["labels_changed"]["reason"] == "deleted"

    def test_notification_accept_publishes_labels_changed(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:00:00",
                "block_end": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )

        snap = client.get("/api/ws/snapshot").json()
        assert snap["labels_changed"]["reason"] == "suggestion_accepted"


# ---------------------------------------------------------------------------
# POST /api/queue/{request_id}/done  (Item 36)
# ---------------------------------------------------------------------------


def _seed_queue(queue_path: Path) -> str:
    """Write a single pending item to the queue JSON and return its request_id."""
    rid = str(uuid.uuid4())
    item = {
        "request_id": rid,
        "user_id": "u1",
        "bucket_start_ts": "2026-02-27T09:00:00+00:00",
        "bucket_end_ts": "2026-02-27T09:01:00+00:00",
        "reason": "low_confidence",
        "confidence": 0.3,
        "predicted_label": "Build",
        "created_at": "2026-02-27T08:00:00+00:00",
        "status": "pending",
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps([item]))
    return rid


class TestMarkQueueDone:
    """TC-UI-QD-001 through TC-UI-QD-004."""

    def test_mark_labeled(self, data_dir: Path, client: TestClient) -> None:
        """TC-UI-QD-001: Mark existing item 'labeled'."""
        rid = _seed_queue(data_dir / "labels_v1" / "queue.json")
        resp = client.post(f"/api/queue/{rid}/done", json={"status": "labeled"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "labeled"}

    def test_mark_skipped(self, data_dir: Path, client: TestClient) -> None:
        """TC-UI-QD-002: Mark existing item 'skipped'."""
        rid = _seed_queue(data_dir / "labels_v1" / "queue.json")
        resp = client.post(f"/api/queue/{rid}/done", json={"status": "skipped"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "skipped"}

    def test_not_found_id(self, data_dir: Path, client: TestClient) -> None:
        """TC-UI-QD-003: Non-existent request_id -> not_found."""
        _seed_queue(data_dir / "labels_v1" / "queue.json")
        resp = client.post("/api/queue/nonexistent-id/done", json={"status": "labeled"})
        assert resp.json() == {"status": "not_found"}

    def test_no_queue_file(self, client: TestClient) -> None:
        """TC-UI-QD-004: No queue file -> not_found."""
        resp = client.post("/api/queue/any-id/done", json={"status": "labeled"})
        assert resp.json() == {"status": "not_found"}


# ---------------------------------------------------------------------------
# GET /api/config/user  (Item 37)
# ---------------------------------------------------------------------------


class TestGetUserConfig:
    """TC-UI-CU-001 through TC-UI-CU-002."""

    def test_returns_default_config(self, client: TestClient) -> None:
        """TC-UI-CU-001: Returns user_id (UUID), username, and suggestion TTL."""
        resp = client.get("/api/config/user")
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        assert "username" in data
        assert "suggestion_banner_ttl_seconds" in data
        assert isinstance(data["suggestion_banner_ttl_seconds"], int)
        assert data["suggestion_banner_ttl_seconds"] >= 0
        uuid.UUID(data["user_id"])  # validates it's a real UUID

    def test_user_id_stable_across_requests(self, client: TestClient) -> None:
        """TC-UI-CU-002: Two GETs return the same user_id."""
        id1 = client.get("/api/config/user").json()["user_id"]
        id2 = client.get("/api/config/user").json()["user_id"]
        assert id1 == id2


# ---------------------------------------------------------------------------
# PUT /api/config/user  (Item 38)
# ---------------------------------------------------------------------------


class TestUpdateUserConfig:
    """TC-UI-CU-003 through TC-UI-CU-006."""

    def test_update_username(self, client: TestClient) -> None:
        """TC-UI-CU-003: PUT username -> 200, response has new username."""
        resp = client.put("/api/config/user", json={"username": "alice"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "alice"

    def test_persisted_across_requests(self, client: TestClient) -> None:
        """TC-UI-CU-004: PUT then GET -> same username."""
        client.put("/api/config/user", json={"username": "bob"})
        resp = client.get("/api/config/user")
        assert resp.json()["username"] == "bob"

    def test_empty_body_noop(self, client: TestClient) -> None:
        """TC-UI-CU-005: PUT {} -> 200, config unchanged."""
        original = client.get("/api/config/user").json()
        resp = client.put("/api/config/user", json={})
        assert resp.status_code == 200
        after = client.get("/api/config/user").json()
        assert after == original

    def test_user_id_unchanged_after_update(self, client: TestClient) -> None:
        """TC-UI-CU-006: PUT with username, user_id stays the same."""
        original_id = client.get("/api/config/user").json()["user_id"]
        client.put("/api/config/user", json={"username": "carol"})
        new_id = client.get("/api/config/user").json()["user_id"]
        assert new_id == original_id

    def test_update_suggestion_banner_ttl_seconds(self, client: TestClient) -> None:
        """TC-UI-CU-007: PUT suggestion_banner_ttl_seconds persists and GET returns it."""
        resp = client.put(
            "/api/config/user",
            json={"suggestion_banner_ttl_seconds": 600},
        )
        assert resp.status_code == 200
        assert resp.json()["suggestion_banner_ttl_seconds"] == 600
        assert (
            client.get("/api/config/user").json()["suggestion_banner_ttl_seconds"]
            == 600
        )


# ---------------------------------------------------------------------------
# POST /api/notification/accept  (Item 39)
# ---------------------------------------------------------------------------


class TestNotificationAccept:
    """TC-UI-NA-001 through TC-UI-NA-005."""

    def test_accept_valid_suggestion(self, client: TestClient) -> None:
        """TC-UI-NA-001: Accept -> 200, provenance is 'suggestion'."""
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:00:00",
                "block_end": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provenance"] == "suggestion"
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_invalid_label_422(self, client: TestClient) -> None:
        """TC-UI-NA-002: Invalid label -> 422."""
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:00:00",
                "block_end": "2026-02-27T10:00:00",
                "label": "NotReal",
            },
        )
        assert resp.status_code == 422

    def test_invalid_timestamps_422(self, client: TestClient) -> None:
        """TC-UI-NA-003: Malformed ISO timestamps -> 422."""
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "not-a-date",
                "block_end": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        assert resp.status_code == 422

    def test_overlap_409(self, client: TestClient) -> None:
        """TC-UI-NA-004: Overlap with existing span -> 409."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:30:00",
                "block_end": "2026-02-27T10:30:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 409

    def test_label_persisted(self, client: TestClient) -> None:
        """TC-UI-NA-005: Accept, then GET -> label visible."""
        client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:00:00",
                "block_end": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["provenance"] == "suggestion"


# ---------------------------------------------------------------------------
# POST /api/notification/skip  (Item 40)
# ---------------------------------------------------------------------------


class TestNotificationSkip:
    """TC-UI-NS-001."""

    def test_skip_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/api/notification/skip")
        assert resp.status_code == 200
        assert resp.json() == {"status": "skipped"}


# ---------------------------------------------------------------------------
# POST /api/window/show-label-grid  (Item 41)
# ---------------------------------------------------------------------------


class TestShowLabelGrid:
    """TC-UI-WS-001 through TC-UI-WS-002."""

    def test_without_window_api(self, client: TestClient) -> None:
        """TC-UI-WS-001: No window_api -> still returns ok."""
        resp = client.post("/api/window/show-label-grid")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_with_window_api(self, data_dir: Path) -> None:
        """TC-UI-WS-002: With window_api -> label_grid_show() called."""
        from unittest.mock import MagicMock

        mock_api = MagicMock()
        mock_api.visible = True
        app = create_app(data_dir=data_dir, event_bus=EventBus(), window_api=mock_api)
        tc = TestClient(app)

        resp = tc.post("/api/window/show-label-grid")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_api.label_grid_show.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/labels — limit parameter  (Item 42)
# ---------------------------------------------------------------------------


class TestLabelsLimit:
    """TC-UI-LL-001 through TC-UI-LL-004."""

    def _seed_labels(self, client: TestClient, count: int = 3) -> None:
        for i in range(count):
            client.post(
                "/api/labels",
                json={
                    "start_ts": f"2026-02-27T{8 + i:02d}:00:00",
                    "end_ts": f"2026-02-27T{8 + i:02d}:30:00",
                    "label": "Build",
                },
            )

    def test_limit_1(self, client: TestClient) -> None:
        """TC-UI-LL-001: limit=1 with 3 labels -> exactly 1."""
        self._seed_labels(client, 3)
        resp = client.get("/api/labels", params={"limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_limit_500(self, client: TestClient) -> None:
        """TC-UI-LL-002: limit=500 -> no error, returns all."""
        self._seed_labels(client, 3)
        resp = client.get("/api/labels", params={"limit": 500})
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_limit_0_rejected(self, client: TestClient) -> None:
        """TC-UI-LL-003: limit=0 violates ge=1 -> 422."""
        resp = client.get("/api/labels", params={"limit": 0})
        assert resp.status_code == 422

    def test_limit_501_rejected(self, client: TestClient) -> None:
        """TC-UI-LL-004: limit=501 violates le=500 -> 422."""
        resp = client.get("/api/labels", params={"limit": 501})
        assert resp.status_code == 422


class TestLabelsLatestEndOrdering:
    """GET /api/labels ordering must match quick-label gap (latest end_ts first)."""

    def test_limit_1_prefers_latest_end_ts_when_overlaps_allowed(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T14:00:00",
                "label": "Build",
            },
        )
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T13:00:00",
                "end_ts": "2026-02-27T13:30:00",
                "label": "Meet",
                "allow_overlap": True,
            },
        )
        resp = client.get("/api/labels", params={"limit": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["end_ts"] == "2026-02-27T14:00:00+00:00"
        assert data[0]["label"] == "Build"


# ---------------------------------------------------------------------------
# on_label_saved callback  (Item 9)
# ---------------------------------------------------------------------------


class TestOnLabelSavedCallback:
    """TC-UI-OLS-001 through TC-UI-OLS-004: verify on_label_saved fires."""

    def test_callback_invoked_on_create_label(self, data_dir: Path) -> None:
        """TC-UI-OLS-001: POST /api/labels calls on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        cb.assert_called_once()

    def test_callback_invoked_on_notification_accept(self, data_dir: Path) -> None:
        """TC-UI-OLS-002: POST /api/notification/accept calls on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:00:00",
                "block_end": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        cb.assert_called_once()

    def test_callback_not_invoked_on_validation_error(self, data_dir: Path) -> None:
        """TC-UI-OLS-003: 422 error does not call on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "NotALabel",
            },
        )
        cb.assert_not_called()

    def test_callback_not_invoked_on_overlap_error(self, data_dir: Path) -> None:
        """TC-UI-OLS-004: 409 overlap does not call on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        cb.reset_mock()

        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Meet",
            },
        )
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/tray/pause, GET /api/tray/state  (Item 4)
# ---------------------------------------------------------------------------


class TestPauseAPI:
    def test_pause_unavailable_without_callbacks(self, client: TestClient) -> None:
        resp = client.post("/api/tray/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"

    def test_state_unavailable_without_callbacks(self, client: TestClient) -> None:
        resp = client.get("/api/tray/state")
        assert resp.status_code == 200
        assert resp.json()["available"] is False

    def test_pause_toggle_with_callbacks(self, data_dir: Path) -> None:
        paused_state = {"paused": False}

        def toggle() -> bool:
            paused_state["paused"] = not paused_state["paused"]
            return paused_state["paused"]

        def is_paused() -> bool:
            return paused_state["paused"]

        app = create_app(
            data_dir=data_dir,
            event_bus=EventBus(),
            pause_toggle=toggle,
            is_paused=is_paused,
        )
        client = TestClient(app)

        resp = client.post("/api/tray/pause")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "paused": True}

        resp = client.post("/api/tray/pause")
        assert resp.json() == {"status": "ok", "paused": False}

    def test_state_with_callbacks(self, data_dir: Path) -> None:
        paused_state = {"paused": False}

        app = create_app(
            data_dir=data_dir,
            event_bus=EventBus(),
            pause_toggle=lambda: True,
            is_paused=lambda: paused_state["paused"],
        )
        client = TestClient(app)

        resp = client.get("/api/tray/state")
        assert resp.status_code == 200
        assert resp.json() == {"available": True, "paused": False}

        paused_state["paused"] = True
        resp = client.get("/api/tray/state")
        assert resp.json() == {"available": True, "paused": True}


# ---------------------------------------------------------------------------
# Structured 409 overlap error  (Item 6)
# ---------------------------------------------------------------------------


class TestStructuredOverlapError:
    """Verify 409 responses contain structured conflict details."""

    def test_overlap_returns_structured_detail(self, client: TestClient) -> None:
        """TC-UI-OVL-001: POST /api/labels overlap includes conflicting span."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Meet",
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "error" in detail
        assert detail["conflicting_start_ts"] is not None
        assert detail["conflicting_end_ts"] is not None
        assert "2026-02-27" in detail["conflicting_start_ts"]
        assert "2026-02-27" in detail["conflicting_end_ts"]
        assert detail["conflicting_label"] == "Build"

    def test_overlap_conflicting_span_is_existing(self, client: TestClient) -> None:
        """TC-UI-OVL-002: the conflicting span points to the previously saved label."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Meet",
            },
        )
        detail = resp.json()["detail"]
        assert "09:00:00" in detail["conflicting_start_ts"]
        assert "10:00:00" in detail["conflicting_end_ts"]

    def test_notification_accept_overlap_structured(self, client: TestClient) -> None:
        """TC-UI-OVL-003: POST /api/notification/accept overlap is also structured."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:30:00",
                "block_end": "2026-02-27T10:30:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "error" in detail
        assert detail["conflicting_start_ts"] is not None
        assert detail["conflicting_end_ts"] is not None

    def test_notification_accept_with_overwrite_succeeds(
        self, client: TestClient
    ) -> None:
        """Same range as overlap failure, but overwrite=true applies suggestion."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:30:00",
                "block_end": "2026-02-27T10:30:00",
                "label": "Write",
                "overwrite": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provenance"] == "suggestion"
        assert data["label"] == "Write"

    def test_notification_accept_with_allow_overlap_succeeds(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/notification/accept",
            json={
                "block_start": "2026-02-27T09:30:00",
                "block_end": "2026-02-27T10:30:00",
                "label": "Write",
                "allow_overlap": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["provenance"] == "suggestion"

    def test_historical_overlap_does_not_cause_false_409(
        self, client: TestClient
    ) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Debug",
                "allow_overlap": True,
            },
        )

        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T11:00:00",
                "end_ts": "2026-02-27T11:01:00",
                "label": "Write",
            },
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Overwrite overlap resolution
# ---------------------------------------------------------------------------


class TestOverwriteLabel:
    """Verify POST /api/labels with overwrite=true resolves overlaps."""

    def test_overwrite_truncates_existing(self, client: TestClient) -> None:
        """Overlapping span is truncated when overwrite is true."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Meet",
                "overwrite": True,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["label"] == "Meet"

        labels = client.get("/api/labels").json()
        assert len(labels) == 2
        by_label = {rec["label"]: rec for rec in labels}
        assert "09:30:00" in by_label["Build"]["end_ts"]
        assert "09:30:00" in by_label["Meet"]["start_ts"]

    def test_overwrite_removes_contained(self, client: TestClient) -> None:
        """Existing span fully inside new span is removed."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:10:00",
                "end_ts": "2026-02-27T09:20:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T09:30:00",
                "label": "Debug",
                "overwrite": True,
            },
        )
        assert resp.status_code == 201
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["label"] == "Debug"

    def test_overwrite_false_still_rejects(self, client: TestClient) -> None:
        """overwrite=false (default) still returns 409."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:30:00",
                "end_ts": "2026-02-27T10:30:00",
                "label": "Meet",
                "overwrite": False,
            },
        )
        assert resp.status_code == 409

    def test_overlap_409_lists_all_conflicts(self, client: TestClient) -> None:
        """409 response lists every conflicting span, not just the first."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T09:30:00",
                "label": "Build",
            },
        )
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:40:00",
                "end_ts": "2026-02-27T10:10:00",
                "label": "Debug",
            },
        )
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:15:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Meet",
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        spans = detail["conflicting_spans"]
        assert len(spans) == 2
        labels = {s["label"] for s in spans}
        assert labels == {"Build", "Debug"}


# ---------------------------------------------------------------------------
# label_created event type  (Item 14)
# ---------------------------------------------------------------------------


class TestLabelCreatedEvent:
    """Verify extend_forward publishes label_created (not prediction)."""

    def test_extend_forward_publishes_label_created(self, data_dir: Path) -> None:
        """TC-UI-LC-001: extend_forward=True emits label_created via WS."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                import time

                time.sleep(0.05)
                tc.post(
                    "/api/labels",
                    json={
                        "start_ts": "2026-02-27T09:00:00",
                        "end_ts": "2026-02-27T10:00:00",
                        "label": "Build",
                        "extend_forward": True,
                    },
                )

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "label_created"
            assert received["label"] == "Build"
            assert received["extend_forward"] is True
            assert received["start_ts"] == "2026-02-27T09:00:00+00:00"
            assert received["ts"] == "2026-02-27T10:00:00+00:00"
            assert "provenance" not in received
            assert "mapped_label" not in received
            changed = ws.receive_json()
            assert changed["type"] == "labels_changed"
            assert changed["reason"] == "created"

    def test_no_event_without_extend_forward(self, data_dir: Path) -> None:
        """TC-UI-LC-002: extend_forward=False does not emit label_created."""
        import threading
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                time.sleep(0.05)
                tc.post(
                    "/api/labels",
                    json={
                        "start_ts": "2026-02-27T09:00:00",
                        "end_ts": "2026-02-27T10:00:00",
                        "label": "Build",
                        "extend_forward": False,
                    },
                )
                time.sleep(0.1)
                bus.publish_threadsafe({"type": "status", "state": "sentinel"})

            threading.Thread(target=_post).start()
            changed = ws.receive_json()
            assert changed["type"] == "labels_changed"
            assert changed["reason"] == "created"
            received = ws.receive_json()
            assert received["type"] == "status"
            assert received["state"] == "sentinel"


# ---------------------------------------------------------------------------
# label_stopped event type
# ---------------------------------------------------------------------------


class TestLabelStoppedEvent:
    """Verify stopping an open-ended label publishes label_stopped."""

    def test_stop_current_label_publishes_label_stopped(self, data_dir: Path) -> None:
        import threading
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc:
            create_resp = tc.post(
                "/api/labels",
                json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T09:00:00",
                    "label": "Build",
                    "extend_forward": True,
                },
            )
            assert create_resp.status_code == 201

            with tc.websocket_connect("/ws/predictions") as ws:

                def _put() -> None:
                    time.sleep(0.05)
                    tc.put(
                        "/api/labels",
                        json={
                            "start_ts": "2026-02-27T09:00:00",
                            "end_ts": "2026-02-27T09:00:00",
                            "label": "Build",
                            "new_end_ts": "2026-02-27T09:30:00",
                            "extend_forward": False,
                        },
                    )

                threading.Thread(target=_put).start()
                received = ws.receive_json()
                assert received["type"] == "label_stopped"
                assert received["ts"] == "2026-02-27T09:30:00+00:00"


# ---------------------------------------------------------------------------
# suggestion_cleared event  (Item 8)
# ---------------------------------------------------------------------------


class TestSuggestionCleared:
    """Verify suggestion_cleared rules: accept/skip; manual POST /api/labels does not clear."""

    def test_manual_label_save_does_not_publish_suggestion_cleared(
        self, data_dir: Path
    ) -> None:
        """TC-UI-SC-001: POST /api/labels does not emit suggestion_cleared (preserves banner)."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                import time

                time.sleep(0.05)
                tc.post(
                    "/api/labels",
                    json={
                        "start_ts": "2026-02-27T09:00:00",
                        "end_ts": "2026-02-27T10:00:00",
                        "label": "Build",
                    },
                )

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "labels_changed"
            assert received["reason"] == "created"

    def test_notification_accept_publishes_suggestion_cleared(
        self, data_dir: Path
    ) -> None:
        """TC-UI-SC-002: POST /api/notification/accept emits suggestion_cleared."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                import time

                time.sleep(0.05)
                tc.post(
                    "/api/notification/accept",
                    json={
                        "block_start": "2026-02-27T09:00:00",
                        "block_end": "2026-02-27T10:00:00",
                        "label": "Build",
                    },
                )

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "suggestion_cleared"
            assert received["reason"] == "label_saved"

    def test_validation_error_no_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-003: Invalid label does not emit suggestion_cleared."""
        import threading
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                time.sleep(0.05)
                tc.post(
                    "/api/labels",
                    json={
                        "start_ts": "2026-02-27T09:00:00",
                        "end_ts": "2026-02-27T10:00:00",
                        "label": "NotALabel",
                    },
                )
                time.sleep(0.1)
                bus.publish_threadsafe({"type": "status", "state": "sentinel"})

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "status"
            assert received["state"] == "sentinel"

    def test_overlap_error_no_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-004: Overlap error does not emit suggestion_cleared."""
        import threading
        import time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc:
            tc.post(
                "/api/labels",
                json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T10:00:00",
                    "label": "Build",
                },
            )

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:

            def _post() -> None:
                time.sleep(0.05)
                tc.post(
                    "/api/labels",
                    json={
                        "start_ts": "2026-02-27T09:30:00",
                        "end_ts": "2026-02-27T10:30:00",
                        "label": "Meet",
                    },
                )
                time.sleep(0.1)
                bus.publish_threadsafe({"type": "status", "state": "sentinel"})

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "status"
            assert received["state"] == "sentinel"


# ---------------------------------------------------------------------------
# UTC helper functions  (Item 17)
# ---------------------------------------------------------------------------


class TestUtcHelpers:
    """Direct unit tests for _utc_iso() and _ensure_utc()."""

    def test_utc_iso_naive_appends_offset(self) -> None:
        from taskclf.ui.server import _utc_iso

        naive = dt.datetime(2026, 3, 1, 12, 0, 0)
        assert _utc_iso(naive) == "2026-03-01T12:00:00+00:00"

    def test_utc_iso_aware_utc_passthrough(self) -> None:
        from taskclf.ui.server import _utc_iso

        aware = dt.datetime(2026, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        assert _utc_iso(aware) == "2026-03-01T12:00:00+00:00"

    def test_utc_iso_aware_non_utc_converts(self) -> None:
        from taskclf.ui.server import _utc_iso

        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        aware = dt.datetime(2026, 3, 1, 17, 30, 0, tzinfo=ist)
        assert _utc_iso(aware) == "2026-03-01T12:00:00+00:00"

    def test_ensure_utc_naive_tags_utc(self) -> None:
        from taskclf.ui.server import _ensure_utc

        naive = dt.datetime(2026, 3, 1, 12, 0, 0)
        result = _ensure_utc(naive)
        assert result == dt.datetime(2026, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        assert result.tzinfo is not None

    def test_ensure_utc_aware_utc_passthrough(self) -> None:
        from taskclf.ui.server import _ensure_utc

        aware = dt.datetime(2026, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        result = _ensure_utc(aware)
        assert result == aware
        assert result.tzinfo == dt.timezone.utc

    def test_ensure_utc_aware_non_utc_converts(self) -> None:
        from taskclf.ui.server import _ensure_utc

        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        aware = dt.datetime(2026, 3, 1, 17, 30, 0, tzinfo=ist)
        result = _ensure_utc(aware)
        assert result == dt.datetime(2026, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        assert result.tzinfo == dt.timezone.utc


# ---------------------------------------------------------------------------
# Aware timestamp round-trip  (Item 17)
# ---------------------------------------------------------------------------


class TestAwareTimestampRoundTrip:
    """Verify the server accepts aware timestamps from the frontend and
    returns +00:00 suffixed ISO strings in all responses."""

    def test_z_suffixed_timestamps_accepted(self, client: TestClient) -> None:
        """Frontend sends Z-suffixed ISO strings after the item-17 fix."""
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00Z",
                "end_ts": "2026-02-27T10:00:00Z",
                "label": "Build",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_z_suffixed_persisted_correctly(self, client: TestClient) -> None:
        """Labels created with Z-suffixed timestamps appear in GET /api/labels."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00Z",
                "end_ts": "2026-02-27T10:00:00Z",
                "label": "Build",
            },
        )
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert labels[0]["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_offset_timestamps_normalized(self, client: TestClient) -> None:
        """Non-UTC aware timestamps are converted to UTC for storage."""
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T14:30:00+05:30",
                "end_ts": "2026-02-27T15:30:00+05:30",
                "label": "Build",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_naive_timestamps_still_accepted(self, client: TestClient) -> None:
        """Backward compat: naive timestamps (no tz suffix) still work."""
        resp = client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_update_with_aware_timestamps(self, client: TestClient) -> None:
        """PUT /api/labels accepts aware timestamps for matching."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.put(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00+00:00",
                "end_ts": "2026-02-27T10:00:00+00:00",
                "label": "Debug",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "Debug"

    def test_delete_with_aware_timestamps(self, client: TestClient) -> None:
        """DELETE /api/labels accepts aware timestamps for matching."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            },
        )
        resp = client.request(
            "DELETE",
            "/api/labels",
            json={
                "start_ts": "2026-02-27T09:00:00+00:00",
                "end_ts": "2026-02-27T10:00:00+00:00",
            },
        )
        assert resp.status_code == 200
        assert client.get("/api/labels").json() == []

    def test_range_filter_handles_aware_spans_from_storage(
        self, client: TestClient, data_dir: Path
    ) -> None:
        """GET /api/labels range filters must tolerate aware legacy spans on disk."""
        labels_path = data_dir / "labels_v1" / "labels.parquet"
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        write_label_spans(
            [
                LabelSpan(
                    start_ts=dt.datetime(2026, 2, 27, 14, 30, tzinfo=ist),
                    end_ts=dt.datetime(2026, 2, 27, 15, 30, tzinfo=ist),
                    label="Build",
                    provenance="manual",
                )
            ],
            labels_path,
        )

        resp = client.get(
            "/api/labels",
            params={
                "range_start": "2026-02-27T09:15:00Z",
                "range_end": "2026-02-27T09:45:00Z",
            },
        )

        assert resp.status_code == 200
        labels = resp.json()
        assert len(labels) == 1
        assert labels[0]["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert labels[0]["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_date_filter_works_with_aware_spans(
        self, client: TestClient, data_dir: Path
    ) -> None:
        """GET /api/labels?date= must work when stored spans are aware UTC."""
        labels_path = data_dir / "labels_v1" / "labels.parquet"
        write_label_spans(
            [
                LabelSpan(
                    start_ts=dt.datetime(2026, 4, 10, 9, 0, tzinfo=dt.timezone.utc),
                    end_ts=dt.datetime(2026, 4, 10, 10, 0, tzinfo=dt.timezone.utc),
                    label="Build",
                    provenance="manual",
                ),
                LabelSpan(
                    start_ts=dt.datetime(2026, 4, 11, 14, 0, tzinfo=dt.timezone.utc),
                    end_ts=dt.datetime(2026, 4, 11, 15, 0, tzinfo=dt.timezone.utc),
                    label="Write",
                    provenance="manual",
                ),
            ],
            labels_path,
        )

        resp = client.get("/api/labels", params={"date": "2026-04-10"})
        assert resp.status_code == 200
        labels = resp.json()
        assert len(labels) == 1
        assert labels[0]["label"] == "Build"

    def test_date_filter_non_utc_spans_normalized(
        self, client: TestClient, data_dir: Path
    ) -> None:
        """Spans stored from a non-UTC timezone are normalized and filtered by UTC date."""
        labels_path = data_dir / "labels_v1" / "labels.parquet"
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        write_label_spans(
            [
                LabelSpan(
                    start_ts=dt.datetime(2026, 4, 11, 2, 0, tzinfo=ist),
                    end_ts=dt.datetime(2026, 4, 11, 3, 0, tzinfo=ist),
                    label="Debug",
                    provenance="manual",
                ),
            ],
            labels_path,
        )

        resp_apr10 = client.get("/api/labels", params={"date": "2026-04-10"})
        assert resp_apr10.status_code == 200
        labels_apr10 = resp_apr10.json()
        assert len(labels_apr10) == 1
        assert labels_apr10[0]["label"] == "Debug"
        assert labels_apr10[0]["start_ts"] == "2026-04-10T20:30:00+00:00"

        resp_apr11 = client.get("/api/labels", params={"date": "2026-04-11"})
        assert resp_apr11.status_code == 200
        assert len(resp_apr11.json()) == 0


# ---------------------------------------------------------------------------
# Label Stats endpoint
# ---------------------------------------------------------------------------


class TestLabelStats:
    def test_no_labels_file(self, client: TestClient) -> None:
        resp = client.get("/api/labels/stats", params={"date": "2026-03-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2026-03-01"
        assert data["count"] == 0
        assert data["total_minutes"] == 0.0
        assert data["breakdown"] == {}

    def test_stats_with_labels(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T09:00:00",
                "end_ts": "2026-03-01T09:45:00",
                "label": "Build",
            },
        )
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T10:00:00",
                "end_ts": "2026-03-01T10:20:00",
                "label": "Meet",
            },
        )

        resp = client.get("/api/labels/stats", params={"date": "2026-03-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2026-03-01"
        assert data["count"] == 2
        assert data["total_minutes"] == 65.0
        assert data["breakdown"]["Build"] == 45.0
        assert data["breakdown"]["Meet"] == 20.0

    def test_stats_filters_by_date(self, client: TestClient) -> None:
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T09:00:00",
                "end_ts": "2026-03-01T09:30:00",
                "label": "Build",
            },
        )

        resp = client.get("/api/labels/stats", params={"date": "2026-03-02"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["total_minutes"] == 0.0

    def test_stats_defaults_to_today(self, client: TestClient) -> None:
        resp = client.get("/api/labels/stats")
        assert resp.status_code == 200
        data = resp.json()
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        assert data["date"] == today

    def test_stats_with_aware_utc_timestamps(self, client: TestClient) -> None:
        """Stats endpoint correctly filters and sums aware-UTC label spans."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-05-15T08:00:00+00:00",
                "end_ts": "2026-05-15T09:00:00+00:00",
                "label": "Build",
            },
        )
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-05-15T10:00:00Z",
                "end_ts": "2026-05-15T10:30:00Z",
                "label": "Debug",
            },
        )

        resp = client.get("/api/labels/stats", params={"date": "2026-05-15"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2026-05-15"
        assert data["count"] == 2
        assert data["total_minutes"] == 90.0
        assert data["breakdown"]["Build"] == 60.0
        assert data["breakdown"]["Debug"] == 30.0

    def test_stats_non_utc_input_converted(self, client: TestClient) -> None:
        """Labels created with non-UTC offset timestamps still land on the correct UTC date."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-06-01T02:00:00+05:30",
                "end_ts": "2026-06-01T03:00:00+05:30",
                "label": "Write",
            },
        )

        resp_may31 = client.get("/api/labels/stats", params={"date": "2026-05-31"})
        assert resp_may31.status_code == 200
        assert resp_may31.json()["count"] == 1

        resp_jun01 = client.get("/api/labels/stats", params={"date": "2026-06-01"})
        assert resp_jun01.status_code == 200
        assert resp_jun01.json()["count"] == 0


# ---------------------------------------------------------------------------
# POST /api/labels/import  (Item 2)
# ---------------------------------------------------------------------------


def _make_csv_bytes(rows: list[dict]) -> bytes:
    """Build CSV bytes from a list of dicts for upload testing."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


_IMPORT_ROWS = [
    {
        "start_ts": "2026-03-01 09:00:00",
        "end_ts": "2026-03-01 09:30:00",
        "label": "Build",
        "provenance": "manual",
    },
    {
        "start_ts": "2026-03-01 10:00:00",
        "end_ts": "2026-03-01 10:30:00",
        "label": "Meet",
        "provenance": "manual",
    },
]


class TestImportLabels:
    """Tests for POST /api/labels/import endpoint."""

    def test_import_merge_happy_path(self, client: TestClient) -> None:
        """Merge preserves existing labels and adds new ones."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T08:00:00",
                "end_ts": "2026-03-01T08:30:00",
                "label": "Write",
            },
        )

        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
            data={"strategy": "merge"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["imported"] == 2
        assert data["total"] == 3
        assert data["strategy"] == "merge"

        labels = client.get("/api/labels").json()
        assert len(labels) == 3

    def test_import_overwrite_happy_path(self, client: TestClient) -> None:
        """Overwrite replaces all existing labels with the imported ones."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T08:00:00",
                "end_ts": "2026-03-01T08:30:00",
                "label": "Write",
            },
        )

        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
            data={"strategy": "overwrite"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["imported"] == 2
        assert data["total"] == 2
        assert data["strategy"] == "overwrite"

        labels = client.get("/api/labels").json()
        assert len(labels) == 2
        label_names = {rec["label"] for rec in labels}
        assert "Write" not in label_names

    def test_import_merge_overlap_409(self, client: TestClient) -> None:
        """Merge fails with 409 when imported spans overlap existing ones."""
        client.post(
            "/api/labels",
            json={
                "start_ts": "2026-03-01T09:10:00",
                "end_ts": "2026-03-01T09:40:00",
                "label": "Debug",
            },
        )

        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
            data={"strategy": "merge"},
        )
        assert resp.status_code == 409

    def test_import_merge_deduplicates(self, client: TestClient) -> None:
        """Duplicate spans (same start_ts, end_ts, user_id) are deduplicated."""
        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
            data={"strategy": "merge"},
        )

        csv_bytes_2 = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels2.csv", csv_bytes_2, "text/csv")},
            data={"strategy": "merge"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

        labels = client.get("/api/labels").json()
        assert len(labels) == 2

    def test_import_invalid_csv_422(self, client: TestClient) -> None:
        """CSV with missing required columns returns 422."""
        bad_csv = b"col_a,col_b\n1,2\n"
        resp = client.post(
            "/api/labels/import",
            files={"file": ("bad.csv", bad_csv, "text/csv")},
            data={"strategy": "merge"},
        )
        assert resp.status_code == 422

    def test_import_invalid_strategy_422(self, client: TestClient) -> None:
        """Invalid strategy value returns 422."""
        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
            data={"strategy": "invalid"},
        )
        assert resp.status_code == 422

    def test_import_default_strategy_is_merge(self, client: TestClient) -> None:
        """Omitting strategy defaults to merge."""
        csv_bytes = _make_csv_bytes(_IMPORT_ROWS)
        resp = client.post(
            "/api/labels/import",
            files={"file": ("labels.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "merge"


class TestModelBundleInspectAPI:
    """Bundle-only model inspection endpoints for the tray UI."""

    def test_current_inspect_no_tray_returns_unavailable(
        self, client: TestClient
    ) -> None:
        """Without get_tray_state, current inspect reports loaded=false."""
        resp = client.get("/api/train/models/current/inspect")
        assert resp.status_code == 200
        assert resp.json() == {
            "loaded": False,
            "reason": "tray_state_unavailable",
        }

    def test_current_inspect_no_model_loaded(self, data_dir: Path) -> None:
        """Tray state without model_dir returns no_model_loaded."""
        app = create_app(
            data_dir=data_dir,
            event_bus=EventBus(),
            get_tray_state=lambda: {"model_dir": None, "paused": False},
        )
        tc = TestClient(app)
        resp = tc.get("/api/train/models/current/inspect")
        assert resp.status_code == 200
        assert resp.json() == {"loaded": False, "reason": "no_model_loaded"}

    def test_inspect_by_id_and_current_with_bundle(self, tmp_path: Path) -> None:
        """Current and by-id inspect return bundle_saved_validation from disk."""
        from typer.testing import CliRunner

        from taskclf.cli.main import app as cli_app

        data_dir = tmp_path / "pdata"
        data_dir.mkdir()
        (data_dir / "labels_v1").mkdir()
        models_dir = tmp_path / "models"
        runner = CliRunner()
        result = runner.invoke(
            cli_app,
            [
                "train",
                "lgbm",
                "--from",
                "2025-06-14",
                "--to",
                "2025-06-15",
                "--synthetic",
                "--models-dir",
                str(models_dir),
                "--num-boost-round",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        run_dir = next(models_dir.iterdir())
        run_resolved = str(run_dir.resolve())

        app = create_app(
            data_dir=data_dir,
            models_dir=models_dir,
            event_bus=EventBus(),
            get_tray_state=lambda: {"model_dir": run_resolved, "paused": False},
        )
        tc = TestClient(app)

        cur = tc.get("/api/train/models/current/inspect")
        assert cur.status_code == 200
        cj = cur.json()
        assert cj["loaded"] is True
        assert cj["bundle_path"] == run_resolved
        assert "macro_f1" in cj["bundle_saved_validation"]
        assert "metadata" in cj

        by_id = tc.get(f"/api/train/models/{run_dir.name}/inspect")
        assert by_id.status_code == 200
        bj = by_id.json()
        assert bj["bundle_path"] == run_resolved
        assert (
            bj["bundle_saved_validation"]["macro_f1"]
            == cj["bundle_saved_validation"]["macro_f1"]
        )

        missing = tc.get("/api/train/models/does_not_exist_run/inspect")
        assert missing.status_code == 404


class TestDevServerTransitions:
    def test_suggestion_transition_publishes_prompt_and_suggest_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        captured: list[dict[str, object]] = []
        transition_cb: Callable[[str, str, dt.datetime, dt.datetime], None] | None = (
            None
        )

        class FakeBus:
            def publish_threadsafe(self, event: dict[str, object]) -> None:
                captured.append(event)

        class FakeMonitor:
            def __init__(
                self,
                *,
                on_transition: Callable[[str, str, dt.datetime, dt.datetime], None]
                | None = None,
                **_: object,
            ) -> None:
                nonlocal transition_cb
                transition_cb = on_transition

            def run(self) -> None:
                return None

            def stop(self) -> None:
                return None

        class FakeSuggester:
            def __init__(self, _path: Path) -> None:
                self._aw_host = ""
                self._title_salt = ""

            def suggest(
                self,
                _start: dt.datetime,
                _end: dt.datetime,
            ) -> tuple[str, float]:
                return ("Build", 0.85)

        monkeypatch.setenv("TASKCLF_UI_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TASKCLF_MODEL_DIR", str(tmp_path / "model"))
        monkeypatch.setattr("taskclf.ui.server.EventBus", FakeBus)
        monkeypatch.setattr("taskclf.ui.runtime.ActivityMonitor", FakeMonitor)
        monkeypatch.setattr("taskclf.ui.runtime._LabelSuggester", FakeSuggester)
        monkeypatch.setattr("taskclf.ui.server.create_app", lambda **_: FastAPI())

        _create_dev_app_from_env()

        assert transition_cb is not None
        start = dt.datetime(2026, 4, 5, 9, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 4, 5, 9, 5, tzinfo=dt.timezone.utc)

        transition_cb("Editor", "Browser", start, end)

        assert [event["type"] for event in captured] == [
            "prompt_label",
            "suggest_label",
        ]

        prompt = captured[0]
        assert prompt["suggested_label"] == "Build"
        assert prompt["block_start"] == start.isoformat()
        assert prompt["block_end"] == end.isoformat()
        assert prompt["suggestion_text"] == transition_suggestion_text(
            "Build",
            start.astimezone().strftime("%H:%M"),
            end.astimezone().strftime("%H:%M"),
        )

        suggest = captured[1]
        assert suggest["reason"] == "app_switch"
        assert suggest["old_label"] == "Editor"
        assert suggest["suggested"] == "Build"
        assert suggest["confidence"] == 0.85
