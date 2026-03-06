"""Tests for the FastAPI labeling web UI server.

Covers REST endpoints for labels, queue, config, and feature summary.
WebSocket broadcast is tested via the EventBus integration.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from taskclf.core.types import LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue, LabelRequest
from taskclf.labels.store import append_label_span
from taskclf.ui.events import EventBus
from taskclf.ui.server import create_app


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
        assert labels[0]["start_ts"] > labels[1]["start_ts"]


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
        build = [l for l in labels if l["label"] == "Build"][0]
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
        build = [l for l in labels if l["label"] == "Build"][0]
        assert build["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_extend_forward_flag_on_previous_not_request(self, client: TestClient) -> None:
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
        build = [l for l in labels if l["label"] == "Build"][0]
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
        build = [l for l in labels if l["label"] == "Build"][0]
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


class TestWebSocket:
    def test_ws_connects(self, data_dir: Path) -> None:
        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        client = TestClient(app)

        with client.websocket_connect("/ws/predictions") as ws:
            bus.publish_threadsafe({"type": "status", "state": "idle", "current_app": "test"})
            import time
            time.sleep(0.1)
            ws.close()

    def test_event_delivery_roundtrip(self, data_dir: Path) -> None:
        """TC-UI-WS-003: publish via EventBus, read from WS, verify match."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        event = {"type": "status", "state": "collecting", "current_app": "Terminal"}

        with TestClient(app) as client, client.websocket_connect("/ws/predictions") as ws:
            threading.Thread(
                target=lambda: (
                    __import__("time").sleep(0.05),
                    bus.publish_threadsafe(event),
                ),
            ).start()
            received = ws.receive_json()
            assert received == event

    def test_multiple_event_types_in_order(self, data_dir: Path) -> None:
        """TC-UI-WS-004: multiple event types received in publish order."""
        import threading, time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        events = [
            {"type": "status", "state": "idle"},
            {"type": "prediction", "label": "Build", "confidence": 0.9},
            {"type": "tray_state", "model_loaded": True},
            {"type": "suggest_label", "reason": "app_switch"},
        ]

        def _publish_all() -> None:
            time.sleep(0.05)
            for ev in events:
                bus.publish_threadsafe(ev)

        with TestClient(app) as client, client.websocket_connect("/ws/predictions") as ws:
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
                threading.Thread(
                    target=lambda: (
                        __import__("time").sleep(0.05),
                        bus.publish_threadsafe(event),
                    ),
                ).start()
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


# ---------------------------------------------------------------------------
# PUT /api/labels  (Item 34)
# ---------------------------------------------------------------------------


class TestUpdateLabel:
    """TC-UI-UPD-001 through TC-UI-UPD-004."""

    def _create_label(self, client: TestClient) -> None:
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })

    def test_happy_path_update(self, client: TestClient) -> None:
        """TC-UI-UPD-001: POST then PUT with new label -> 200, label changed."""
        self._create_label(client)
        resp = client.put("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Write",
        })
        assert resp.status_code == 200
        assert resp.json()["label"] == "Write"

    def test_404_no_matching_span(self, client: TestClient) -> None:
        """TC-UI-UPD-002: PUT with non-existent timestamps -> 404."""
        self._create_label(client)
        resp = client.put("/api/labels", json={
            "start_ts": "2026-02-27T11:00:00",
            "end_ts": "2026-02-27T12:00:00",
            "label": "Write",
        })
        assert resp.status_code == 404

    def test_422_invalid_timestamp(self, client: TestClient) -> None:
        """TC-UI-UPD-003: Malformed start_ts -> 422."""
        resp = client.put("/api/labels", json={
            "start_ts": "not-a-date",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Write",
        })
        assert resp.status_code == 422

    def test_updated_label_persisted(self, client: TestClient) -> None:
        """TC-UI-UPD-004: PUT then GET -> updated label visible."""
        self._create_label(client)
        client.put("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Debug",
        })
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["label"] == "Debug"


# ---------------------------------------------------------------------------
# DELETE /api/labels  (Item 35)
# ---------------------------------------------------------------------------


class TestDeleteLabel:
    """TC-UI-DEL-001 through TC-UI-DEL-004."""

    def _create_label(self, client: TestClient, start: str, end: str, label: str = "Build") -> None:
        resp = client.post("/api/labels", json={
            "start_ts": start,
            "end_ts": end,
            "label": label,
        })
        assert resp.status_code == 201

    def test_happy_path_delete(self, client: TestClient) -> None:
        """TC-UI-DEL-001: POST then DELETE -> 200, status deleted."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        resp = client.request("DELETE", "/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
        })
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}

    def test_404_no_matching_span(self, client: TestClient) -> None:
        """TC-UI-DEL-002: DELETE with non-existent timestamps -> 404."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        resp = client.request("DELETE", "/api/labels", json={
            "start_ts": "2026-02-27T11:00:00",
            "end_ts": "2026-02-27T12:00:00",
        })
        assert resp.status_code == 404

    def test_span_removed_from_storage(self, client: TestClient) -> None:
        """TC-UI-DEL-003: POST, DELETE, then GET -> empty list."""
        self._create_label(client, "2026-02-27T09:00:00", "2026-02-27T10:00:00")
        client.request("DELETE", "/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
        })
        labels = client.get("/api/labels").json()
        assert labels == []

    def test_422_invalid_timestamp(self, client: TestClient) -> None:
        """TC-UI-DEL-004: Malformed timestamps -> 422."""
        resp = client.request("DELETE", "/api/labels", json={
            "start_ts": "garbage",
            "end_ts": "2026-02-27T10:00:00",
        })
        assert resp.status_code == 422


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
        """TC-UI-CU-001: Returns user_id (UUID) and username."""
        resp = client.get("/api/config/user")
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        assert "username" in data
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


# ---------------------------------------------------------------------------
# POST /api/notification/accept  (Item 39)
# ---------------------------------------------------------------------------


class TestNotificationAccept:
    """TC-UI-NA-001 through TC-UI-NA-005."""

    def test_accept_valid_suggestion(self, client: TestClient) -> None:
        """TC-UI-NA-001: Accept -> 200, provenance is 'suggestion'."""
        resp = client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:00:00",
            "block_end": "2026-02-27T10:00:00",
            "label": "Build",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["provenance"] == "suggestion"
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_invalid_label_422(self, client: TestClient) -> None:
        """TC-UI-NA-002: Invalid label -> 422."""
        resp = client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:00:00",
            "block_end": "2026-02-27T10:00:00",
            "label": "NotReal",
        })
        assert resp.status_code == 422

    def test_invalid_timestamps_422(self, client: TestClient) -> None:
        """TC-UI-NA-003: Malformed ISO timestamps -> 422."""
        resp = client.post("/api/notification/accept", json={
            "block_start": "not-a-date",
            "block_end": "2026-02-27T10:00:00",
            "label": "Build",
        })
        assert resp.status_code == 422

    def test_overlap_409(self, client: TestClient) -> None:
        """TC-UI-NA-004: Overlap with existing span -> 409."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:30:00",
            "block_end": "2026-02-27T10:30:00",
            "label": "Write",
        })
        assert resp.status_code == 409

    def test_label_persisted(self, client: TestClient) -> None:
        """TC-UI-NA-005: Accept, then GET -> label visible."""
        client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:00:00",
            "block_end": "2026-02-27T10:00:00",
            "label": "Build",
        })
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
        """TC-UI-WS-002: With window_api -> show_label_grid() called."""
        from unittest.mock import MagicMock

        mock_api = MagicMock()
        mock_api.visible = True
        app = create_app(data_dir=data_dir, event_bus=EventBus(), window_api=mock_api)
        tc = TestClient(app)

        resp = tc.post("/api/window/show-label-grid")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_api.show_label_grid.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/labels — limit parameter  (Item 42)
# ---------------------------------------------------------------------------


class TestLabelsLimit:
    """TC-UI-LL-001 through TC-UI-LL-004."""

    def _seed_labels(self, client: TestClient, count: int = 3) -> None:
        for i in range(count):
            client.post("/api/labels", json={
                "start_ts": f"2026-02-27T{8 + i:02d}:00:00",
                "end_ts": f"2026-02-27T{8 + i:02d}:30:00",
                "label": "Build",
            })

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

        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        cb.assert_called_once()

    def test_callback_invoked_on_notification_accept(self, data_dir: Path) -> None:
        """TC-UI-OLS-002: POST /api/notification/accept calls on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:00:00",
            "block_end": "2026-02-27T10:00:00",
            "label": "Build",
        })
        cb.assert_called_once()

    def test_callback_not_invoked_on_validation_error(self, data_dir: Path) -> None:
        """TC-UI-OLS-003: 422 error does not call on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "NotALabel",
        })
        cb.assert_not_called()

    def test_callback_not_invoked_on_overlap_error(self, data_dir: Path) -> None:
        """TC-UI-OLS-004: 409 overlap does not call on_label_saved."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        app = create_app(data_dir=data_dir, event_bus=EventBus(), on_label_saved=cb)
        client = TestClient(app)

        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        cb.reset_mock()

        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:30:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "Meet",
        })
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
            data_dir=data_dir, event_bus=EventBus(),
            pause_toggle=toggle, is_paused=is_paused,
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
            data_dir=data_dir, event_bus=EventBus(),
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
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:30:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "Meet",
        })
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "error" in detail
        assert detail["conflicting_start_ts"] is not None
        assert detail["conflicting_end_ts"] is not None
        assert "2026-02-27" in detail["conflicting_start_ts"]
        assert "2026-02-27" in detail["conflicting_end_ts"]

    def test_overlap_conflicting_span_is_existing(self, client: TestClient) -> None:
        """TC-UI-OVL-002: the conflicting span points to the previously saved label."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:30:00",
            "end_ts": "2026-02-27T10:30:00",
            "label": "Meet",
        })
        detail = resp.json()["detail"]
        assert "09:00:00" in detail["conflicting_start_ts"]
        assert "10:00:00" in detail["conflicting_end_ts"]

    def test_notification_accept_overlap_structured(self, client: TestClient) -> None:
        """TC-UI-OVL-003: POST /api/notification/accept overlap is also structured."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.post("/api/notification/accept", json={
            "block_start": "2026-02-27T09:30:00",
            "block_end": "2026-02-27T10:30:00",
            "label": "Write",
        })
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "error" in detail
        assert detail["conflicting_start_ts"] is not None
        assert detail["conflicting_end_ts"] is not None


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
                tc.post("/api/labels", json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T10:00:00",
                    "label": "Build",
                    "extend_forward": True,
                })

            threading.Thread(target=_post).start()
            cleared = ws.receive_json()
            assert cleared["type"] == "suggestion_cleared"
            received = ws.receive_json()
            assert received["type"] == "label_created"
            assert received["label"] == "Build"
            assert received["extend_forward"] is True
            assert received["start_ts"] == "2026-02-27T09:00:00+00:00"
            assert received["ts"] == "2026-02-27T10:00:00+00:00"
            assert "provenance" not in received
            assert "mapped_label" not in received

    def test_no_event_without_extend_forward(self, data_dir: Path) -> None:
        """TC-UI-LC-002: extend_forward=False does not emit label_created."""
        import threading, time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:
            def _post() -> None:
                time.sleep(0.05)
                tc.post("/api/labels", json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T10:00:00",
                    "label": "Build",
                    "extend_forward": False,
                })
                time.sleep(0.1)
                bus.publish_threadsafe({"type": "status", "state": "sentinel"})

            threading.Thread(target=_post).start()
            cleared = ws.receive_json()
            assert cleared["type"] == "suggestion_cleared"
            received = ws.receive_json()
            assert received["type"] == "status"
            assert received["state"] == "sentinel"


# ---------------------------------------------------------------------------
# suggestion_cleared event  (Item 8)
# ---------------------------------------------------------------------------


class TestSuggestionCleared:
    """Verify label saves publish suggestion_cleared, errors do not."""

    def test_label_save_publishes_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-001: POST /api/labels emits suggestion_cleared via WS."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:
            def _post() -> None:
                import time
                time.sleep(0.05)
                tc.post("/api/labels", json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T10:00:00",
                    "label": "Build",
                })

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "suggestion_cleared"
            assert received["reason"] == "label_saved"

    def test_notification_accept_publishes_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-002: POST /api/notification/accept emits suggestion_cleared."""
        import threading

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:
            def _post() -> None:
                import time
                time.sleep(0.05)
                tc.post("/api/notification/accept", json={
                    "block_start": "2026-02-27T09:00:00",
                    "block_end": "2026-02-27T10:00:00",
                    "label": "Build",
                })

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "suggestion_cleared"
            assert received["reason"] == "label_saved"

    def test_validation_error_no_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-003: Invalid label does not emit suggestion_cleared."""
        import threading, time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:
            def _post() -> None:
                time.sleep(0.05)
                tc.post("/api/labels", json={
                    "start_ts": "2026-02-27T09:00:00",
                    "end_ts": "2026-02-27T10:00:00",
                    "label": "NotALabel",
                })
                time.sleep(0.1)
                bus.publish_threadsafe({"type": "status", "state": "sentinel"})

            threading.Thread(target=_post).start()
            received = ws.receive_json()
            assert received["type"] == "status"
            assert received["state"] == "sentinel"

    def test_overlap_error_no_suggestion_cleared(self, data_dir: Path) -> None:
        """TC-UI-SC-004: Overlap error does not emit suggestion_cleared."""
        import threading, time

        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)

        with TestClient(app) as tc:
            tc.post("/api/labels", json={
                "start_ts": "2026-02-27T09:00:00",
                "end_ts": "2026-02-27T10:00:00",
                "label": "Build",
            })

        with TestClient(app) as tc, tc.websocket_connect("/ws/predictions") as ws:
            def _post() -> None:
                time.sleep(0.05)
                tc.post("/api/labels", json={
                    "start_ts": "2026-02-27T09:30:00",
                    "end_ts": "2026-02-27T10:30:00",
                    "label": "Meet",
                })
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
    """Direct unit tests for _utc_iso() and _to_naive_utc()."""

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

    def test_to_naive_utc_naive_passthrough(self) -> None:
        from taskclf.ui.server import _to_naive_utc

        naive = dt.datetime(2026, 3, 1, 12, 0, 0)
        result = _to_naive_utc(naive)
        assert result == naive
        assert result.tzinfo is None

    def test_to_naive_utc_aware_utc_strips(self) -> None:
        from taskclf.ui.server import _to_naive_utc

        aware = dt.datetime(2026, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        result = _to_naive_utc(aware)
        assert result == dt.datetime(2026, 3, 1, 12, 0, 0)
        assert result.tzinfo is None

    def test_to_naive_utc_aware_non_utc_converts_then_strips(self) -> None:
        from taskclf.ui.server import _to_naive_utc

        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        aware = dt.datetime(2026, 3, 1, 17, 30, 0, tzinfo=ist)
        result = _to_naive_utc(aware)
        assert result == dt.datetime(2026, 3, 1, 12, 0, 0)
        assert result.tzinfo is None


# ---------------------------------------------------------------------------
# Aware timestamp round-trip  (Item 17)
# ---------------------------------------------------------------------------


class TestAwareTimestampRoundTrip:
    """Verify the server accepts aware timestamps from the frontend and
    returns +00:00 suffixed ISO strings in all responses."""

    def test_z_suffixed_timestamps_accepted(self, client: TestClient) -> None:
        """Frontend sends Z-suffixed ISO strings after the item-17 fix."""
        resp = client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00Z",
            "end_ts": "2026-02-27T10:00:00Z",
            "label": "Build",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_z_suffixed_persisted_correctly(self, client: TestClient) -> None:
        """Labels created with Z-suffixed timestamps appear in GET /api/labels."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00Z",
            "end_ts": "2026-02-27T10:00:00Z",
            "label": "Build",
        })
        labels = client.get("/api/labels").json()
        assert len(labels) == 1
        assert labels[0]["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert labels[0]["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_offset_timestamps_normalized(self, client: TestClient) -> None:
        """Non-UTC aware timestamps are converted to UTC for storage."""
        resp = client.post("/api/labels", json={
            "start_ts": "2026-02-27T14:30:00+05:30",
            "end_ts": "2026-02-27T15:30:00+05:30",
            "label": "Build",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_naive_timestamps_still_accepted(self, client: TestClient) -> None:
        """Backward compat: naive timestamps (no tz suffix) still work."""
        resp = client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["start_ts"] == "2026-02-27T09:00:00+00:00"
        assert data["end_ts"] == "2026-02-27T10:00:00+00:00"

    def test_update_with_aware_timestamps(self, client: TestClient) -> None:
        """PUT /api/labels accepts aware timestamps for matching."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.put("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00+00:00",
            "end_ts": "2026-02-27T10:00:00+00:00",
            "label": "Debug",
        })
        assert resp.status_code == 200
        assert resp.json()["label"] == "Debug"

    def test_delete_with_aware_timestamps(self, client: TestClient) -> None:
        """DELETE /api/labels accepts aware timestamps for matching."""
        client.post("/api/labels", json={
            "start_ts": "2026-02-27T09:00:00",
            "end_ts": "2026-02-27T10:00:00",
            "label": "Build",
        })
        resp = client.request("DELETE", "/api/labels", json={
            "start_ts": "2026-02-27T09:00:00+00:00",
            "end_ts": "2026-02-27T10:00:00+00:00",
        })
        assert resp.status_code == 200
        assert client.get("/api/labels").json() == []


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
        client.post("/api/labels", json={
            "start_ts": "2026-03-01T09:00:00",
            "end_ts": "2026-03-01T09:45:00",
            "label": "Build",
        })
        client.post("/api/labels", json={
            "start_ts": "2026-03-01T10:00:00",
            "end_ts": "2026-03-01T10:20:00",
            "label": "Meet",
        })

        resp = client.get("/api/labels/stats", params={"date": "2026-03-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2026-03-01"
        assert data["count"] == 2
        assert data["total_minutes"] == 65.0
        assert data["breakdown"]["Build"] == 45.0
        assert data["breakdown"]["Meet"] == 20.0

    def test_stats_filters_by_date(self, client: TestClient) -> None:
        client.post("/api/labels", json={
            "start_ts": "2026-03-01T09:00:00",
            "end_ts": "2026-03-01T09:30:00",
            "label": "Build",
        })

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
