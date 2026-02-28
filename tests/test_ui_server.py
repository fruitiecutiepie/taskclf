"""Tests for the FastAPI labeling web UI server.

Covers REST endpoints for labels, queue, config, and feature summary.
WebSocket broadcast is tested via the EventBus integration.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from taskclf.core.types import LabelSpan
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
        assert build["end_ts"] == "2026-02-27T10:29:00"

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
        assert build["end_ts"] == "2026-02-27T10:00:00"

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
        assert build["end_ts"] == "2026-02-27T10:29:00"

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
        assert build["end_ts"] == "2026-02-27T10:00:00"

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
        assert data["start_ts"] == "2026-02-27T10:29:00"
        assert data["end_ts"] == "2026-02-27T10:30:00"
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
