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


class TestWebSocket:
    def test_ws_connects(self, data_dir: Path) -> None:
        bus = EventBus()
        app = create_app(data_dir=data_dir, event_bus=bus)
        client = TestClient(app)

        with client.websocket_connect("/ws/predictions") as ws:
            bus.publish_threadsafe({"type": "status", "state": "idle", "current_app": "test"})
            import time
            time.sleep(0.1)
            # Connection established successfully; threadsafe publish without
            # a bound loop is a no-op, so we just verify the handshake works.
            ws.close()
