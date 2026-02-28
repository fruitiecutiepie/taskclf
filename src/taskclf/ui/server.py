"""FastAPI backend for the taskclf labeling web UI.

Provides REST endpoints for label CRUD, queue management, and feature
summaries, plus a WebSocket channel for live prediction streaming.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from taskclf.core.config import UserConfig
from taskclf.core.defaults import DEFAULT_AW_HOST, DEFAULT_DATA_DIR, DEFAULT_TITLE_SALT
from taskclf.core.store import read_parquet
from taskclf.core.types import CoreLabel, LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue
from taskclf.labels.store import (
    append_label_span,
    generate_label_summary,
    read_label_spans,
)
from taskclf.ui.events import EventBus

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LabelCreateRequest(BaseModel):
    start_ts: str = Field(description="ISO-8601 start timestamp")
    end_ts: str = Field(description="ISO-8601 end timestamp")
    label: str
    user_id: str | None = None
    confidence: float | None = None
    extend_forward: bool = Field(
        default=False,
        description="When true, this label extends forward until the next "
        "label is created, producing contiguous coverage.",
    )


class LabelResponse(BaseModel):
    start_ts: str
    end_ts: str
    label: str
    provenance: str
    user_id: str | None = None
    confidence: float | None = None
    extend_forward: bool = False


class QueueItemResponse(BaseModel):
    request_id: str
    user_id: str
    bucket_start_ts: str
    bucket_end_ts: str
    reason: str
    confidence: float | None = None
    predicted_label: str | None = None
    status: str


class MarkDoneRequest(BaseModel):
    status: Literal["labeled", "skipped"] = "labeled"


class FeatureSummaryResponse(BaseModel):
    top_apps: list[dict[str, Any]]
    mean_keys_per_min: float | None
    mean_clicks_per_min: float | None
    mean_scroll_per_min: float | None
    total_buckets: int
    session_count: int


class AWLiveEntry(BaseModel):
    app: str
    events: int


class UserConfigResponse(BaseModel):
    user_id: str
    username: str


class UserConfigUpdateRequest(BaseModel):
    username: str | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    data_dir: Path = Path(DEFAULT_DATA_DIR),
    aw_host: str = DEFAULT_AW_HOST,
    title_salt: str = DEFAULT_TITLE_SALT,
    event_bus: EventBus | None = None,
    window_api: Any = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        data_dir: Path to the processed data directory.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        event_bus: Shared event bus for WebSocket broadcasting.
        window_api: Optional ``WindowAPI`` for pywebview window control.
    """
    bus = event_bus or EventBus()
    labels_path = data_dir / "labels_v1" / "labels.parquet"
    queue_path = data_dir / "labels_v1" / "queue.json"
    user_config = UserConfig(data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        bus.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(
        title="taskclf",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # -- REST: labels ---------------------------------------------------------

    @app.get("/api/labels")
    async def list_labels(limit: int = Query(50, ge=1, le=500)) -> list[LabelResponse]:
        if not labels_path.exists():
            return []
        spans = read_label_spans(labels_path)
        spans.sort(key=lambda s: s.start_ts, reverse=True)
        return [
            LabelResponse(
                start_ts=s.start_ts.isoformat(),
                end_ts=s.end_ts.isoformat(),
                label=s.label,
                provenance=s.provenance,
                user_id=s.user_id,
                confidence=s.confidence,
                extend_forward=s.extend_forward,
            )
            for s in spans[:limit]
        ]

    @app.post("/api/labels", status_code=201)
    async def create_label(body: LabelCreateRequest) -> LabelResponse:
        uid = body.user_id if body.user_id is not None else user_config.user_id
        try:
            span = LabelSpan(
                start_ts=dt.datetime.fromisoformat(body.start_ts),
                end_ts=dt.datetime.fromisoformat(body.end_ts),
                label=body.label,
                provenance="manual",
                user_id=uid,
                confidence=body.confidence,
                extend_forward=body.extend_forward,
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            append_label_span(span, labels_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return LabelResponse(
            start_ts=span.start_ts.isoformat(),
            end_ts=span.end_ts.isoformat(),
            label=span.label,
            provenance=span.provenance,
            user_id=span.user_id,
            confidence=span.confidence,
            extend_forward=span.extend_forward,
        )

    # -- REST: queue ----------------------------------------------------------

    @app.get("/api/queue")
    async def list_queue(limit: int = Query(20, ge=1, le=100)) -> list[QueueItemResponse]:
        if not queue_path.exists():
            return []
        queue = ActiveLabelingQueue(queue_path)
        pending = queue.get_pending(limit=limit)
        return [
            QueueItemResponse(
                request_id=r.request_id,
                user_id=r.user_id,
                bucket_start_ts=r.bucket_start_ts.isoformat(),
                bucket_end_ts=r.bucket_end_ts.isoformat(),
                reason=r.reason,
                confidence=r.confidence,
                predicted_label=r.predicted_label,
                status=r.status,
            )
            for r in pending
        ]

    @app.post("/api/queue/{request_id}/done")
    async def mark_queue_done(request_id: str, body: MarkDoneRequest) -> dict[str, str]:
        if not queue_path.exists():
            return {"status": "not_found"}
        queue = ActiveLabelingQueue(queue_path)
        result = queue.mark_done(request_id, status=body.status)
        if result is None:
            return {"status": "not_found"}
        return {"status": result.status}

    # -- REST: features -------------------------------------------------------

    @app.get("/api/features/summary")
    async def feature_summary(
        start: str = Query(..., description="ISO-8601 start"),
        end: str = Query(..., description="ISO-8601 end"),
    ) -> FeatureSummaryResponse:
        import pandas as pd

        start_ts = dt.datetime.fromisoformat(start)
        end_ts = dt.datetime.fromisoformat(end)

        frames: list[pd.DataFrame] = []
        current = start_ts.date()
        while current <= end_ts.date():
            fp = data_dir / f"features_v1/date={current.isoformat()}" / "features.parquet"
            if fp.exists():
                frames.append(read_parquet(fp))
            current += dt.timedelta(days=1)

        if not frames:
            return FeatureSummaryResponse(
                top_apps=[], mean_keys_per_min=None, mean_clicks_per_min=None,
                mean_scroll_per_min=None, total_buckets=0, session_count=0,
            )

        df = pd.concat(frames, ignore_index=True)
        summary = generate_label_summary(df, start_ts, end_ts)
        return FeatureSummaryResponse(**summary)

    # -- REST: ActivityWatch live proxy ---------------------------------------

    @app.get("/api/aw/live")
    async def aw_live_summary(
        start: str = Query(...),
        end: str = Query(...),
    ) -> list[AWLiveEntry]:
        try:
            from taskclf.adapters.activitywatch.client import (
                fetch_aw_events,
                find_window_bucket_id,
            )

            start_ts = dt.datetime.fromisoformat(start)
            end_ts = dt.datetime.fromisoformat(end)

            bucket_id = find_window_bucket_id(aw_host)
            events = fetch_aw_events(aw_host, bucket_id, start_ts, end_ts, title_salt=title_salt)
            if not events:
                return []
            counts = Counter(ev.app_id for ev in events)
            return [AWLiveEntry(app=app, events=cnt) for app, cnt in counts.most_common(5)]
        except Exception:
            logger.debug("AW live summary unavailable", exc_info=True)
            return []

    # -- REST: config ---------------------------------------------------------

    @app.get("/api/config/labels")
    async def config_labels() -> list[str]:
        return [cl.value for cl in CoreLabel]

    @app.get("/api/config/user")
    async def get_user_config() -> UserConfigResponse:
        return UserConfigResponse(
            user_id=user_config.user_id,
            username=user_config.username,
        )

    @app.put("/api/config/user")
    async def update_user_config(body: UserConfigUpdateRequest) -> UserConfigResponse:
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        if patch:
            try:
                user_config.update(patch)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return UserConfigResponse(
            user_id=user_config.user_id,
            username=user_config.username,
        )

    # -- REST: window control -------------------------------------------------

    @app.post("/api/window/toggle")
    async def window_toggle() -> dict[str, Any]:
        if window_api is None:
            return {"status": "no_window", "visible": False}
        window_api.toggle_window()
        return {"status": "ok", "visible": window_api.visible}

    @app.get("/api/window/state")
    async def window_state() -> dict[str, Any]:
        if window_api is None:
            return {"available": False, "visible": False}
        return {"available": True, "visible": window_api.visible}

    # -- WebSocket ------------------------------------------------------------

    @app.websocket("/ws/predictions")
    async def ws_predictions(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async with bus.subscribe() as queue:
                while True:
                    event = await queue.get()
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket error", exc_info=True)

    # -- Static files (SPA) ---------------------------------------------------

    if _STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa_fallback(path: str) -> FileResponse:
            file_path = _STATIC_DIR / path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(_STATIC_DIR / "index.html")

    return app
