"""FastAPI backend for the taskclf labeling web UI.

Provides REST endpoints for label CRUD, queue management, and feature
summaries, plus a WebSocket channel for live prediction streaming.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from taskclf.core.config import UserConfig
from taskclf.core.defaults import DEFAULT_AW_HOST, DEFAULT_DATA_DIR, DEFAULT_TITLE_SALT
from taskclf.core.store import read_parquet
from taskclf.core.time import to_naive_utc
from taskclf.core.types import CoreLabel, LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue
from taskclf.labels.store import (
    append_label_span,
    delete_label_span,
    export_labels_to_csv,
    generate_label_summary,
    import_labels_from_csv,
    merge_label_spans,
    read_label_spans,
    update_label_span,
    write_label_spans,
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


class LabelUpdateRequest(BaseModel):
    start_ts: str = Field(description="ISO-8601 start timestamp (identifies the span)")
    end_ts: str = Field(description="ISO-8601 end timestamp (identifies the span)")
    label: str = Field(description="New label to assign")


class LabelDeleteRequest(BaseModel):
    start_ts: str = Field(description="ISO-8601 start timestamp (identifies the span)")
    end_ts: str = Field(description="ISO-8601 end timestamp (identifies the span)")


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


class NotificationAcceptRequest(BaseModel):
    block_start: str = Field(description="ISO-8601 start of the activity block")
    block_end: str = Field(description="ISO-8601 end of the activity block")
    label: str = Field(description="Suggested label to accept")


class LabelStatsResponse(BaseModel):
    date: str
    count: int
    total_minutes: float
    breakdown: dict[str, float]


class LabelImportResponse(BaseModel):
    status: str
    imported: int
    total: int
    strategy: str


class OverlapErrorDetail(BaseModel):
    error: str
    conflicting_start_ts: str | None = None
    conflicting_end_ts: str | None = None


def _utc_iso(ts: dt.datetime) -> str:
    """Format a datetime as ISO-8601 with explicit UTC timezone suffix."""
    if ts.tzinfo is None:
        return ts.isoformat() + "+00:00"
    return ts.astimezone(dt.timezone.utc).isoformat()


_to_naive_utc = to_naive_utc


_OVERLAP_RE = re.compile(
    r"Span \[(.+?), (.+?)\) overlaps \[(.+?), (.+?)\) for user",
)


def _parse_overlap_error(
    msg: str,
    new_start: dt.datetime,
    new_end: dt.datetime,
) -> OverlapErrorDetail:
    """Extract the conflicting (existing) span timestamps from the overlap error."""
    m = _OVERLAP_RE.search(msg)
    if m is None:
        return OverlapErrorDetail(error=msg)

    span_a = (m.group(1).strip(), m.group(2).strip())
    span_b = (m.group(3).strip(), m.group(4).strip())

    new_start_str = str(new_start)
    new_end_str = str(new_end)

    if span_a == (new_start_str, new_end_str):
        conflict = span_b
    elif span_b == (new_start_str, new_end_str):
        conflict = span_a
    else:
        conflict = span_a

    return OverlapErrorDetail(
        error=msg,
        conflicting_start_ts=conflict[0],
        conflicting_end_ts=conflict[1],
    )


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
    on_label_saved: Callable[[], None] | None = None,
    pause_toggle: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        data_dir: Path to the processed data directory.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        event_bus: Shared event bus for WebSocket broadcasting.
        window_api: Optional ``WindowAPI`` for pywebview window control.
        on_label_saved: Optional callback invoked after a label is
            successfully saved (via ``POST /api/labels`` or
            ``POST /api/notification/accept``).
        pause_toggle: Optional callback to toggle pause state; returns
            new paused boolean.
        is_paused: Optional callable returning current paused state.
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
    async def list_labels(
        limit: int = Query(50, ge=1, le=500),
        date: str | None = Query(None, description="ISO-8601 date to filter labels by (e.g. 2025-03-07)"),
    ) -> list[LabelResponse]:
        if not labels_path.exists():
            return []
        spans = read_label_spans(labels_path)

        if date is not None:
            try:
                target = dt.date.fromisoformat(date)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid date: {date}") from exc
            day_start = dt.datetime.combine(target, dt.time.min)
            day_end = dt.datetime.combine(target, dt.time.max)
            spans = [s for s in spans if s.end_ts > day_start and s.start_ts < day_end]

        spans.sort(key=lambda s: s.start_ts, reverse=True)
        return [
            LabelResponse(
                start_ts=_utc_iso(s.start_ts),
                end_ts=_utc_iso(s.end_ts),
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
                start_ts=_to_naive_utc(dt.datetime.fromisoformat(body.start_ts)),
                end_ts=_to_naive_utc(dt.datetime.fromisoformat(body.end_ts)),
                label=body.label,
                provenance="manual",
                user_id=uid,
                confidence=body.confidence if body.confidence is not None else 1.0,
                extend_forward=body.extend_forward,
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            append_label_span(span, labels_path)
        except ValueError as exc:
            detail = _parse_overlap_error(str(exc), span.start_ts, span.end_ts)
            raise HTTPException(
                status_code=409, detail=detail.model_dump(),
            ) from exc

        if on_label_saved is not None:
            on_label_saved()

        await bus.publish({"type": "suggestion_cleared", "reason": "label_saved"})

        if span.extend_forward:
            await bus.publish({
                "type": "label_created",
                "label": span.label,
                "confidence": span.confidence if span.confidence is not None else 1.0,
                "ts": _utc_iso(span.end_ts),
                "start_ts": _utc_iso(span.start_ts),
                "extend_forward": True,
            })

        return LabelResponse(
            start_ts=_utc_iso(span.start_ts),
            end_ts=_utc_iso(span.end_ts),
            label=span.label,
            provenance=span.provenance,
            user_id=span.user_id,
            confidence=span.confidence,
            extend_forward=span.extend_forward,
        )

    @app.put("/api/labels")
    async def update_label(body: LabelUpdateRequest) -> LabelResponse:
        try:
            start = _to_naive_utc(dt.datetime.fromisoformat(body.start_ts))
            end = _to_naive_utc(dt.datetime.fromisoformat(body.end_ts))
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            span = update_label_span(start, end, body.label, labels_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LabelResponse(
            start_ts=_utc_iso(span.start_ts),
            end_ts=_utc_iso(span.end_ts),
            label=span.label,
            provenance=span.provenance,
            user_id=span.user_id,
            confidence=span.confidence,
            extend_forward=span.extend_forward,
        )

    @app.delete("/api/labels")
    async def delete_label(body: LabelDeleteRequest) -> dict[str, str]:
        try:
            start = _to_naive_utc(dt.datetime.fromisoformat(body.start_ts))
            end = _to_naive_utc(dt.datetime.fromisoformat(body.end_ts))
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            delete_label_span(start, end, labels_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted"}

    @app.get("/api/labels/export")
    async def export_labels() -> StreamingResponse:
        """Download all label spans as a CSV file."""
        import io
        import tempfile

        if not labels_path.exists():
            raise HTTPException(status_code=404, detail="No labels file found")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels_export.csv"
            try:
                export_labels_to_csv(labels_path, csv_path)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            csv_bytes = csv_path.read_bytes()

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=labels_export.csv"},
        )

    @app.get("/api/labels/stats")
    async def label_stats(
        date: str | None = Query(None, description="ISO-8601 date (defaults to today UTC)"),
    ) -> LabelStatsResponse:
        """Return labeling stats for a given day."""
        target = (
            dt.date.fromisoformat(date) if date
            else dt.datetime.now(dt.timezone.utc).date()
        )
        if not labels_path.exists():
            return LabelStatsResponse(
                date=target.isoformat(), count=0,
                total_minutes=0.0, breakdown={},
            )
        spans = read_label_spans(labels_path)
        day_spans = [s for s in spans if s.start_ts.date() == target]
        breakdown: dict[str, float] = {}
        for s in day_spans:
            mins = round((s.end_ts - s.start_ts).total_seconds() / 60, 1)
            breakdown[s.label] = round(breakdown.get(s.label, 0) + mins, 1)
        total = round(sum(breakdown.values()), 1)
        return LabelStatsResponse(
            date=target.isoformat(), count=len(day_spans),
            total_minutes=total, breakdown=breakdown,
        )

    @app.post("/api/labels/import")
    async def import_labels(
        file: UploadFile,
        strategy: str = Form("merge"),
    ) -> LabelImportResponse:
        """Import label spans from an uploaded CSV file.

        Accepts ``strategy`` of ``"merge"`` (deduplicate and combine
        with existing labels) or ``"overwrite"`` (replace all labels).
        """
        import tempfile

        if strategy not in ("merge", "overwrite"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid strategy {strategy!r}; must be 'merge' or 'overwrite'",
            )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            imported = import_labels_from_csv(tmp_path)
        except ValueError as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        if strategy == "overwrite":
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            write_label_spans(imported, labels_path)
            total = len(imported)
        else:
            existing: list = []
            if labels_path.exists():
                existing = read_label_spans(labels_path)
            try:
                merged = merge_label_spans(existing, imported)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            write_label_spans(merged, labels_path)
            total = len(merged)

        return LabelImportResponse(
            status="ok",
            imported=len(imported),
            total=total,
            strategy=strategy,
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
                bucket_start_ts=_utc_iso(r.bucket_start_ts),
                bucket_end_ts=_utc_iso(r.bucket_end_ts),
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

        start_ts = _to_naive_utc(dt.datetime.fromisoformat(start))
        end_ts = _to_naive_utc(dt.datetime.fromisoformat(end))

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

            start_ts = _to_naive_utc(dt.datetime.fromisoformat(start))
            end_ts = _to_naive_utc(dt.datetime.fromisoformat(end))

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

    @app.post("/api/window/show-label-grid")
    async def window_show_label_grid() -> dict[str, str]:
        if window_api is not None:
            window_api.show_label_grid()
        await bus.publish({"type": "show_label_grid"})
        return {"status": "ok"}

    # -- REST: notification actions -------------------------------------------

    @app.post("/api/notification/skip")
    async def notification_skip() -> dict[str, str]:
        logger.info("Notification skipped by user (no label change needed)")
        return {"status": "skipped"}

    @app.post("/api/notification/accept")
    async def notification_accept(body: NotificationAcceptRequest) -> LabelResponse:
        uid = user_config.user_id
        try:
            span = LabelSpan(
                start_ts=_to_naive_utc(dt.datetime.fromisoformat(body.block_start)),
                end_ts=_to_naive_utc(dt.datetime.fromisoformat(body.block_end)),
                label=body.label,
                provenance="suggestion",
                user_id=uid,
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            append_label_span(span, labels_path)
        except ValueError as exc:
            detail = _parse_overlap_error(str(exc), span.start_ts, span.end_ts)
            raise HTTPException(
                status_code=409, detail=detail.model_dump(),
            ) from exc

        if on_label_saved is not None:
            on_label_saved()

        await bus.publish({"type": "suggestion_cleared", "reason": "label_saved"})

        logger.info("Accepted suggested label: %s (%s → %s)", body.label, body.block_start, body.block_end)
        return LabelResponse(
            start_ts=_utc_iso(span.start_ts),
            end_ts=_utc_iso(span.end_ts),
            label=span.label,
            provenance=span.provenance,
            user_id=span.user_id,
        )

    # -- REST: tray control ---------------------------------------------------

    @app.post("/api/tray/pause")
    async def tray_pause_toggle() -> dict[str, Any]:
        if pause_toggle is None:
            return {"status": "unavailable", "paused": False}
        paused = pause_toggle()
        return {"status": "ok", "paused": paused}

    @app.get("/api/tray/state")
    async def tray_state() -> dict[str, Any]:
        if is_paused is None:
            return {"available": False, "paused": False}
        return {"available": True, "paused": is_paused()}

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
