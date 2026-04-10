"""FastAPI backend for the taskclf labeling web UI.

Provides REST endpoints for label CRUD, queue management, feature
summaries, and model training, plus a WebSocket channel for live
prediction streaming and training progress.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from taskclf.core.config import UserConfig
from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_AW_TIMEOUT_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)
from taskclf.core.schema import resolve_feature_parquet_path
from taskclf.core.time import ts_utc_aware_get
from taskclf.core.types import CoreLabel, LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue
from taskclf.labels.store import (
    _extend_forward_coverage_contains,
    append_label_span,
    delete_label_span,
    export_labels_to_csv,
    generate_label_summary,
    import_labels_from_csv,
    merge_label_spans,
    overwrite_label_span,
    read_label_spans,
    update_label_span,
    write_label_spans,
)
from taskclf.ui.events import EventBus
from taskclf.ui.activity_provider import (
    ActivityProviderUnavailableError,
    ActivityWatchProvider,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _feature_parquet_for_date(data_dir: Path, target_date: dt.date) -> Path | None:
    return resolve_feature_parquet_path(data_dir, target_date)


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
    overwrite: bool = Field(
        default=False,
        description="When true, overlapping same-user spans are "
        "truncated/split/removed to make room for this label.",
    )
    allow_overlap: bool = Field(
        default=False,
        description="When true, skip overlap checks and allow "
        "multiple labels to coexist on the same time range.",
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
    new_start_ts: str | None = Field(
        default=None, description="New start timestamp (if changing time)"
    )
    new_end_ts: str | None = Field(
        default=None, description="New end timestamp (if changing time)"
    )
    extend_forward: bool | None = Field(
        default=None,
        description="When provided, replaces the span's extend_forward flag.",
    )


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


class ActivityProviderStatusResponse(BaseModel):
    provider_id: str
    provider_name: str
    state: Literal["checking", "ready", "setup_required"]
    summary_available: bool
    endpoint: str
    source_id: str | None = None
    last_sample_count: int = 0
    last_sample_breakdown: dict[str, int] = Field(default_factory=dict)
    setup_title: str
    setup_message: str
    setup_steps: list[str]
    help_url: str


class ActivitySummaryResponse(FeatureSummaryResponse):
    activity_provider: ActivityProviderStatusResponse
    recent_apps: list[AWLiveEntry] = Field(default_factory=list)
    range_state: Literal["ok", "no_data", "provider_unavailable"]
    message: str | None = None


class UserConfigResponse(BaseModel):
    user_id: str
    username: str
    suggestion_banner_ttl_seconds: int = Field(
        default=0,
        ge=0,
        description="0 disables auto-dismiss; positive values set the client timer in seconds.",
    )


class UserConfigUpdateRequest(BaseModel):
    username: str | None = None
    suggestion_banner_ttl_seconds: int | None = Field(
        default=None,
        ge=0,
        description="When set, persists suggestion banner TTL (0 = disabled).",
    )


class NotificationAcceptRequest(BaseModel):
    block_start: str = Field(description="ISO-8601 start of the activity block")
    block_end: str = Field(description="ISO-8601 end of the activity block")
    label: str = Field(description="Suggested label to accept")
    overwrite: bool = Field(
        default=False,
        description="When true, overlapping same-user spans are "
        "truncated/split/removed to make room for this label.",
    )
    allow_overlap: bool = Field(
        default=False,
        description="When true, skip overlap checks and allow "
        "multiple labels to coexist on the same time range.",
    )


class LabelStoppedEventResponse(BaseModel):
    type: Literal["label_stopped"] = "label_stopped"
    ts: str


class LabelsChangedEventResponse(BaseModel):
    type: Literal["labels_changed"] = "labels_changed"
    reason: str
    ts: str


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


class ConflictingSpan(BaseModel):
    start_ts: str
    end_ts: str
    label: str


class OverlapErrorDetail(BaseModel):
    error: str
    conflicting_start_ts: str | None = None
    conflicting_end_ts: str | None = None
    conflicting_label: str | None = None
    conflicting_spans: list[ConflictingSpan] = []


class TrayActionRequest(BaseModel):
    model_id: str | None = None


class TrainStartRequest(BaseModel):
    date_from: str = Field(description="Start date (YYYY-MM-DD)")
    date_to: str = Field(description="End date (YYYY-MM-DD, inclusive)")
    num_boost_round: int = Field(default=DEFAULT_NUM_BOOST_ROUND)
    class_weight: Literal["balanced", "none"] = "balanced"
    synthetic: bool = False


class BuildFeaturesRequest(BaseModel):
    date_from: str = Field(description="Start date (YYYY-MM-DD)")
    date_to: str = Field(description="End date (YYYY-MM-DD, inclusive)")


class TrainStatusResponse(BaseModel):
    job_id: str | None = None
    status: Literal["idle", "running", "complete", "failed"]
    step: str | None = None
    progress_pct: int | None = None
    message: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None
    model_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ModelBundleResponse(BaseModel):
    model_id: str
    path: str
    valid: bool
    invalid_reason: str | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    created_at: str | None = None


def _bundle_inspect_bundle_only_payload(model_dir: Path) -> dict[str, Any]:
    """Build JSON-serializable bundle-only inspection (validation metrics from disk)."""
    from taskclf.model_inspection import inspect_bundle_only

    bundle_path, metadata, section = inspect_bundle_only(model_dir)
    return {
        "bundle_path": str(bundle_path),
        "metadata": metadata.model_dump(),
        "bundle_saved_validation": section.model_dump(),
    }


class DataCheckResponse(BaseModel):
    date_from: str
    date_to: str
    dates_with_features: list[str]
    dates_missing_features: list[str]
    total_feature_rows: int
    label_span_count: int
    trainable_rows: int = 0
    trainable_labels: list[str] = []
    dates_built: list[str] = []
    build_errors: list[str] = []


@dataclass(eq=False)
class _TrainJob:
    """Mutable state for a single background training job."""

    job_id: str = ""
    status: str = "idle"
    step: str | None = None
    progress_pct: int | None = None
    message: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None
    model_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_response(self) -> TrainStatusResponse:
        return TrainStatusResponse(
            job_id=self.job_id or None,
            status=self.status,  # type: ignore[arg-type]
            step=self.step,
            progress_pct=self.progress_pct,
            message=self.message,
            error=self.error,
            metrics=self.metrics,
            model_dir=self.model_dir,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


def _utc_iso(ts: dt.datetime) -> str:
    """Format a datetime as ISO-8601 with explicit UTC timezone suffix."""
    if ts.tzinfo is None:
        return ts.isoformat() + "+00:00"
    return ts.astimezone(dt.timezone.utc).isoformat()


def _label_span_is_open_ended(span: LabelSpan) -> bool:
    return span.extend_forward and span.end_ts <= span.start_ts


def _label_response_from_span(span: LabelSpan) -> LabelResponse:
    return LabelResponse(
        start_ts=_utc_iso(span.start_ts),
        end_ts=_utc_iso(span.end_ts),
        label=span.label,
        provenance=span.provenance,
        user_id=span.user_id,
        confidence=span.confidence,
        extend_forward=span.extend_forward,
    )


_ensure_utc = ts_utc_aware_get


_OVERLAP_RE = re.compile(
    r"Span \[(.+?), (.+?)\) overlaps \[(.+?), (.+?)\) for user",
)


def _parse_overlap_error(
    msg: str,
    new_span: LabelSpan,
    existing: list[LabelSpan] | None = None,
) -> OverlapErrorDetail:
    """Extract all conflicting (existing) spans from the overlap error."""
    m = _OVERLAP_RE.search(msg)
    if m is None:
        return OverlapErrorDetail(error=msg)

    span_a = (m.group(1).strip(), m.group(2).strip())
    span_b = (m.group(3).strip(), m.group(4).strip())

    new_start_str = str(new_span.start_ts)
    new_end_str = str(new_span.end_ts)

    if span_a == (new_start_str, new_end_str):
        fc_raw = span_b
    elif span_b == (new_start_str, new_end_str):
        fc_raw = span_a
    else:
        fc_raw = span_a
    first_conflict = (
        _utc_iso(dt.datetime.fromisoformat(fc_raw[0])),
        _utc_iso(dt.datetime.fromisoformat(fc_raw[1])),
    )

    conflicts: list[ConflictingSpan] = []
    first_label: str | None = None

    if existing:
        for s in existing:
            if s.start_ts == new_span.start_ts and s.end_ts == new_span.end_ts:
                continue
            same_user = (
                s.user_id is None
                or new_span.user_id is None
                or s.user_id == new_span.user_id
            )
            if (
                same_user
                and s.start_ts < new_span.end_ts
                and new_span.start_ts < s.end_ts
            ):
                conflicts.append(
                    ConflictingSpan(
                        start_ts=_utc_iso(s.start_ts),
                        end_ts=_utc_iso(s.end_ts),
                        label=s.label,
                    )
                )

        fc_start, fc_end = first_conflict
        for c in conflicts:
            if c.start_ts == fc_start and c.end_ts == fc_end:
                first_label = c.label
                break

    return OverlapErrorDetail(
        error=msg,
        conflicting_start_ts=first_conflict[0],
        conflicting_end_ts=first_conflict[1],
        conflicting_label=first_label,
        conflicting_spans=conflicts,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    data_dir: Path = Path(DEFAULT_DATA_DIR),
    models_dir: Path | None = None,
    aw_host: str = DEFAULT_AW_HOST,
    title_salt: str = DEFAULT_TITLE_SALT,
    event_bus: EventBus | None = None,
    window_api: Any = None,
    on_label_saved: Callable[[], None] | None = None,
    on_model_trained: Callable[[str], None] | None = None,
    on_suggestion_accepted: Callable[[], None] | None = None,
    pause_toggle: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,
    tray_actions: dict[str, Callable[..., Any]] | None = None,
    get_tray_state: Callable[[], dict[str, Any]] | None = None,
    get_activity_provider_status: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        data_dir: Path to the processed data directory.
        models_dir: Path to the directory containing model bundles.
        aw_host: ActivityWatch server URL.
        title_salt: Salt for hashing window titles.
        event_bus: Shared event bus for WebSocket broadcasting.
        window_api: Optional ``WindowAPI`` for pywebview window control.
        on_label_saved: Optional callback invoked after a label is
            successfully saved (via ``POST /api/labels`` or
            ``POST /api/notification/accept``).
        on_model_trained: Optional callback invoked with the model run
            directory path after training completes successfully.
        on_suggestion_accepted: Optional callback invoked after a
            transition suggestion is accepted via
            ``POST /api/notification/accept``.  Used by ``TrayLabeler``
            to trigger gap-fill prompting when adjacent unlabeled time
            exists.
        pause_toggle: Optional callback to toggle pause state; returns
            new paused boolean.
        is_paused: Optional callable returning current paused state.
        tray_actions: Optional mapping of action names to callbacks for the tray menu.
        get_tray_state: Optional callable returning the full tray state dictionary.
        get_activity_provider_status: Optional callable returning the latest
            cached activity-source status snapshot from the runtime monitor.
    """
    bus = event_bus or EventBus()
    labels_path = data_dir / "labels_v1" / "labels.parquet"
    queue_path = data_dir / "labels_v1" / "queue.json"
    user_config = UserConfig(data_dir)
    effective_title_salt = title_salt or user_config.title_secret
    title_salt = effective_title_salt
    effective_models_dir = models_dir or Path(DEFAULT_MODELS_DIR)
    train_job = _TrainJob()
    activity_provider = ActivityWatchProvider(
        endpoint=aw_host,
        title_salt=effective_title_salt,
    )

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

    async def publish_labels_changed(reason: str) -> None:
        await bus.publish(
            LabelsChangedEventResponse(
                reason=reason,
                ts=_utc_iso(dt.datetime.now(dt.timezone.utc)),
            ).model_dump()
        )

    def _empty_feature_summary() -> FeatureSummaryResponse:
        return FeatureSummaryResponse(
            top_apps=[],
            mean_keys_per_min=None,
            mean_clicks_per_min=None,
            mean_scroll_per_min=None,
            total_buckets=0,
            session_count=0,
        )

    def _feature_summary_for_range(
        start_ts: dt.datetime,
        end_ts: dt.datetime,
    ) -> FeatureSummaryResponse:
        import pandas as pd

        from taskclf.core.store import read_parquet

        empty_resp = _empty_feature_summary()
        frames: list[pd.DataFrame] = []
        dates_missing_parquet: list[dt.date] = []
        current = start_ts.date()
        while current <= end_ts.date():
            fp = _feature_parquet_for_date(data_dir, current)
            if fp is not None and fp.exists():
                tmp = read_parquet(fp)
                if not tmp.empty:
                    frames.append(tmp)
                else:
                    dates_missing_parquet.append(current)
            else:
                dates_missing_parquet.append(current)
            current += dt.timedelta(days=1)

        if dates_missing_parquet:
            try:
                from taskclf.features.build import _fetch_aw_features_for_date

                for date_value in dates_missing_parquet:
                    rows = _fetch_aw_features_for_date(
                        date_value,
                        aw_host=aw_host,
                        title_salt=title_salt,
                    )
                    if rows:
                        frames.append(pd.DataFrame([row.model_dump() for row in rows]))
            except Exception:
                logger.debug("AW live feature fallback unavailable", exc_info=True)

        if not frames:
            return empty_resp

        df = pd.concat(frames, ignore_index=True)
        if "bucket_start_ts" not in df.columns:
            return empty_resp

        summary = generate_label_summary(df, start_ts, end_ts)
        return FeatureSummaryResponse(**summary)

    def _activity_provider_status_snapshot() -> ActivityProviderStatusResponse:
        snapshot = (
            get_activity_provider_status() if get_activity_provider_status else None
        )
        if snapshot is not None:
            validated = ActivityProviderStatusResponse.model_validate(snapshot)
            if validated.state != "checking":
                return validated
        probe_timeout = min(
            2,
            int(
                user_config.as_dict().get(
                    "aw_timeout_seconds", DEFAULT_AW_TIMEOUT_SECONDS
                )
            ),
        )
        return ActivityProviderStatusResponse.model_validate(
            activity_provider.probe_status(timeout_seconds=probe_timeout).to_payload()
        )

    # -- REST: labels ---------------------------------------------------------

    @app.get("/api/labels")
    async def api_op_labels_get(
        limit: int = Query(50, ge=1, le=500),
        date: str | None = Query(
            None, description="ISO-8601 date to filter labels by (e.g. 2025-03-07)"
        ),
        range_start: str | None = Query(
            None, description="UTC start of visible range (ISO-8601)"
        ),
        range_end: str | None = Query(
            None, description="UTC end of visible range (ISO-8601)"
        ),
    ) -> list[LabelResponse]:
        if not labels_path.exists():
            return []
        spans = read_label_spans(labels_path)

        if range_start is not None and range_end is not None:
            try:
                rs = _ensure_utc(dt.datetime.fromisoformat(range_start))
                re_ = _ensure_utc(dt.datetime.fromisoformat(range_end))
            except (ValueError, Exception) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid range: {exc}"
                ) from exc
            spans = [s for s in spans if s.end_ts > rs and s.start_ts < re_]
        elif date is not None:
            try:
                target = dt.date.fromisoformat(date)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid date: {date}"
                ) from exc
            day_start = dt.datetime.combine(target, dt.time.min, tzinfo=dt.timezone.utc)
            day_end = dt.datetime.combine(target, dt.time.max, tzinfo=dt.timezone.utc)
            spans = [s for s in spans if s.end_ts > day_start and s.start_ts < day_end]

        # Latest-ended first: matches tray gap-fill (max end_ts) and quick-label gap.
        spans.sort(key=lambda s: s.end_ts, reverse=True)
        return [_label_response_from_span(s) for s in spans[:limit]]

    @app.get("/api/labels/current")
    async def api_op_labels_current_get() -> LabelResponse | None:
        if not labels_path.exists():
            return None

        spans = read_label_spans(labels_path)
        current = [span for span in spans if _label_span_is_open_ended(span)]
        if not current:
            return None

        return _label_response_from_span(max(current, key=lambda span: span.start_ts))

    @app.post("/api/labels", status_code=201)
    async def api_op_labels_post(body: LabelCreateRequest) -> LabelResponse:
        uid = body.user_id if body.user_id is not None else user_config.user_id
        try:
            span = LabelSpan(
                start_ts=_ensure_utc(dt.datetime.fromisoformat(body.start_ts)),
                end_ts=_ensure_utc(dt.datetime.fromisoformat(body.end_ts)),
                label=body.label,
                provenance="manual",
                user_id=uid,
                confidence=body.confidence if body.confidence is not None else 1.0,
                extend_forward=body.extend_forward,
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.overwrite:
            overwrite_label_span(span, labels_path)
        else:
            try:
                append_label_span(
                    span,
                    labels_path,
                    allow_overlap=body.allow_overlap,
                )
            except ValueError as exc:
                existing = read_label_spans(labels_path) if labels_path.exists() else []
                detail = _parse_overlap_error(str(exc), span, existing)
                raise HTTPException(
                    status_code=409,
                    detail=detail.model_dump(),
                ) from exc

        if on_label_saved is not None:
            on_label_saved()

        if span.extend_forward:
            await bus.publish(
                {
                    "type": "label_created",
                    "label": span.label,
                    "confidence": span.confidence
                    if span.confidence is not None
                    else 1.0,
                    "ts": _utc_iso(span.end_ts),
                    "start_ts": _utc_iso(span.start_ts),
                    "extend_forward": True,
                }
            )
        await publish_labels_changed("created")

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
    async def api_op_labels_put(body: LabelUpdateRequest) -> LabelResponse:
        prior_span: LabelSpan | None = None
        try:
            start = _ensure_utc(dt.datetime.fromisoformat(body.start_ts))
            end = _ensure_utc(dt.datetime.fromisoformat(body.end_ts))
            new_start = (
                _ensure_utc(dt.datetime.fromisoformat(body.new_start_ts))
                if body.new_start_ts
                else None
            )
            new_end = (
                _ensure_utc(dt.datetime.fromisoformat(body.new_end_ts))
                if body.new_end_ts
                else None
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if labels_path.exists():
            for existing in read_label_spans(labels_path):
                if existing.start_ts == start and existing.end_ts == end:
                    prior_span = existing
                    break
        try:
            span = update_label_span(
                start,
                end,
                body.label,
                labels_path,
                new_start_ts=new_start,
                new_end_ts=new_end,
                new_extend_forward=body.extend_forward,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        was_open_ended = prior_span is not None and _label_span_is_open_ended(
            prior_span
        )
        is_closed = not span.extend_forward and span.end_ts > span.start_ts
        if was_open_ended and is_closed:
            await bus.publish(
                LabelStoppedEventResponse(ts=_utc_iso(span.end_ts)).model_dump()
            )
        await publish_labels_changed("updated")
        return _label_response_from_span(span)

    @app.delete("/api/labels")
    async def api_op_labels_delete(body: LabelDeleteRequest) -> dict[str, str]:
        try:
            start = _ensure_utc(dt.datetime.fromisoformat(body.start_ts))
            end = _ensure_utc(dt.datetime.fromisoformat(body.end_ts))
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            delete_label_span(start, end, labels_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await publish_labels_changed("deleted")
        return {"status": "deleted"}

    @app.get("/api/labels/export")
    async def api_op_labels_export_get() -> StreamingResponse:
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
        date: str | None = Query(
            None, description="ISO-8601 date (defaults to today UTC)"
        ),
    ) -> LabelStatsResponse:
        """Return labeling stats for a given day."""
        target = (
            dt.date.fromisoformat(date)
            if date
            else dt.datetime.now(dt.timezone.utc).date()
        )
        if not labels_path.exists():
            return LabelStatsResponse(
                date=target.isoformat(),
                count=0,
                total_minutes=0.0,
                breakdown={},
            )
        spans = read_label_spans(labels_path)
        day_spans = [s for s in spans if s.start_ts.date() == target]
        breakdown: dict[str, float] = {}
        for s in day_spans:
            mins = round((s.end_ts - s.start_ts).total_seconds() / 60, 1)
            breakdown[s.label] = round(breakdown.get(s.label, 0) + mins, 1)
        total = round(sum(breakdown.values()), 1)
        return LabelStatsResponse(
            date=target.isoformat(),
            count=len(day_spans),
            total_minutes=total,
            breakdown=breakdown,
        )

    @app.post("/api/labels/import")
    async def api_op_labels_import_post(
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

        await publish_labels_changed("imported")

        return LabelImportResponse(
            status="ok",
            imported=len(imported),
            total=total,
            strategy=strategy,
        )

    # -- REST: queue ----------------------------------------------------------

    @app.get("/api/queue")
    async def api_op_queue_get(
        limit: int = Query(20, ge=1, le=100),
    ) -> list[QueueItemResponse]:
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
    async def api_op_queue_done_post(
        request_id: str, body: MarkDoneRequest
    ) -> dict[str, str]:
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
        start_ts = _ensure_utc(dt.datetime.fromisoformat(start))
        end_ts = _ensure_utc(dt.datetime.fromisoformat(end))
        return _feature_summary_for_range(start_ts, end_ts)

    @app.get("/api/activity/summary")
    async def activity_summary(
        start: str = Query(..., description="ISO-8601 start"),
        end: str = Query(..., description="ISO-8601 end"),
    ) -> ActivitySummaryResponse:
        start_ts = _ensure_utc(dt.datetime.fromisoformat(start))
        end_ts = _ensure_utc(dt.datetime.fromisoformat(end))

        provider_status = _activity_provider_status_snapshot()
        empty_summary = _empty_feature_summary()

        if provider_status.state == "setup_required":
            return ActivitySummaryResponse(
                **empty_summary.model_dump(),
                activity_provider=provider_status,
                recent_apps=[],
                range_state="provider_unavailable",
                message=provider_status.setup_message,
            )

        try:
            provider_snapshot, recent_apps = activity_provider.recent_app_summary(
                start_ts,
                end_ts,
                source_id=provider_status.source_id,
                timeout_seconds=min(
                    2,
                    int(
                        user_config.as_dict().get(
                            "aw_timeout_seconds",
                            DEFAULT_AW_TIMEOUT_SECONDS,
                        )
                    ),
                ),
            )
            provider_status = ActivityProviderStatusResponse.model_validate(
                provider_snapshot.to_payload()
            )
        except ActivityProviderUnavailableError:
            provider_status = ActivityProviderStatusResponse.model_validate(
                activity_provider.setup_required_status(
                    source_id=provider_status.source_id,
                    last_sample_count=provider_status.last_sample_count,
                    last_sample_breakdown=provider_status.last_sample_breakdown,
                ).to_payload()
            )
            return ActivitySummaryResponse(
                **empty_summary.model_dump(),
                activity_provider=provider_status,
                recent_apps=[],
                range_state="provider_unavailable",
                message=provider_status.setup_message,
            )

        feature_payload = _feature_summary_for_range(start_ts, end_ts)
        has_context = bool(recent_apps) or feature_payload.total_buckets > 0
        range_state: Literal["ok", "no_data", "provider_unavailable"] = (
            "ok" if has_context else "no_data"
        )
        message = None if has_context else "No activity data for this window"

        return ActivitySummaryResponse(
            **feature_payload.model_dump(),
            activity_provider=provider_status,
            recent_apps=[
                AWLiveEntry(app=entry.app, events=entry.events) for entry in recent_apps
            ],
            range_state=range_state,
            message=message,
        )

    # -- REST: ActivityWatch live proxy ---------------------------------------

    @app.get("/api/aw/live")
    async def aw_live_summary(
        start: str = Query(...),
        end: str = Query(...),
    ) -> list[AWLiveEntry]:
        try:
            start_ts = _ensure_utc(dt.datetime.fromisoformat(start))
            end_ts = _ensure_utc(dt.datetime.fromisoformat(end))
            _, recent_apps = activity_provider.recent_app_summary(
                start_ts,
                end_ts,
                timeout_seconds=min(
                    2,
                    int(
                        user_config.as_dict().get(
                            "aw_timeout_seconds",
                            DEFAULT_AW_TIMEOUT_SECONDS,
                        )
                    ),
                ),
            )
            return [
                AWLiveEntry(app=entry.app, events=entry.events) for entry in recent_apps
            ]
        except Exception:
            logger.debug("AW live summary unavailable", exc_info=True)
            return []

    # -- REST: config ---------------------------------------------------------

    @app.get("/api/config/labels")
    async def config_labels() -> list[str]:
        return [cl.value for cl in CoreLabel]

    def _suggestion_banner_ttl_seconds() -> int:
        raw = user_config.as_dict().get("suggestion_banner_ttl_seconds", 0)
        try:
            n = int(raw)
        except TypeError, ValueError:
            return 0
        return max(0, n)

    @app.get("/api/config/user")
    async def api_op_config_user_get() -> UserConfigResponse:
        return UserConfigResponse(
            user_id=user_config.user_id,
            username=user_config.username,
            suggestion_banner_ttl_seconds=_suggestion_banner_ttl_seconds(),
        )

    @app.put("/api/config/user")
    async def api_op_config_user_put(
        body: UserConfigUpdateRequest,
    ) -> UserConfigResponse:
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        if patch:
            try:
                user_config.update(patch)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return UserConfigResponse(
            user_id=user_config.user_id,
            username=user_config.username,
            suggestion_banner_ttl_seconds=_suggestion_banner_ttl_seconds(),
        )

    # -- REST: window control -------------------------------------------------

    @app.post("/api/window/toggle")
    async def window_toggle() -> dict[str, Any]:
        if window_api is None:
            return {"status": "no_window", "visible": False}
        window_api.window_toggle()
        return {"status": "ok", "visible": window_api.visible}

    @app.get("/api/window/state")
    async def window_state() -> dict[str, Any]:
        if window_api is None:
            return {"available": False, "visible": False}
        return {"available": True, "visible": window_api.visible}

    @app.post("/api/window/show-label-grid")
    async def window_label_grid_show() -> dict[str, str]:
        if window_api is not None:
            window_api.label_grid_show()
        await bus.publish({"type": "label_grid_show"})
        return {"status": "ok"}

    # -- REST: notification actions -------------------------------------------

    @app.post("/api/notification/skip")
    async def notification_skip() -> dict[str, str]:
        await bus.publish({"type": "suggestion_cleared", "reason": "skipped"})
        logger.info("Notification skipped by user (no label change needed)")
        return {"status": "skipped"}

    @app.post("/api/notification/accept")
    async def notification_accept(body: NotificationAcceptRequest) -> LabelResponse:
        uid = user_config.user_id
        try:
            span = LabelSpan(
                start_ts=_ensure_utc(dt.datetime.fromisoformat(body.block_start)),
                end_ts=_ensure_utc(dt.datetime.fromisoformat(body.block_end)),
                label=body.label,
                provenance="suggestion",
                user_id=uid,
            )
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        existing = read_label_spans(labels_path) if labels_path.exists() else []
        auto_resume_active = _extend_forward_coverage_contains(existing, span)

        if body.overwrite or auto_resume_active:
            overwrite_label_span(span, labels_path)
        else:
            try:
                append_label_span(
                    span,
                    labels_path,
                    allow_overlap=body.allow_overlap,
                )
            except ValueError as exc:
                existing = read_label_spans(labels_path) if labels_path.exists() else []
                detail = _parse_overlap_error(str(exc), span, existing)
                raise HTTPException(
                    status_code=409,
                    detail=detail.model_dump(),
                ) from exc

        if on_label_saved is not None:
            on_label_saved()

        await bus.publish({"type": "suggestion_cleared", "reason": "label_saved"})
        await publish_labels_changed("suggestion_accepted")

        if on_suggestion_accepted is not None:
            on_suggestion_accepted()

        logger.info(
            "Accepted suggested label: %s (%s → %s)",
            body.label,
            body.block_start,
            body.block_end,
        )
        return LabelResponse(
            start_ts=_utc_iso(span.start_ts),
            end_ts=_utc_iso(span.end_ts),
            label=span.label,
            provenance=span.provenance,
            user_id=span.user_id,
            confidence=span.confidence,
            extend_forward=span.extend_forward,
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
        if get_tray_state is not None:
            state = get_tray_state()
            state["available"] = True
            return state
        if is_paused is None:
            return {"available": False, "paused": False}
        return {"available": True, "paused": is_paused()}

    @app.post("/api/tray/action/{action}")
    async def tray_action(
        action: str, body: TrayActionRequest | None = None
    ) -> dict[str, Any]:
        if tray_actions is None or action not in tray_actions:
            raise HTTPException(
                status_code=404,
                detail=f"Action {action} not found or no tray actions configured",
            )

        try:
            if action == "switch_model" and body and body.model_id:
                from taskclf.model_registry import list_bundles

                bundles = list_bundles(effective_models_dir)
                target = next((b for b in bundles if b.model_id == body.model_id), None)
                if not target:
                    raise HTTPException(
                        status_code=404, detail=f"Model {body.model_id} not found"
                    )
                tray_actions[action](target.path)
            else:
                tray_actions[action]()
            return {"status": "ok"}
        except Exception as exc:
            logger.warning("Tray action %s failed: %s", action, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # -- REST: training -------------------------------------------------------

    def _run_training_pipeline(
        job: _TrainJob,
        *,
        date_from: str,
        date_to: str,
        num_boost_round: int,
        class_weight: str,
        synthetic: bool,
    ) -> None:
        """Background thread: load data, train, save bundle, publish progress."""
        import pandas as pd

        try:
            start = dt.date.fromisoformat(date_from)
            end = dt.date.fromisoformat(date_to)

            def _update(step: str, pct: int, msg: str) -> None:
                job.step = step
                job.progress_pct = pct
                job.message = msg
                bus.publish_threadsafe(
                    {
                        "type": "train_progress",
                        "job_id": job.job_id,
                        "step": step,
                        "progress_pct": pct,
                        "message": msg,
                    }
                )

            if job._cancel.is_set():
                raise InterruptedError("Cancelled")

            _update("loading_features", 10, "Loading features…")

            from taskclf.core.store import read_parquet as _read_pq
            from taskclf.features.build import generate_dummy_features
            from taskclf.labels.store import generate_dummy_labels

            all_features: list[pd.DataFrame] = []
            all_labels: list = []
            current = start

            if not synthetic:
                lp = data_dir / "labels_v1" / "labels.parquet"
                if lp.exists():
                    from taskclf.labels.store import read_label_spans as _read_ls

                    all_spans = _read_ls(lp)
                    start_dt = pd.Timestamp(
                        year=start.year,
                        month=start.month,
                        day=start.day,
                        tz="UTC",
                    )
                    end_dt = pd.Timestamp(
                        year=end.year,
                        month=end.month,
                        day=end.day,
                        hour=23,
                        minute=59,
                        second=59,
                        tz="UTC",
                    )

                    def _to_utc(ts: dt.datetime) -> pd.Timestamp:
                        t = pd.Timestamp(ts)
                        if t.tzinfo is None:
                            return t.tz_localize("UTC")
                        return t.tz_convert("UTC")

                    all_labels = [
                        s
                        for s in all_spans
                        if _to_utc(s.end_ts) >= start_dt
                        and _to_utc(s.start_ts) <= end_dt
                    ]

            while current <= end:
                if job._cancel.is_set():
                    raise InterruptedError("Cancelled")
                if synthetic:
                    rows = generate_dummy_features(current, n_rows=60)
                    df = pd.DataFrame([r.model_dump() for r in rows])
                    labels = generate_dummy_labels(current, n_rows=60)
                    all_labels.extend(labels)
                else:
                    fp = _feature_parquet_for_date(data_dir, current)
                    if fp is not None and fp.exists():
                        df = _read_pq(fp)
                    else:
                        current += dt.timedelta(days=1)
                        continue
                all_features.append(df)
                current += dt.timedelta(days=1)

            if not all_features:
                raise ValueError("No feature data found for the given date range")

            features_df = pd.concat(all_features, ignore_index=True)

            if features_df.empty or "bucket_start_ts" not in features_df.columns:
                raise ValueError(
                    "Feature files exist but contain 0 rows — "
                    "ActivityWatch may not be running or has no data for the selected range"
                )

            if not all_labels:
                raise ValueError(
                    "No label spans overlap the selected date range — "
                    "create or import labels before training"
                )

            if job._cancel.is_set():
                raise InterruptedError("Cancelled")

            _update(
                "projecting_labels",
                30,
                f"Projecting {len(all_labels)} labels onto {len(features_df)} rows…",
            )

            from taskclf.labels.projection import project_blocks_to_windows

            labeled_df = project_blocks_to_windows(features_df, all_labels)
            if labeled_df.empty:
                raise ValueError(
                    "No labeled rows after projection — label spans may not "
                    "temporally overlap any feature windows in the selected range"
                )

            if job._cancel.is_set():
                raise InterruptedError("Cancelled")

            _update("splitting", 40, f"Splitting {len(labeled_df)} labeled rows…")

            from taskclf.train.dataset import split_by_time

            splits = split_by_time(labeled_df)
            train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
            val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)

            if job._cancel.is_set():
                raise InterruptedError("Cancelled")

            _update(
                "training",
                50,
                f"Training LightGBM ({num_boost_round} rounds, "
                f"{len(train_df)} train / {len(val_df)} val)…",
            )

            from taskclf.train.lgbm import train_lgbm as _train

            cw: Literal["balanced", "none"] = (
                "none" if class_weight == "none" else "balanced"
            )
            model, metrics, cm_df, params, cat_encoders = _train(
                train_df,
                val_df,
                num_boost_round=num_boost_round,
                class_weight=cw,
            )

            if job._cancel.is_set():
                raise InterruptedError("Cancelled")

            _update(
                "saving", 85, f"Saving bundle (macro_f1={metrics['macro_f1']:.3f})…"
            )

            from taskclf.core.model_io import build_metadata, save_model_bundle
            from taskclf.train.retrain import compute_dataset_hash

            dataset_hash = compute_dataset_hash(features_df, all_labels)
            metadata = build_metadata(
                label_set=metrics["label_names"],
                train_date_from=start,
                train_date_to=end,
                params=params,
                dataset_hash=dataset_hash,
                data_provenance="synthetic" if synthetic else "real",
                unknown_category_freq_threshold=params.get(
                    "unknown_category_freq_threshold"
                ),
                unknown_category_mask_rate=params.get("unknown_category_mask_rate"),
            )

            run_dir = save_model_bundle(
                model=model,
                metadata=metadata,
                metrics=metrics,
                confusion_df=cm_df,
                base_dir=effective_models_dir,
                cat_encoders=cat_encoders,
            )

            job.metrics = metrics
            job.model_dir = str(run_dir)
            job.status = "complete"
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            _update("done", 100, f"Model saved to {run_dir.name}")

            bus.publish_threadsafe(
                {
                    "type": "train_complete",
                    "job_id": job.job_id,
                    "metrics": {
                        "macro_f1": metrics.get("macro_f1"),
                        "weighted_f1": metrics.get("weighted_f1"),
                    },
                    "model_dir": str(run_dir),
                }
            )

            if on_model_trained is not None:
                try:
                    on_model_trained(str(run_dir))
                except Exception:
                    logger.debug("on_model_trained callback failed", exc_info=True)

        except InterruptedError:
            job.status = "failed"
            job.error = "Cancelled by user"
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            bus.publish_threadsafe(
                {
                    "type": "train_failed",
                    "job_id": job.job_id,
                    "error": "Cancelled by user",
                }
            )
        except Exception as exc:
            logger.warning("Training failed: %s", exc, exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            bus.publish_threadsafe(
                {
                    "type": "train_failed",
                    "job_id": job.job_id,
                    "error": str(exc),
                }
            )

    def _run_feature_build(
        job: _TrainJob,
        *,
        date_from: str,
        date_to: str,
    ) -> None:
        """Background thread: build features for each date in the range."""
        try:
            from taskclf.features.build import build_features_for_date

            start = dt.date.fromisoformat(date_from)
            end = dt.date.fromisoformat(date_to)
            total_days = (end - start).days + 1
            current = start
            built = 0

            while current <= end:
                if job._cancel.is_set():
                    raise InterruptedError("Cancelled")
                pct = int((built / total_days) * 100)
                job.step = "building_features"
                job.progress_pct = pct
                job.message = (
                    f"Building features for {current} ({built + 1}/{total_days})…"
                )
                bus.publish_threadsafe(
                    {
                        "type": "train_progress",
                        "job_id": job.job_id,
                        "step": "building_features",
                        "progress_pct": pct,
                        "message": job.message,
                    }
                )

                build_features_for_date(
                    current,
                    data_dir,
                    aw_host=aw_host,
                    title_salt=title_salt,
                    user_id=user_config.user_id,
                )
                built += 1
                current += dt.timedelta(days=1)

            job.status = "complete"
            job.step = "done"
            job.progress_pct = 100
            job.message = f"Built features for {built} day(s)"
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            bus.publish_threadsafe(
                {
                    "type": "train_complete",
                    "job_id": job.job_id,
                    "metrics": None,
                    "model_dir": None,
                }
            )
        except InterruptedError:
            job.status = "failed"
            job.error = "Cancelled by user"
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            bus.publish_threadsafe(
                {
                    "type": "train_failed",
                    "job_id": job.job_id,
                    "error": "Cancelled by user",
                }
            )
        except Exception as exc:
            logger.warning("Feature build failed: %s", exc, exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            bus.publish_threadsafe(
                {
                    "type": "train_failed",
                    "job_id": job.job_id,
                    "error": str(exc),
                }
            )

    @app.post("/api/train/start", status_code=202)
    async def train_start(body: TrainStartRequest) -> TrainStatusResponse:
        with train_job._lock:
            if train_job.status == "running":
                raise HTTPException(
                    status_code=409,
                    detail="A training job is already running",
                )
            train_job.job_id = uuid.uuid4().hex[:12]
            train_job.status = "running"
            train_job.step = "initializing"
            train_job.progress_pct = 0
            train_job.message = "Starting…"
            train_job.error = None
            train_job.metrics = None
            train_job.model_dir = None
            train_job.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            train_job.finished_at = None
            train_job._cancel.clear()

        thread = threading.Thread(
            target=_run_training_pipeline,
            args=(train_job,),
            kwargs={
                "date_from": body.date_from,
                "date_to": body.date_to,
                "num_boost_round": body.num_boost_round,
                "class_weight": body.class_weight,
                "synthetic": body.synthetic,
            },
            daemon=True,
        )
        thread.start()
        return train_job.to_response()

    @app.post("/api/train/build-features", status_code=202)
    async def train_build_features(body: BuildFeaturesRequest) -> TrainStatusResponse:
        with train_job._lock:
            if train_job.status == "running":
                raise HTTPException(
                    status_code=409,
                    detail="A training job is already running",
                )
            train_job.job_id = uuid.uuid4().hex[:12]
            train_job.status = "running"
            train_job.step = "building_features"
            train_job.progress_pct = 0
            train_job.message = "Starting feature build…"
            train_job.error = None
            train_job.metrics = None
            train_job.model_dir = None
            train_job.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            train_job.finished_at = None
            train_job._cancel.clear()

        thread = threading.Thread(
            target=_run_feature_build,
            args=(train_job,),
            kwargs={
                "date_from": body.date_from,
                "date_to": body.date_to,
            },
            daemon=True,
        )
        thread.start()
        return train_job.to_response()

    @app.get("/api/train/status")
    async def train_status() -> TrainStatusResponse:
        return train_job.to_response()

    @app.post("/api/train/cancel")
    async def train_cancel() -> TrainStatusResponse:
        if train_job.status != "running":
            raise HTTPException(status_code=409, detail="No running job to cancel")
        train_job._cancel.set()
        return train_job.to_response()

    @app.get("/api/train/models")
    async def train_list_models() -> list[ModelBundleResponse]:
        from taskclf.model_registry import list_bundles

        bundles = list_bundles(effective_models_dir)
        return [
            ModelBundleResponse(
                model_id=b.model_id,
                path=str(b.path),
                valid=b.valid,
                invalid_reason=b.invalid_reason,
                macro_f1=b.metrics.macro_f1 if b.metrics else None,
                weighted_f1=b.metrics.weighted_f1 if b.metrics else None,
                created_at=b.created_at.isoformat() if b.created_at else None,
            )
            for b in bundles
        ]

    @app.get("/api/train/models/current/inspect")
    async def train_current_model_bundle_inspect() -> dict[str, Any]:
        """Bundle-saved validation metrics for the model currently loaded in the tray."""
        if get_tray_state is None:
            return {"loaded": False, "reason": "tray_state_unavailable"}
        state = get_tray_state()
        md = state.get("model_dir")
        if not md:
            return {"loaded": False, "reason": "no_model_loaded"}
        path = Path(str(md)).resolve()
        if not path.is_dir():
            return {"loaded": False, "reason": "model_dir_missing"}
        try:
            payload = _bundle_inspect_bundle_only_payload(path)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            logger.warning("Bundle inspect failed for %s: %s", path, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Could not read bundle inspection: {exc}",
            ) from exc
        return {"loaded": True, **payload}

    @app.get("/api/train/models/{model_id}/inspect")
    async def train_model_bundle_inspect_by_id(model_id: str) -> dict[str, Any]:
        """Bundle-saved validation metrics for a known model bundle under models_dir."""
        from taskclf.model_registry import list_bundles

        bundles = list_bundles(effective_models_dir)
        target = next((b for b in bundles if b.model_id == model_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        if not target.valid:
            raise HTTPException(
                status_code=422,
                detail=target.invalid_reason or "invalid bundle",
            )
        try:
            return _bundle_inspect_bundle_only_payload(target.path)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            logger.warning("Bundle inspect failed for %s: %s", target.path, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Could not read bundle inspection: {exc}",
            ) from exc

    @app.get("/api/train/data-check")
    async def train_data_check(
        date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
        date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    ) -> DataCheckResponse:
        from taskclf.core.store import read_parquet
        from taskclf.features.build import build_features_for_date

        start = dt.date.fromisoformat(date_from)
        end = dt.date.fromisoformat(date_to)

        dates_built: list[str] = []
        build_errors: list[str] = []

        current = start
        while current <= end:
            fp = _feature_parquet_for_date(data_dir, current)
            if fp is None or not fp.exists():
                try:
                    build_features_for_date(
                        current,
                        data_dir,
                        aw_host=aw_host,
                        title_salt=title_salt,
                        user_id=user_config.user_id,
                    )
                    dates_built.append(current.isoformat())
                except Exception as exc:
                    build_errors.append(f"{current}: {exc}")
            current += dt.timedelta(days=1)

        current = start
        dates_with: list[str] = []
        dates_missing: list[str] = []
        total_rows = 0

        while current <= end:
            fp = _feature_parquet_for_date(data_dir, current)
            if fp is not None and fp.exists():
                try:
                    df = read_parquet(fp)
                    n = len(df)
                except Exception:
                    n = 0
                if n > 0:
                    dates_with.append(current.isoformat())
                    total_rows += n
                else:
                    dates_missing.append(current.isoformat())
            else:
                dates_missing.append(current.isoformat())
            current += dt.timedelta(days=1)

        label_count = 0
        matching_spans: list = []
        lp = data_dir / "labels_v1" / "labels.parquet"
        if lp.exists():
            try:
                spans = read_label_spans(lp)
                start_dt = dt.datetime(
                    start.year, start.month, start.day, tzinfo=dt.timezone.utc
                )
                end_dt = dt.datetime(
                    end.year, end.month, end.day, 23, 59, 59, tzinfo=dt.timezone.utc
                )
                matching_spans = [
                    s for s in spans if s.end_ts >= start_dt and s.start_ts <= end_dt
                ]
                label_count = len(matching_spans)
            except Exception:
                pass

        trainable_rows = 0
        trainable_labels: list[str] = []
        if total_rows > 0 and matching_spans:
            try:
                import pandas as pd
                from taskclf.labels.projection import project_blocks_to_windows

                feature_frames = []
                cur = start
                while cur <= end:
                    fp = _feature_parquet_for_date(data_dir, cur)
                    if fp is not None and fp.exists():
                        frame = read_parquet(fp)
                        if not frame.empty:
                            feature_frames.append(frame)
                    cur += dt.timedelta(days=1)

                if feature_frames:
                    features_df = pd.concat(feature_frames, ignore_index=True)
                    projected = project_blocks_to_windows(features_df, matching_spans)
                    trainable_rows = len(projected)
                    if not projected.empty and "label" in projected.columns:
                        trainable_labels = sorted(projected["label"].unique().tolist())
            except Exception:
                pass

        return DataCheckResponse(
            date_from=date_from,
            date_to=date_to,
            dates_with_features=dates_with,
            dates_missing_features=dates_missing,
            total_feature_rows=total_rows,
            label_span_count=label_count,
            trainable_rows=trainable_rows,
            trainable_labels=trainable_labels,
            dates_built=dates_built,
            build_errors=build_errors,
        )

    # -- WebSocket ------------------------------------------------------------

    @app.get("/api/ws/snapshot")
    async def ws_snapshot() -> dict[str, Any]:
        """Return the latest event for each type so reconnecting clients
        can hydrate their store without waiting for the next push."""
        return bus.snapshot()

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
        app.mount(
            "/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets"
        )

        @app.get("/{path:path}")
        async def spa_fallback(path: str) -> FileResponse:
            file_path = _STATIC_DIR / path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(_STATIC_DIR / "index.html")

    return app


def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable or *default* when unset."""
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive configuration error
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _create_dev_app_from_env() -> FastAPI:
    """Build a reloadable dev app from CLI-provided environment variables.

    Used by ``taskclf ui --dev --browser`` so uvicorn can own the backend
    process and restart it on Python source changes.
    """
    import os

    from taskclf.ui.copy import transition_suggestion_text
    from taskclf.ui.runtime import ActivityMonitor, _LabelSuggester
    from taskclf.ui.window import WindowAPI

    if os.environ.get("TASKCLF_UI_DEV") == "1":
        logging.getLogger("taskclf").setLevel(logging.DEBUG)

    data_dir = Path(os.environ.get("TASKCLF_UI_DATA_DIR", DEFAULT_DATA_DIR))
    user_config = UserConfig(data_dir)
    aw_host = os.environ.get("TASKCLF_AW_HOST", DEFAULT_AW_HOST)
    title_salt = os.environ.get("TASKCLF_TITLE_SALT") or user_config.title_secret
    poll_seconds = _env_int("TASKCLF_POLL_SECONDS", DEFAULT_POLL_SECONDS)
    transition_minutes = _env_int(
        "TASKCLF_TRANSITION_MINUTES",
        DEFAULT_TRANSITION_MINUTES,
    )
    model_dir_raw = os.environ.get("TASKCLF_MODEL_DIR")

    bus = EventBus()
    win_api = WindowAPI()

    suggester: _LabelSuggester | None = None
    if model_dir_raw:
        try:
            suggester = _LabelSuggester(Path(model_dir_raw))
            suggester._aw_host = aw_host
            suggester._title_salt = title_salt
        except Exception as exc:  # pragma: no cover - depends on model bundle
            logger.warning("Could not load dev suggester model: %s", exc)

    def on_transition(
        prev: str,
        new: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> None:
        suggested_label: str | None = None
        confidence: float | None = None
        if suggester is not None:
            result = suggester.suggest(start, end)
            if result is not None:
                suggested_label, confidence = result

        suggestion_text = None
        if suggested_label is not None:
            suggestion_text = transition_suggestion_text(
                suggested_label,
                start.astimezone().strftime("%H:%M"),
                end.astimezone().strftime("%H:%M"),
            )

        bus.publish_threadsafe(
            {
                "type": "prompt_label",
                "prev_app": prev,
                "new_app": new,
                "block_start": start.isoformat(),
                "block_end": end.isoformat(),
                "duration_min": max(1, int((end - start).total_seconds() / 60)),
                "suggested_label": suggested_label,
                "suggestion_text": suggestion_text,
            }
        )

        if suggested_label is not None and confidence is not None:
            bus.publish_threadsafe(
                {
                    "type": "suggest_label",
                    "reason": "app_switch",
                    "old_label": prev,
                    "suggested": suggested_label,
                    "confidence": confidence,
                    "block_start": start.isoformat(),
                    "block_end": end.isoformat(),
                }
            )
            return
        bus.publish_threadsafe(
            {
                "type": "prediction",
                "label": "unknown",
                "confidence": 0.0,
                "ts": end.isoformat(),
                "mapped_label": "unknown",
                "current_app": new,
            }
        )

    monitor = ActivityMonitor(
        aw_host=aw_host,
        title_salt=title_salt,
        poll_seconds=poll_seconds,
        transition_minutes=transition_minutes,
        on_transition=on_transition,
        event_bus=bus,
    )
    monitor_thread = threading.Thread(target=monitor.run, daemon=True)

    app = create_app(
        data_dir=data_dir,
        aw_host=aw_host,
        title_salt=title_salt,
        event_bus=bus,
        window_api=win_api,
        get_activity_provider_status=lambda: monitor.activity_provider_status,
    )

    @app.on_event("startup")
    async def _start_monitor() -> None:
        if not monitor_thread.is_alive():
            monitor_thread.start()

    @app.on_event("shutdown")
    async def _stop_monitor() -> None:
        monitor.stop()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=1)

    return app
