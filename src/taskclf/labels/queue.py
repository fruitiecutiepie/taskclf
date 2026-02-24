"""Active labeling queue: prioritise windows/blocks that need human labels.

The queue tracks buckets flagged for labeling (low model confidence or
detected drift) and enforces a daily ask limit so users are not
overwhelmed.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.defaults import (
    DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
    DEFAULT_LABEL_MAX_ASKS_PER_DAY,
)


class LabelRequest(BaseModel, frozen=True):
    """A single item in the active labeling queue."""

    request_id: str = Field(description="Unique request identifier (UUID).")
    user_id: str = Field(description="User whose bucket needs labeling.")
    bucket_start_ts: datetime = Field(description="Start of the bucket (UTC).")
    bucket_end_ts: datetime = Field(description="End of the bucket (UTC, exclusive).")
    reason: Literal["low_confidence", "drift"] = Field(
        description="Why this bucket was enqueued."
    )
    confidence: float | None = Field(
        default=None, description="Model confidence at time of enqueue."
    )
    predicted_label: str | None = Field(
        default=None, description="Model prediction at time of enqueue."
    )
    created_at: datetime = Field(description="When the request was created (UTC).")
    status: Literal["pending", "labeled", "skipped"] = Field(
        default="pending", description="Current lifecycle state."
    )


class ActiveLabelingQueue:
    """Manages a persisted queue of labeling requests.

    State lives in a single JSON file; mutations are atomic
    (write-to-temp then rename).

    Args:
        queue_path: Path to the JSON file backing the queue.
        max_asks_per_day: Upper bound on pending items served per
            calendar day (UTC).
    """

    def __init__(
        self,
        queue_path: Path,
        max_asks_per_day: int = DEFAULT_LABEL_MAX_ASKS_PER_DAY,
    ) -> None:
        self._path = queue_path
        self._max_asks = max_asks_per_day
        self._items: list[LabelRequest] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text())
        self._items = [LabelRequest.model_validate(r) for r in raw]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [r.model_dump(mode="json") for r in self._items],
            indent=2,
            default=str,
        )
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            os.write(fd, payload.encode())
            os.close(fd)
            os.replace(tmp, str(self._path))
        except BaseException:
            os.close(fd) if not os.get_inheritable(fd) else None  # pragma: no cover
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _bucket_key(self, user_id: str, bucket_start_ts: datetime) -> tuple[str, str]:
        return (user_id, bucket_start_ts.isoformat())

    def _existing_keys(self) -> set[tuple[str, str]]:
        return {
            self._bucket_key(r.user_id, r.bucket_start_ts)
            for r in self._items
            if r.status == "pending"
        }

    def enqueue_low_confidence(
        self,
        predictions_df: pd.DataFrame,
        threshold: float = DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
    ) -> int:
        """Add buckets whose model confidence is below *threshold*.

        *predictions_df* must have columns: ``user_id``,
        ``bucket_start_ts``, ``bucket_end_ts``, ``confidence``,
        ``predicted_label``.

        Returns the number of newly enqueued items.
        """
        existing = self._existing_keys()
        added = 0
        now = datetime.now(tz=timezone.utc)

        for _, row in predictions_df.iterrows():
            if row["confidence"] >= threshold:
                continue
            key = self._bucket_key(row["user_id"], pd.Timestamp(row["bucket_start_ts"]).to_pydatetime())
            if key in existing:
                continue
            req = LabelRequest(
                request_id=str(uuid.uuid4()),
                user_id=row["user_id"],
                bucket_start_ts=pd.Timestamp(row["bucket_start_ts"]).to_pydatetime(),
                bucket_end_ts=pd.Timestamp(row["bucket_end_ts"]).to_pydatetime(),
                reason="low_confidence",
                confidence=float(row["confidence"]),
                predicted_label=str(row["predicted_label"]),
                created_at=now,
                status="pending",
            )
            self._items.append(req)
            existing.add(key)
            added += 1

        if added:
            self._save()
        return added

    def enqueue_drift(
        self,
        buckets: Sequence[dict],
    ) -> int:
        """Add drift-flagged buckets.

        Each dict in *buckets* must contain ``user_id``,
        ``bucket_start_ts``, ``bucket_end_ts``, and optionally
        ``predicted_label`` and ``confidence``.

        Returns the number of newly enqueued items.
        """
        existing = self._existing_keys()
        added = 0
        now = datetime.now(tz=timezone.utc)

        for b in buckets:
            ts = pd.Timestamp(b["bucket_start_ts"]).to_pydatetime()
            key = self._bucket_key(b["user_id"], ts)
            if key in existing:
                continue
            req = LabelRequest(
                request_id=str(uuid.uuid4()),
                user_id=b["user_id"],
                bucket_start_ts=ts,
                bucket_end_ts=pd.Timestamp(b["bucket_end_ts"]).to_pydatetime(),
                reason="drift",
                confidence=b.get("confidence"),
                predicted_label=b.get("predicted_label"),
                created_at=now,
                status="pending",
            )
            self._items.append(req)
            existing.add(key)
            added += 1

        if added:
            self._save()
        return added

    def get_pending(
        self,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[LabelRequest]:
        """Return pending items, respecting the daily ask cap.

        Items are sorted by confidence ascending (lowest first) so the
        most uncertain buckets surface first.

        Args:
            user_id: Filter to a specific user (``None`` = all users).
            limit: Maximum items to return (capped by daily limit).

        Returns:
            List of pending ``LabelRequest`` instances.
        """
        pending = [r for r in self._items if r.status == "pending"]
        if user_id is not None:
            pending = [r for r in pending if r.user_id == user_id]

        pending.sort(key=lambda r: (r.confidence if r.confidence is not None else 0.0))

        today = datetime.now(tz=timezone.utc).date()
        served_today = sum(
            1
            for r in self._items
            if r.status in ("labeled", "skipped")
            and r.created_at.date() == today
        )
        daily_remaining = max(0, self._max_asks - served_today)

        cap = daily_remaining
        if limit is not None:
            cap = min(cap, limit)

        return pending[:cap]

    def mark_done(
        self,
        request_id: str,
        status: Literal["labeled", "skipped"] = "labeled",
    ) -> LabelRequest | None:
        """Transition a request to *status*.

        Returns the updated request, or ``None`` if *request_id* was
        not found.
        """
        for i, r in enumerate(self._items):
            if r.request_id == request_id:
                updated = r.model_copy(update={"status": status})
                self._items[i] = updated
                self._save()
                return updated
        return None

    @property
    def all_items(self) -> list[LabelRequest]:
        """All items currently in the queue (any status)."""
        return list(self._items)
