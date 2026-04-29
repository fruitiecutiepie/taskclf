"""Asyncio event bus for broadcasting prediction and suggestion events to WebSocket clients."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class EventBus:
    """Thread-safe asyncio pub/sub for server-push events.

    The ``ActivityMonitor`` (running in a background thread) publishes
    events via :meth:`publish_threadsafe`; WebSocket handlers subscribe
    via :meth:`subscribe` and receive events as an async iterator.

    The bus also retains the most recent event of each ``type`` so that
    newly-connected (or reconnecting) clients can hydrate their state
    immediately via :meth:`snapshot`.
    """

    _subscribers: set[asyncio.Queue[dict[str, Any]]] = field(
        init=False, default_factory=set
    )
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None)
    _ready: threading.Event = field(init=False, default_factory=threading.Event)
    _latest: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    _pending_suggestions: dict[str, dict[str, Any]] = field(
        init=False, default_factory=dict
    )
    _latest_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the running event loop (call once at startup)."""
        self._loop = loop
        self._ready.set()

    def wait_ready(self, timeout: float = 30) -> bool:
        """Block until :meth:`bind_loop` has been called.

        Returns ``True`` if the loop was bound within *timeout* seconds.
        Safe to call from any thread.
        """
        return self._ready.wait(timeout=timeout)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return the latest event for each known type (thread-safe).

        Used by the REST hydration endpoint so reconnecting WebSocket
        clients can immediately recover current state.
        """
        with self._latest_lock:
            return copy.deepcopy(self._latest)

    def _suggestion_key_get(self, event: dict[str, Any]) -> str | None:
        suggestion_id = event.get("suggestion_id")
        if isinstance(suggestion_id, str) and suggestion_id:
            return suggestion_id
        block_start = event.get("block_start")
        block_end = event.get("block_end")
        if isinstance(block_start, str) and isinstance(block_end, str):
            return f"{block_start}|{block_end}"
        return None

    def _pending_suggestions_snapshot_get(self) -> dict[str, Any]:
        suggestions = sorted(
            self._pending_suggestions.values(),
            key=lambda item: (
                str(item.get("block_start", "")),
                str(item.get("block_end", "")),
                str(item.get("suggestion_id", "")),
            ),
        )
        return {
            "type": "pending_suggestions",
            "suggestions": [copy.deepcopy(item) for item in suggestions],
        }

    async def publish(self, event: dict[str, Any]) -> None:
        """Broadcast *event* to all current subscribers.

        When a subscriber's queue is full, the oldest event is evicted
        so the subscriber keeps receiving new events (at the cost of
        missing stale ones).  The subscriber is never silently dropped.
        """
        event_type = event.get("type")
        if event_type:
            with self._latest_lock:
                if event_type == "suggest_label":
                    suggestion_key = self._suggestion_key_get(event)
                    if suggestion_key is not None:
                        self._pending_suggestions[suggestion_key] = event
                        self._latest["pending_suggestions"] = (
                            self._pending_suggestions_snapshot_get()
                        )
                elif event_type == "suggestion_cleared":
                    suggestion_key = self._suggestion_key_get(event)
                    if suggestion_key is not None:
                        self._pending_suggestions.pop(suggestion_key, None)
                    else:
                        self._pending_suggestions.clear()
                    self._latest["pending_suggestions"] = (
                        self._pending_suggestions_snapshot_get()
                    )
                self._latest[event_type] = event
        async with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("EventBus: failed to enqueue after eviction")

    @property
    def has_subscribers(self) -> bool:
        """Return ``True`` if at least one WebSocket client is subscribed."""
        return bool(self._subscribers)

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Schedule a publish from a non-async thread (e.g. ``ActivityMonitor``)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Context manager that yields a queue receiving all published events."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(q)
        try:
            yield q
        finally:
            async with self._lock:
                self._subscribers.discard(q)
