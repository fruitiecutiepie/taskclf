"""Asyncio event bus for broadcasting prediction and suggestion events to WebSocket clients."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe asyncio pub/sub for server-push events.

    The ``ActivityMonitor`` (running in a background thread) publishes
    events via :meth:`publish_threadsafe`; WebSocket handlers subscribe
    via :meth:`subscribe` and receive events as an async iterator.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

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

    async def publish(self, event: dict[str, Any]) -> None:
        """Broadcast *event* to all current subscribers.

        When a subscriber's queue is full, the oldest event is evicted
        so the subscriber keeps receiving new events (at the cost of
        missing stale ones).  The subscriber is never silently dropped.
        """
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
