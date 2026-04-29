"""Tests for the EventBus async pub/sub module.

Covers publish/subscribe delivery, multiple subscribers, dead subscriber
eviction, publish_threadsafe variants, and cleanup on context exit.
"""

from __future__ import annotations

import asyncio
from typing import Any


from taskclf.ui.events import EventBus


def _run(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


class TestEventBusPublishSubscribe:
    """TC-UI-EB-001: publish -> subscribe delivery."""

    def test_publish_delivers_to_subscriber(self) -> None:
        async def _test() -> dict[str, Any]:
            bus = EventBus()
            event = {"type": "status", "state": "idle"}
            async with bus.subscribe() as q:
                await bus.publish(event)
                received = q.get_nowait()
            return received

        result = _run(_test())
        assert result == {"type": "status", "state": "idle"}


class TestEventBusMultipleSubscribers:
    """TC-UI-EB-002: multiple subscribers each receive published event."""

    def test_multiple_subscribers(self) -> None:
        async def _test() -> tuple[dict, dict]:
            bus = EventBus()
            event = {"type": "prediction", "label": "Build"}
            async with bus.subscribe() as q1, bus.subscribe() as q2:
                await bus.publish(event)
                return q1.get_nowait(), q2.get_nowait()

        r1, r2 = _run(_test())
        assert r1 == r2 == {"type": "prediction", "label": "Build"}


class TestEventBusBackpressure:
    """TC-UI-EB-003: full queue evicts oldest event, subscriber stays."""

    def test_subscriber_retained_after_overflow(self) -> None:
        async def _test() -> tuple[int, int]:
            bus = EventBus()
            async with bus.subscribe() as q:
                for i in range(256):
                    await bus.publish({"i": i})
                assert q.full()
                await bus.publish({"i": 256})
                inside = len(bus._subscribers)
            after = len(bus._subscribers)
            return inside, after

        inside, after = _run(_test())
        assert inside == 1  # subscriber retained despite overflow
        assert after == 0  # cleaned up on context exit

    def test_newest_event_always_in_queue(self) -> None:
        """After overflow, the most recent event is always present."""

        async def _test() -> dict[str, Any]:
            bus = EventBus()
            async with bus.subscribe() as q:
                for i in range(300):
                    await bus.publish({"i": i})
                items = []
                while not q.empty():
                    items.append(q.get_nowait())
                return items[-1]

        last = _run(_test())
        assert last["i"] == 299

    def test_queue_stays_full_after_overflow(self) -> None:
        """Queue contains exactly 256 events after 300 publishes."""

        async def _test() -> int:
            bus = EventBus()
            async with bus.subscribe() as q:
                for i in range(300):
                    await bus.publish({"i": i})
                return q.qsize()

        size = _run(_test())
        assert size == 256

    def test_subscriber_still_receives_after_overflow(self) -> None:
        """A subscriber that overflowed still gets subsequent events."""

        async def _test() -> dict[str, Any]:
            bus = EventBus()
            async with bus.subscribe() as q:
                for i in range(260):
                    await bus.publish({"i": i})
                assert len(bus._subscribers) == 1
                while not q.empty():
                    q.get_nowait()
                await bus.publish({"type": "after_overflow"})
                return q.get_nowait()

        result = _run(_test())
        assert result["type"] == "after_overflow"


class TestPublishThreadsafeBoundLoop:
    """TC-UI-EB-004: publish_threadsafe with a bound loop delivers event."""

    def test_threadsafe_delivery(self) -> None:
        async def _test() -> dict[str, Any]:
            bus = EventBus()
            bus.bind_loop(asyncio.get_running_loop())
            event = {"type": "tray_state", "model_loaded": True}
            async with bus.subscribe() as q:
                bus.publish_threadsafe(event)
                await asyncio.sleep(0.05)
                return q.get_nowait()

        result = _run(_test())
        assert result["type"] == "tray_state"


class TestPublishThreadsafeNoLoop:
    """TC-UI-EB-005: publish_threadsafe with no loop bound -> no-op, no error."""

    def test_no_loop_no_error(self) -> None:
        bus = EventBus()
        bus.publish_threadsafe({"type": "status"})


class TestPublishThreadsafeClosedLoop:
    """TC-UI-EB-006: publish_threadsafe with closed loop -> no-op, no error."""

    def test_closed_loop_no_error(self) -> None:
        loop = asyncio.new_event_loop()
        bus = EventBus()
        bus.bind_loop(loop)
        loop.close()
        bus.publish_threadsafe({"type": "status"})


class TestSubscriberCleanupOnExit:
    """TC-UI-EB-007: exiting subscribe() context removes queue from _subscribers."""

    def test_cleanup_on_exit(self) -> None:
        async def _test() -> int:
            bus = EventBus()
            async with bus.subscribe():
                assert len(bus._subscribers) == 1
            return len(bus._subscribers)

        remaining = _run(_test())
        assert remaining == 0


class TestSnapshot:
    """TC-UI-EB-008: snapshot() returns latest event per type."""

    def test_snapshot_empty_initially(self) -> None:
        bus = EventBus()
        assert bus.snapshot() == {}

    def test_snapshot_retains_latest(self) -> None:
        async def _test() -> dict[str, dict[str, Any]]:
            bus = EventBus()
            await bus.publish({"type": "status", "state": "idle"})
            await bus.publish({"type": "status", "state": "collecting"})
            await bus.publish({"type": "prediction", "label": "Build"})
            return bus.snapshot()

        snap = _run(_test())
        assert snap["status"]["state"] == "collecting"
        assert snap["prediction"]["label"] == "Build"

    def test_snapshot_is_copy(self) -> None:
        async def _test() -> tuple[dict, dict]:
            bus = EventBus()
            await bus.publish({"type": "status", "state": "idle"})
            s1 = bus.snapshot()
            s1["status"]["state"] = "mutated"
            s2 = bus.snapshot()
            return s1, s2

        s1, s2 = _run(_test())
        assert s1["status"]["state"] == "mutated"
        assert s2["status"]["state"] == "idle"

    def test_snapshot_threadsafe(self) -> None:
        """publish_threadsafe updates are reflected in snapshot."""

        async def _test() -> dict[str, dict[str, Any]]:
            bus = EventBus()
            bus.bind_loop(asyncio.get_running_loop())
            bus.publish_threadsafe({"type": "tray_state", "paused": True})
            await asyncio.sleep(0.05)
            return bus.snapshot()

        snap = _run(_test())
        assert snap["tray_state"]["paused"] is True

    def test_snapshot_retains_pending_suggestions(self) -> None:
        async def _test() -> dict[str, dict[str, Any]]:
            bus = EventBus()
            await bus.publish(
                {
                    "type": "suggest_label",
                    "suggestion_id": "first",
                    "reason": "app_switch",
                    "old_label": "Write",
                    "suggested": "Review",
                    "confidence": 0.93,
                    "block_start": "2026-04-05T09:30:00Z",
                    "block_end": "2026-04-05T10:00:00Z",
                }
            )
            await bus.publish(
                {
                    "type": "suggest_label",
                    "suggestion_id": "second",
                    "reason": "app_switch",
                    "old_label": "Review",
                    "suggested": "Build",
                    "confidence": 0.88,
                    "block_start": "2026-04-05T10:00:00Z",
                    "block_end": "2026-04-05T10:30:00Z",
                }
            )
            await bus.publish(
                {
                    "type": "suggestion_cleared",
                    "reason": "skipped",
                    "suggestion_id": "first",
                }
            )
            return bus.snapshot()

        snap = _run(_test())
        pending = snap["pending_suggestions"]["suggestions"]
        assert [item["suggestion_id"] for item in pending] == ["second"]
        assert pending[0]["suggested"] == "Build"
