# ui.events

Thread-safe asyncio event bus for broadcasting prediction and
suggestion events to WebSocket clients.

## Overview

`EventBus` bridges the gap between synchronous background threads
(like `ActivityMonitor`) and the async FastAPI WebSocket layer.  It
provides a pub/sub mechanism where publishers push events and
subscribers receive them via asyncio queues.
`EventBus` is implemented as a slotted dataclass; public method behavior
is unchanged.

```
ActivityMonitor (thread) → publish_threadsafe → EventBus
                                                   ↓
                                              subscribe (async)
                                                   ↓
                                          WebSocket /ws/predictions
```

## EventBus

### Methods

| Method | Sync/Async | Description |
|--------|------------|-------------|
| `bind_loop(loop)` | sync | Bind to the running asyncio event loop (call once at startup) |
| `publish(event)` | async | Broadcast an event dict to all current subscribers |
| `publish_threadsafe(event)` | sync | Schedule a publish from a non-async thread |
| `subscribe()` | async context manager | Yields an `asyncio.Queue` receiving all published events |
| `snapshot()` | sync (thread-safe) | Return latest event per type for reconnecting client hydration |

### bind_loop

```python
bind_loop(loop: asyncio.AbstractEventLoop) -> None
```

Must be called once at startup to associate the bus with the running
event loop.  Without a bound loop, `publish_threadsafe` silently
drops events.

### publish

```python
async publish(event: dict[str, Any]) -> None
```

Broadcasts `event` to every subscriber queue.  If a subscriber's
queue is full (capacity exceeded), that subscriber is evicted from
the subscriber set on the same call.

### publish_threadsafe

```python
publish_threadsafe(event: dict[str, Any]) -> None
```

Schedules a `publish` coroutine on the bound event loop via
`asyncio.run_coroutine_threadsafe`.  Safe to call from any thread
(e.g. `ActivityMonitor`).  No-ops silently when no loop is bound or
the loop is closed.

### snapshot

```python
snapshot() -> dict[str, dict[str, Any]]
```

Returns a copy of the most recent event for each known event type.
Thread-safe (guarded by a `threading.Lock`).  Used by
`GET /api/ws/snapshot` so that reconnecting WebSocket clients can
hydrate their store immediately instead of waiting for the next push.

### subscribe

```python
@asynccontextmanager
async subscribe() -> AsyncIterator[asyncio.Queue[dict[str, Any]]]
```

Context manager that creates a queue (capacity 256), adds it to the
subscriber set, yields it, and removes it on exit.  Typical usage is
inside a WebSocket handler that reads from the queue in a loop.

## Queue behaviour

- Each subscriber gets an independent `asyncio.Queue` with
  `maxsize=256`.
- When a queue is full, the subscriber is evicted on the next
  `publish` call (the overflowing event is dropped for that
  subscriber).
- On context-manager exit, the queue is removed from the subscriber
  set regardless of whether it was previously evicted.

## Usage

```python
import asyncio
from taskclf.ui.events import EventBus

bus = EventBus()

async def main():
    bus.bind_loop(asyncio.get_running_loop())

    async with bus.subscribe() as queue:
        bus.publish_threadsafe({"type": "status", "state": "idle"})
        await asyncio.sleep(0.05)
        event = queue.get_nowait()
        print(event)  # {"type": "status", "state": "idle"}

asyncio.run(main())
```

See [`ui.server`](labeling.md) for the WebSocket endpoint that
consumes events, and [`ui.window`](window.md) for the native window
integration.

::: taskclf.ui.events
