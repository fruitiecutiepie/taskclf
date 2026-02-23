"""ActivityWatch data access: JSON export parsing and REST API client.

Provides two data-ingestion paths:

* **File-based** -- :func:`parse_aw_export` reads an AW JSON export
  (the format produced by *Export all buckets as JSON* in the AW web UI
  or ``GET /api/0/export``).
* **REST-based** -- :func:`fetch_aw_events` queries a running
  ``aw-server`` instance for events in a time range.

Both paths normalize application names via
:func:`~taskclf.adapters.activitywatch.mapping.normalize_app` and
replace raw window titles with salted hashes so that no sensitive text
is ever persisted.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taskclf.adapters.activitywatch.mapping import normalize_app
from taskclf.adapters.activitywatch.types import AWEvent, AWInputEvent
from taskclf.core.defaults import DEFAULT_AW_TIMEOUT_SECONDS
from taskclf.core.hashing import salted_hash

logger = logging.getLogger(__name__)

_CURRENTWINDOW_TYPE = "currentwindow"
_INPUT_TYPE = "os.hid.input"


def _parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp from AW into a naive-UTC datetime."""
    ts = datetime.fromisoformat(raw)
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _raw_event_to_aw_event(raw: dict[str, Any], *, title_salt: str) -> AWEvent:
    """Convert a single raw AW event dict into a normalized :class:`AWEvent`."""
    data = raw.get("data", {})
    app_name = data.get("app", "unknown")
    title = data.get("title", "")

    app_id, is_browser, is_editor, is_terminal, app_category = normalize_app(app_name)
    title_hash = salted_hash(title, salt=title_salt)

    return AWEvent(
        timestamp=_parse_timestamp(raw["timestamp"]),
        duration_seconds=float(raw.get("duration", 0)),
        app_id=app_id,
        window_title_hash=title_hash,
        is_browser=is_browser,
        is_editor=is_editor,
        is_terminal=is_terminal,
        app_category=app_category,
    )


# ---------------------------------------------------------------------------
# File-based ingestion
# ---------------------------------------------------------------------------


def parse_aw_export(path: Path, *, title_salt: str) -> list[AWEvent]:
    """Parse an ActivityWatch JSON export file into normalized events.

    Filters for buckets of type ``currentwindow`` (i.e.
    ``aw-watcher-window`` data).  Each event's application name is
    normalized and its window title is replaced with a salted hash.

    Args:
        path: Path to the AW export JSON file.
        title_salt: Salt used for hashing window titles.

    Returns:
        Sorted (by timestamp) list of :class:`AWEvent` instances.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the JSON structure is missing expected keys.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    buckets: dict[str, Any] = raw.get("buckets", raw)

    events: list[AWEvent] = []
    for bucket_id, bucket in buckets.items():
        bucket_type = bucket.get("type", "")
        if bucket_type != _CURRENTWINDOW_TYPE:
            logger.debug("Skipping bucket %s (type=%s)", bucket_id, bucket_type)
            continue

        logger.info(
            "Processing bucket %s (%d events)",
            bucket_id,
            len(bucket.get("events", [])),
        )
        for raw_event in bucket.get("events", []):
            events.append(_raw_event_to_aw_event(raw_event, title_salt=title_salt))

    events.sort(key=lambda e: e.timestamp)
    return events


def _raw_to_input_event(raw: dict[str, Any]) -> AWInputEvent:
    """Convert a single raw AW input event dict into an :class:`AWInputEvent`."""
    data = raw.get("data", {})
    return AWInputEvent(
        timestamp=_parse_timestamp(raw["timestamp"]),
        duration_seconds=float(raw.get("duration", 0)),
        presses=int(data.get("presses", 0)),
        clicks=int(data.get("clicks", 0)),
        delta_x=int(data.get("deltaX", 0)),
        delta_y=int(data.get("deltaY", 0)),
        scroll_x=int(data.get("scrollX", 0)),
        scroll_y=int(data.get("scrollY", 0)),
    )


def parse_aw_input_export(path: Path) -> list[AWInputEvent]:
    """Parse ``aw-watcher-input`` events from an AW JSON export.

    Filters for buckets of type ``os.hid.input``.  These events carry
    only aggregate counts (key presses, mouse clicks, movement, scroll)
    and contain no sensitive payload.

    Args:
        path: Path to the AW export JSON file.

    Returns:
        Sorted (by timestamp) list of :class:`AWInputEvent` instances.
        Empty if no ``os.hid.input`` bucket exists in the export.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[str, Any] = raw.get("buckets", raw)

    events: list[AWInputEvent] = []
    for bucket_id, bucket in buckets.items():
        bucket_type = bucket.get("type", "")
        if bucket_type != _INPUT_TYPE:
            continue

        logger.info(
            "Processing input bucket %s (%d events)",
            bucket_id,
            len(bucket.get("events", [])),
        )
        for raw_event in bucket.get("events", []):
            events.append(_raw_to_input_event(raw_event))

    events.sort(key=lambda e: e.timestamp)
    return events


# ---------------------------------------------------------------------------
# REST API helpers
# ---------------------------------------------------------------------------


def _api_get(url: str) -> Any:
    """Issue a GET request and return the parsed JSON body."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=DEFAULT_AW_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_aw_buckets(host: str) -> dict[str, dict]:
    """List all buckets from a running AW server.

    Args:
        host: Base URL of the AW server (e.g. ``"http://localhost:5600"``).

    Returns:
        Dict mapping bucket IDs to their metadata.
    """
    url = f"{host.rstrip('/')}/api/0/buckets/"
    return _api_get(url)


def find_window_bucket_id(host: str) -> str:
    """Auto-discover the ``aw-watcher-window`` bucket on *host*.

    Args:
        host: Base URL of the AW server.

    Returns:
        The bucket ID whose ``type`` is ``currentwindow``.

    Raises:
        ValueError: If no ``currentwindow`` bucket exists on the server.
    """
    buckets = list_aw_buckets(host)
    for bucket_id, meta in buckets.items():
        if meta.get("type") == _CURRENTWINDOW_TYPE:
            return bucket_id
    raise ValueError(
        f"No bucket with type={_CURRENTWINDOW_TYPE!r} found on {host}. "
        f"Available: {list(buckets.keys())}"
    )


def find_input_bucket_id(host: str) -> str | None:
    """Auto-discover the ``aw-watcher-input`` bucket on *host*.

    Unlike :func:`find_window_bucket_id`, this returns ``None`` when no
    input bucket exists because ``aw-watcher-input`` is an optional
    watcher that many users don't run.

    Args:
        host: Base URL of the AW server.

    Returns:
        The bucket ID whose ``type`` is ``os.hid.input``, or ``None``.
    """
    buckets = list_aw_buckets(host)
    for bucket_id, meta in buckets.items():
        if meta.get("type") == _INPUT_TYPE:
            return bucket_id
    return None


def fetch_aw_events(
    host: str,
    bucket_id: str,
    start: datetime,
    end: datetime,
    *,
    title_salt: str,
) -> list[AWEvent]:
    """Fetch events from the AW REST API for a time range.

    Args:
        host: Base URL of the AW server (e.g. ``"http://localhost:5600"``).
        bucket_id: Bucket to query (e.g. ``"aw-watcher-window_myhostname"``).
        start: Inclusive start of the query window (UTC).
        end: Exclusive end of the query window (UTC).
        title_salt: Salt used for hashing window titles.

    Returns:
        Sorted list of :class:`AWEvent` instances.
    """
    base = host.rstrip("/")
    start_iso = start.isoformat() + "Z" if start.tzinfo is None else start.isoformat()
    end_iso = end.isoformat() + "Z" if end.tzinfo is None else end.isoformat()

    url = (
        f"{base}/api/0/buckets/{bucket_id}/events"
        f"?start={start_iso}&end={end_iso}"
    )
    raw_events: list[dict] = _api_get(url)

    events = [
        _raw_event_to_aw_event(e, title_salt=title_salt) for e in raw_events
    ]
    events.sort(key=lambda e: e.timestamp)
    return events


def fetch_aw_input_events(
    host: str,
    bucket_id: str,
    start: datetime,
    end: datetime,
) -> list[AWInputEvent]:
    """Fetch input events from the AW REST API for a time range.

    Args:
        host: Base URL of the AW server.
        bucket_id: Input bucket to query (e.g.
            ``"aw-watcher-input_myhostname"``).
        start: Inclusive start of the query window (UTC).
        end: Exclusive end of the query window (UTC).

    Returns:
        Sorted list of :class:`AWInputEvent` instances.
    """
    base = host.rstrip("/")
    start_iso = start.isoformat() + "Z" if start.tzinfo is None else start.isoformat()
    end_iso = end.isoformat() + "Z" if end.tzinfo is None else end.isoformat()

    url = (
        f"{base}/api/0/buckets/{bucket_id}/events"
        f"?start={start_iso}&end={end_iso}"
    )
    raw_events: list[dict] = _api_get(url)

    events = [_raw_to_input_event(e) for e in raw_events]
    events.sort(key=lambda e: e.timestamp)
    return events
