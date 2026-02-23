"""Privacy-safe normalized event type for ActivityWatch data."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AWEvent(BaseModel, frozen=True):
    """A single ActivityWatch window event, normalized and privacy-scrubbed.

    Raw ``app`` names are mapped to reverse-domain identifiers via
    :func:`~taskclf.adapters.activitywatch.mapping.normalize_app`.
    Raw ``title`` strings are replaced with a salted hash via
    :func:`~taskclf.core.hashing.salted_hash` -- the original title
    is never persisted.
    """

    timestamp: datetime = Field(description="Event start (UTC).")
    duration_seconds: float = Field(ge=0, description="Duration in seconds.")
    app_id: str = Field(description="Reverse-domain app identifier.")
    window_title_hash: str = Field(description="Salted SHA-256 of the window title.")
    is_browser: bool = Field(description="True if the app is a web browser.")
    is_editor: bool = Field(description="True if the app is a code editor.")
    is_terminal: bool = Field(description="True if the app is a terminal emulator.")
