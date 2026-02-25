"""Privacy-safe normalized event types for ActivityWatch data."""

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
    app_category: str = Field(description="Semantic app category (e.g. 'editor', 'chat').")
class AWInputEvent(BaseModel, frozen=True):
    """Aggregated keyboard/mouse activity from ``aw-watcher-input``.    Each event covers a short polling interval (typically 5 s) and
    carries only aggregate counts -- never individual key identities.
    This makes the type privacy-safe by construction.    The upstream AW fields ``deltaX``/``deltaY`` and ``scrollX``/``scrollY``
    are mapped to snake_case for consistency with project conventions.
    """

    timestamp: datetime = Field(description="Interval start (UTC).")
    duration_seconds: float = Field(ge=0, description="Duration in seconds.")
    presses: int = Field(ge=0, description="Key presses in this interval.")
    clicks: int = Field(ge=0, description="Mouse clicks in this interval.")
    delta_x: int = Field(ge=0, description="Absolute horizontal mouse movement (px).")
    delta_y: int = Field(ge=0, description="Absolute vertical mouse movement (px).")
    scroll_x: int = Field(ge=0, description="Absolute horizontal scroll delta.")
    scroll_y: int = Field(ge=0, description="Absolute vertical scroll delta.")
