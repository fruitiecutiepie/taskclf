"""Core data contracts: Event protocol, feature rows, and label spans."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


@runtime_checkable
class Event(Protocol):
    """Normalized activity event contract.

    Any adapter event type that exposes these read-only attributes
    satisfies the protocol -- no inheritance required.  For example,
    :class:`~taskclf.adapters.activitywatch.types.AWEvent` is a valid
    ``Event`` without importing or subclassing this protocol.
    """

    @property
    def timestamp(self) -> datetime: ...
    @property
    def duration_seconds(self) -> float: ...
    @property
    def app_id(self) -> str: ...
    @property
    def window_title_hash(self) -> str: ...
    @property
    def is_browser(self) -> bool: ...
    @property
    def is_editor(self) -> bool: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def app_category(self) -> str: ...


class CoreLabel(StrEnum):
    """Canonical task-type labels (v1).

    Member ordering matches ``schema/labels_v1.json`` label IDs.
    Do NOT reorder or remove members without a version bump.
    """

    Build = "Build"
    Debug = "Debug"
    Review = "Review"
    Write = "Write"
    ReadResearch = "ReadResearch"
    Communicate = "Communicate"
    Meet = "Meet"
    BreakIdle = "BreakIdle"


LABEL_SET_V1: Final[frozenset[str]] = frozenset(CoreLabel)

_PROHIBITED_FIELD_PREFIXES = ("raw_",)


class FeatureRow(BaseModel, frozen=True):
    """One bucketed observation (typically 60 s).

    All persisted feature rows carry schema metadata so downstream
    consumers can detect silent drift.

    Fields are grouped into four sections:

    - **meta** — ``bucket_start_ts``, ``schema_version``, ``schema_hash``,
      ``source_ids``.
    - **context** — ``app_id``, ``app_category``, ``window_title_hash``,
      ``is_browser``, ``is_editor``, ``is_terminal``,
      ``app_switch_count_last_5m``, ``app_foreground_time_ratio``,
      ``app_change_count``.
    - **keyboard / mouse** — nullable until the corresponding collector is
      wired (``keys_per_min``, ``backspace_ratio``, ``shortcut_rate``,
      ``clicks_per_min``, ``scroll_events_per_min``, ``mouse_distance``).
    - **activity occupancy** — nullable; derived from ``aw-watcher-input``
      (``active_seconds_keyboard``, ``active_seconds_mouse``,
      ``active_seconds_any``, ``max_idle_run_seconds``, ``event_density``).
    - **temporal** — ``hour_of_day``, ``day_of_week``, ``session_length_so_far``.

    A pre-validator rejects any field whose name starts with ``raw_`` to
    enforce the privacy invariant (no raw keystrokes / titles).
    """

    # -- meta --
    bucket_start_ts: datetime = Field(description="Start of the 60 s bucket (UTC).")
    schema_version: str = Field(description="Schema version tag, e.g. 'v1'.")
    schema_hash: str = Field(description="Deterministic hash of the column registry.")
    source_ids: list[str] = Field(min_length=1, description="Collector IDs that contributed to this row.")

    # -- context --
    app_id: str = Field(description="Reverse-domain app identifier, e.g. 'com.apple.Terminal'.")
    app_category: str = Field(description="Semantic app category, e.g. 'editor', 'chat', 'meeting'.")
    window_title_hash: str = Field(description="Hashed window title (never raw).")
    is_browser: bool = Field(description="True if the foreground app is a web browser.")
    is_editor: bool = Field(description="True if the foreground app is a code editor.")
    is_terminal: bool = Field(description="True if the foreground app is a terminal emulator.")
    app_switch_count_last_5m: int = Field(ge=0, description="Number of unique app switches in the last 5 minutes.")
    app_foreground_time_ratio: float = Field(ge=0.0, le=1.0, description="Fraction of the bucket the dominant app was foreground.")
    app_change_count: int = Field(ge=0, description="Number of app transitions within this bucket.")

    # -- keyboard (nullable until collector is wired) --
    keys_per_min: float | None = Field(default=None, description="Keystrokes per minute.")
    backspace_ratio: float | None = Field(default=None, ge=0.0, le=1.0, description="Fraction of keystrokes that are backspace.")
    shortcut_rate: float | None = Field(default=None, ge=0.0, description="Keyboard shortcuts per minute.")

    # -- mouse (nullable until collector is wired) --
    clicks_per_min: float | None = Field(default=None, ge=0.0, description="Mouse clicks per minute.")
    scroll_events_per_min: float | None = Field(default=None, ge=0.0, description="Scroll events per minute.")
    mouse_distance: float | None = Field(default=None, ge=0.0, description="Mouse distance in pixels.")

    # -- activity occupancy (nullable until input collector is wired) --
    active_seconds_keyboard: float | None = Field(default=None, ge=0.0, description="Seconds with keyboard activity within this bucket.")
    active_seconds_mouse: float | None = Field(default=None, ge=0.0, description="Seconds with mouse activity within this bucket.")
    active_seconds_any: float | None = Field(default=None, ge=0.0, description="Seconds with any input activity within this bucket.")
    max_idle_run_seconds: float | None = Field(default=None, ge=0.0, description="Longest consecutive idle run (seconds) within this bucket.")
    event_density: float | None = Field(default=None, ge=0.0, description="Input events per active second within this bucket.")

    # -- temporal --
    hour_of_day: int = Field(ge=0, le=23, description="Hour component of bucket_start_ts (0-23).")
    day_of_week: int = Field(ge=0, le=6, description="Day of week (0=Monday, 6=Sunday).")
    session_length_so_far: float = Field(ge=0.0, description="Minutes since session start.")

    @model_validator(mode="before")
    @classmethod
    def reject_prohibited_fields(cls, values: dict) -> dict:  # type: ignore[override]
        if isinstance(values, dict):
            for key in values:
                for prefix in _PROHIBITED_FIELD_PREFIXES:
                    if key.startswith(prefix):
                        raise ValueError(
                            f"Prohibited field '{key}': fields starting with "
                            f"'{prefix}' must not appear in a FeatureRow"
                        )
        return values


class LabelSpan(BaseModel, frozen=True):
    """A contiguous time span carrying a single task-type label.

    Gold labels and weak labels share this structure; ``provenance``
    distinguishes them (e.g. ``"manual"`` vs ``"weak:app_rule"``).
    """

    start_ts: datetime = Field(description="Span start (UTC, inclusive).")
    end_ts: datetime = Field(description="Span end (UTC, exclusive).")
    label: str = Field(description="Task-type label from LABEL_SET_V1.")
    provenance: str = Field(description="Origin tag, e.g. 'manual' or 'weak:app_rule'.")

    @model_validator(mode="after")
    def _check_invariants(self) -> LabelSpan:
        if self.end_ts <= self.start_ts:
            raise ValueError(
                f"end_ts ({self.end_ts}) must be strictly after "
                f"start_ts ({self.start_ts})"
            )
        if self.label not in LABEL_SET_V1:
            raise ValueError(
                f"Unknown label {self.label!r}; "
                f"must be one of {sorted(LABEL_SET_V1)}"
            )
        return self
