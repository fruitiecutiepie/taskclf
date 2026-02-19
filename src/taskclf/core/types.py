"""Pydantic models for the core data contracts: feature rows and label spans."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from pydantic import BaseModel, Field, model_validator

LABEL_SET_V1: Final[frozenset[str]] = frozenset({
    "coding",
    "writing_docs",
    "messaging_email",
    "browsing_research",
    "meetings_calls",
    "break_idle",
})

_PROHIBITED_FIELD_PREFIXES = ("raw_",)


class FeatureRow(BaseModel, frozen=True):
    """One bucketed observation (typically 60 s).

    All persisted feature rows carry schema metadata so downstream
    consumers can detect silent drift.

    Fields are grouped into four sections:

    - **meta** — ``bucket_start_ts``, ``schema_version``, ``schema_hash``,
      ``source_ids``.
    - **context** — ``app_id``, ``window_title_hash``, ``is_browser``,
      ``is_editor``, ``is_terminal``, ``app_switch_count_last_5m``.
    - **keyboard / mouse** — nullable until the corresponding collector is
      wired (``keys_per_min``, ``backspace_ratio``, ``shortcut_rate``,
      ``clicks_per_min``, ``scroll_events_per_min``, ``mouse_distance``).
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
    window_title_hash: str = Field(description="Hashed window title (never raw).")
    is_browser: bool = Field(description="True if the foreground app is a web browser.")
    is_editor: bool = Field(description="True if the foreground app is a code editor.")
    is_terminal: bool = Field(description="True if the foreground app is a terminal emulator.")
    app_switch_count_last_5m: int = Field(ge=0, description="Number of app switches in the last 5 minutes.")

    # -- keyboard (nullable until collector is wired) --
    keys_per_min: float | None = Field(default=None, description="Keystrokes per minute.")
    backspace_ratio: float | None = Field(default=None, ge=0.0, le=1.0, description="Fraction of keystrokes that are backspace.")
    shortcut_rate: float | None = Field(default=None, ge=0.0, description="Keyboard shortcuts per minute.")

    # -- mouse (nullable until collector is wired) --
    clicks_per_min: float | None = Field(default=None, ge=0.0, description="Mouse clicks per minute.")
    scroll_events_per_min: float | None = Field(default=None, ge=0.0, description="Scroll events per minute.")
    mouse_distance: float | None = Field(default=None, ge=0.0, description="Mouse distance in pixels.")

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
