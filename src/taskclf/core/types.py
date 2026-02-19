from __future__ import annotations

# TODO: Event, LabelSpan pydantic models

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

_PROHIBITED_FIELD_PREFIXES = ("raw_",)


class FeatureRow(BaseModel, frozen=True):
    """One bucketed observation (typically 60 s).

    All persisted feature rows carry schema metadata so downstream
    consumers can detect silent drift.
    """

    # -- meta --
    bucket_start_ts: datetime
    schema_version: str
    schema_hash: str
    source_ids: list[str] = Field(min_length=1)

    # -- context --
    app_id: str
    window_title_hash: str
    is_browser: bool
    is_editor: bool
    is_terminal: bool
    app_switch_count_last_5m: int = Field(ge=0)

    # -- keyboard (nullable until collector is wired) --
    keys_per_min: float | None = None
    backspace_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    shortcut_rate: float | None = Field(default=None, ge=0.0)

    # -- mouse (nullable until collector is wired) --
    clicks_per_min: float | None = Field(default=None, ge=0.0)
    scroll_events_per_min: float | None = Field(default=None, ge=0.0)
    mouse_distance: float | None = Field(default=None, ge=0.0)

    # -- temporal --
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    session_length_so_far: float = Field(ge=0.0)

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
