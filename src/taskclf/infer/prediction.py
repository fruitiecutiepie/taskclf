"""Structured prediction output for a single time-window inference."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class WindowPrediction(BaseModel, frozen=True):
    """Full inference output for one time bucket.

    Matches the contract defined in ``docs/guide/model_io.md`` section 6.
    Every field is populated regardless of whether a taxonomy or
    calibrator is active; when no taxonomy is configured the mapped
    fields mirror the core prediction.
    """

    user_id: str
    bucket_start_ts: datetime
    core_label_id: int = Field(ge=0, le=7)
    core_label_name: str
    core_probs: list[float] = Field(min_length=8, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    is_rejected: bool
    mapped_label_name: str
    mapped_probs: dict[str, float]
    model_version: str
    schema_version: str = Field(default="features_v1")
    label_version: str = Field(default="labels_v1")

    @model_validator(mode="after")
    def _check_probs(self) -> WindowPrediction:
        total = sum(self.core_probs)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"core_probs must sum to 1.0, got {total:.6f}"
            )
        mapped_total = sum(self.mapped_probs.values())
        if abs(mapped_total - 1.0) > 1e-4:
            raise ValueError(
                f"mapped_probs must sum to 1.0, got {mapped_total:.6f}"
            )
        return self
