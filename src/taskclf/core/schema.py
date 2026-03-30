"""Feature schema versioning, deterministic hashing, and DataFrame validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Final

import pandas as pd

from taskclf.core.hashing import stable_hash
from taskclf.core.types import FeatureRow

# Canonical column registry for feature schema v1.
# Keys are ordered; the deterministic hash depends on this ordering.
_COLUMNS_V1: Final[dict[str, type]] = {
    "user_id": str,
    "device_id": str,
    "session_id": str,
    "bucket_start_ts": datetime,
    "bucket_end_ts": datetime,
    "schema_version": str,
    "schema_hash": str,
    "source_ids": list,
    "app_id": str,
    "app_category": str,
    "window_title_hash": str,
    "is_browser": bool,
    "is_editor": bool,
    "is_terminal": bool,
    "app_switch_count_last_5m": int,
    "app_foreground_time_ratio": float,
    "app_change_count": int,
    "app_dwell_time_seconds": float,
    "app_entropy_5m": float,
    "app_entropy_15m": float,
    "top2_app_concentration_15m": float,
    "idle_return_indicator": bool,
    "keys_per_min": float,
    "backspace_ratio": float,
    "shortcut_rate": float,
    "clicks_per_min": float,
    "scroll_events_per_min": float,
    "mouse_distance": float,
    "active_seconds_keyboard": float,
    "active_seconds_mouse": float,
    "active_seconds_any": float,
    "max_idle_run_seconds": float,
    "event_density": float,
    "domain_category": str,
    "window_title_bucket": int,
    "title_repeat_count_session": int,
    "keys_per_min_rolling_5": float,
    "keys_per_min_rolling_15": float,
    "mouse_distance_rolling_5": float,
    "mouse_distance_rolling_15": float,
    "keys_per_min_delta": float,
    "clicks_per_min_delta": float,
    "mouse_distance_delta": float,
    "app_switch_count_last_15m": int,
    "hour_of_day": int,
    "day_of_week": int,
    "session_length_so_far": float,
}

_COLUMNS_V2: Final[dict[str, type]] = {
    k: v for k, v in _COLUMNS_V1.items() if k != "user_id"
}


def _build_schema_hash(columns: dict[str, type]) -> str:
    payload = json.dumps(
        [[name, tp.__name__] for name, tp in columns.items()],
        separators=(",", ":"),
    )
    return stable_hash(payload)


@dataclass(frozen=True, eq=False)
class FeatureSchemaV1:
    """Schema contract for feature rows (v1).

    Holds the canonical column list, computes a deterministic schema hash,
    and validates individual rows or DataFrames against the contract.
    """

    VERSION: ClassVar[Final[str]] = "v1"
    COLUMNS: ClassVar[Final[dict[str, type]]] = _COLUMNS_V1
    SCHEMA_HASH: ClassVar[Final[str]] = _build_schema_hash(_COLUMNS_V1)

    # -- single-row validation ------------------------------------------

    @classmethod
    def validate_row(cls, data: dict[str, Any]) -> FeatureRow:
        """Validate *data* as a ``FeatureRow`` and verify schema metadata.

        Args:
            data: Raw dict of field values (e.g. from JSON or ``model_dump()``).

        Returns:
            The validated ``FeatureRow``.

        Raises:
            ValueError: If pydantic validation fails, or ``schema_version`` /
                ``schema_hash`` do not match the current contract.
        """
        row = FeatureRow.model_validate(data)
        if row.schema_version != cls.VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {cls.VERSION!r}, "
                f"got {row.schema_version!r}"
            )
        if row.schema_hash != cls.SCHEMA_HASH:
            raise ValueError(
                f"schema_hash mismatch: expected {cls.SCHEMA_HASH!r}, "
                f"got {row.schema_hash!r}"
            )
        return row

    # -- DataFrame-level validation -------------------------------------

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """Check that *df* conforms to the v1 column contract.

        Args:
            df: DataFrame to validate (typically built from ``FeatureRow.model_dump()``).

        Raises:
            ValueError: If columns are missing, unexpected columns are present,
                or pandas dtype kinds do not match the expected Python types.
        """
        expected = set(cls.COLUMNS)
        actual = set(df.columns)

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        extra = actual - expected
        if extra:
            raise ValueError(f"Unexpected columns: {sorted(extra)}")

        _check_dataframe_dtypes(df, cls.COLUMNS)


@dataclass(frozen=True, eq=False)
class FeatureSchemaV2:
    """Schema contract for feature rows (v2).

    Identical to :class:`FeatureSchemaV1` except ``user_id`` has been
    removed from the column registry.  Personalization shifts to
    calibrators and per-user post-processing.
    """

    VERSION: ClassVar[Final[str]] = "v2"
    COLUMNS: ClassVar[Final[dict[str, type]]] = _COLUMNS_V2
    SCHEMA_HASH: ClassVar[Final[str]] = _build_schema_hash(_COLUMNS_V2)

    @classmethod
    def validate_row(cls, data: dict[str, Any]) -> FeatureRow:
        """Validate *data* as a ``FeatureRow`` and verify schema metadata.

        Args:
            data: Raw dict of field values (e.g. from JSON or ``model_dump()``).

        Returns:
            The validated ``FeatureRow``.

        Raises:
            ValueError: If pydantic validation fails, or ``schema_version`` /
                ``schema_hash`` do not match the v2 contract.
        """
        row = FeatureRow.model_validate(data)
        if row.schema_version != cls.VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {cls.VERSION!r}, "
                f"got {row.schema_version!r}"
            )
        if row.schema_hash != cls.SCHEMA_HASH:
            raise ValueError(
                f"schema_hash mismatch: expected {cls.SCHEMA_HASH!r}, "
                f"got {row.schema_hash!r}"
            )
        return row

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """Check that *df* conforms to the v2 column contract.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If columns are missing, unexpected columns are present,
                or pandas dtype kinds do not match the expected Python types.
        """
        expected = set(cls.COLUMNS)
        actual = set(df.columns)

        missing = expected - actual
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        extra = actual - expected
        if extra:
            raise ValueError(f"Unexpected columns: {sorted(extra)}")

        _check_dataframe_dtypes(df, cls.COLUMNS)


# Maps Python types to sets of acceptable pandas dtype *kind* codes.
_PD_KIND_MAP: Final[dict[type, set[str]]] = {
    int: {"i", "u"},  # signed / unsigned integer
    float: {"f", "i", "u"},  # float (ints acceptable — promotion is safe)
    bool: {"b", "i", "u"},  # bool (numpy stores as int sometimes)
    str: {"O", "U"},  # object / unicode
}


def coerce_nullable_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nullable numeric columns from object (None) to float64 (NaN).

    When ``FeatureRow.model_dump()`` emits ``None`` for ``Optional[float]``
    fields, pandas stores the column as ``object`` dtype.  This helper
    coerces those columns to ``float64`` so downstream validation and
    parquet writing see the correct dtype.

    The DataFrame is modified **in-place** and also returned for convenience.
    """
    for col, expected_type in _COLUMNS_V1.items():
        if col not in df.columns:
            continue
        if expected_type in (float, int) and df[col].dtype.kind == "O":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _check_dataframe_dtypes(
    df: pd.DataFrame,
    columns: dict[str, type] = _COLUMNS_V1,
) -> None:
    errors: list[str] = []
    for col, expected_type in columns.items():
        if col not in df.columns:
            continue
        accepted = _PD_KIND_MAP.get(expected_type)
        if accepted is None:
            continue
        kind = df[col].dtype.kind
        if kind not in accepted:
            errors.append(
                f"Column '{col}': expected kind in {sorted(accepted)}, "
                f"got '{kind}' (dtype={df[col].dtype})"
            )
    if errors:
        raise ValueError("DataFrame dtype mismatches:\n" + "\n".join(errors))
