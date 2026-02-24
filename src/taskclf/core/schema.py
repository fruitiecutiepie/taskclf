"""Feature schema versioning, deterministic hashing, and DataFrame validation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Final

import pandas as pd

from taskclf.core.hashing import stable_hash
from taskclf.core.types import FeatureRow

# Canonical column registry for feature schema v1.
# Keys are ordered; the deterministic hash depends on this ordering.
_COLUMNS_V1: Final[dict[str, type]] = {
    "bucket_start_ts": datetime,
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
    "keys_per_min": float,
    "backspace_ratio": float,
    "shortcut_rate": float,
    "clicks_per_min": float,
    "scroll_events_per_min": float,
    "mouse_distance": float,
    "hour_of_day": int,
    "day_of_week": int,
    "session_length_so_far": float,
}


def _build_schema_hash(columns: dict[str, type]) -> str:
    payload = json.dumps(
        [[name, tp.__name__] for name, tp in columns.items()],
        separators=(",", ":"),
    )
    return stable_hash(payload)


class FeatureSchemaV1:
    """Schema contract for feature rows (v1).

    Holds the canonical column list, computes a deterministic schema hash,
    and validates individual rows or DataFrames against the contract.
    """

    VERSION: Final[str] = "v1"
    COLUMNS: Final[dict[str, type]] = _COLUMNS_V1
    SCHEMA_HASH: Final[str] = _build_schema_hash(_COLUMNS_V1)

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

        _check_dataframe_dtypes(df)


# Maps Python types to sets of acceptable pandas dtype *kind* codes.
_PD_KIND_MAP: Final[dict[type, set[str]]] = {
    int: {"i", "u"},       # signed / unsigned integer
    float: {"f", "i", "u"},  # float (ints acceptable — promotion is safe)
    bool: {"b", "i", "u"},   # bool (numpy stores as int sometimes)
    str: {"O", "U"},       # object / unicode
}


def _check_dataframe_dtypes(df: pd.DataFrame) -> None:
    errors: list[str] = []
    for col, expected_type in _COLUMNS_V1.items():
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
        raise ValueError(
            "DataFrame dtype mismatches:\n" + "\n".join(errors)
        )
