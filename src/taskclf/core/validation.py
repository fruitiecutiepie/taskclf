"""Data validation: range checks, missing rates, monotonic timestamps, and more."""

from __future__ import annotations

import warnings
from datetime import timedelta
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS

# Range constraints drawn from schema/features_v1.json.
_RANGE_CONSTRAINTS: dict[str, dict[str, float]] = {
    "app_foreground_time_ratio": {"min": 0.0, "max": 1.0},
    "app_change_count": {"min": 0},
    "app_switch_count_last_5m": {"min": 0},
    "hour_of_day": {"min": 0, "max": 23},
    "day_of_week": {"min": 0, "max": 6},
    "session_length_so_far": {"min": 0},
    "keys_per_min": {"min": 0},
    "backspace_ratio": {"min": 0, "max": 1},
    "shortcut_rate": {"min": 0},
    "clicks_per_min": {"min": 0},
    "scroll_events_per_min": {"min": 0},
    "mouse_distance": {"min": 0},
    "active_seconds_keyboard": {"min": 0, "max": 60},
    "active_seconds_mouse": {"min": 0, "max": 60},
    "active_seconds_any": {"min": 0, "max": 60},
    "max_idle_run_seconds": {"min": 0},
    "event_density": {"min": 0},
}

# Columns that must never contain nulls.
_NON_NULLABLE: frozenset[str] = frozenset([
    "user_id", "bucket_start_ts", "bucket_end_ts", "session_id",
    "schema_version", "schema_hash",
    "app_id", "app_category",
    "is_browser", "is_editor", "is_terminal",
    "app_switch_count_last_5m", "app_foreground_time_ratio", "app_change_count",
    "hour_of_day", "day_of_week", "session_length_so_far",
])


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Finding(BaseModel, frozen=True):
    severity: Severity
    check: str
    column: str | None = None
    message: str
    detail: Any = None


class ValidationReport(BaseModel):
    """Collects all findings from :func:`validate_feature_dataframe`."""

    findings: list[Finding] = []

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_feature_dataframe(
    df: pd.DataFrame,
    *,
    max_missing_rate: float = 0.5,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> ValidationReport:
    """Run hard and soft checks on a feature DataFrame.

    Hard checks (errors):
        * Non-nullable columns contain nulls.
        * Nullable columns exceed *max_missing_rate*.
        * Numeric values outside declared ranges.
        * ``bucket_end_ts != bucket_start_ts + bucket_seconds``.
        * Non-monotonic ``bucket_start_ts`` within ``(user_id, session_id)``
          groups.

    Soft checks (warnings):
        * Constant-value columns (std == 0).
        * Dominant-value columns (>90% identical).
        * Label class imbalance (<5% representation) if ``label`` column
          exists.

    Args:
        df: Feature DataFrame to validate.
        max_missing_rate: Maximum allowed null fraction for nullable columns.
        bucket_seconds: Expected window width in seconds.

    Returns:
        A :class:`ValidationReport` with all findings.
    """
    report = ValidationReport()
    if df.empty:
        report.findings.append(Finding(
            severity=Severity.WARNING,
            check="empty_dataframe",
            message="DataFrame is empty; no checks performed.",
        ))
        return report

    _check_non_nullable(df, report)
    _check_missing_rates(df, max_missing_rate, report)
    _check_ranges(df, report)
    _check_bucket_end_consistency(df, bucket_seconds, report)
    _check_monotonic_timestamps(df, report)
    _check_session_boundaries(df, report)
    _check_distributions(df, report)
    _check_class_balance(df, report)

    return report


def _check_non_nullable(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in _NON_NULLABLE:
        if col not in df.columns:
            continue
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            report.findings.append(Finding(
                severity=Severity.ERROR,
                check="non_nullable",
                column=col,
                message=f"Non-nullable column '{col}' has {null_count} null(s).",
                detail={"null_count": null_count},
            ))


def _check_missing_rates(
    df: pd.DataFrame, max_rate: float, report: ValidationReport,
) -> None:
    for col in df.columns:
        if col in _NON_NULLABLE:
            continue
        null_frac = float(df[col].isnull().mean())
        if null_frac > max_rate:
            report.findings.append(Finding(
                severity=Severity.ERROR,
                check="missing_rate",
                column=col,
                message=(
                    f"Column '{col}' missing rate {null_frac:.2%} "
                    f"exceeds threshold {max_rate:.2%}."
                ),
                detail={"missing_rate": round(null_frac, 4)},
            ))


def _check_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    for col, bounds in _RANGE_CONSTRAINTS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        lo = bounds.get("min")
        hi = bounds.get("max")
        if lo is not None:
            below = int((series < lo).sum())
            if below > 0:
                report.findings.append(Finding(
                    severity=Severity.ERROR,
                    check="range_min",
                    column=col,
                    message=f"{below} value(s) in '{col}' below minimum {lo}.",
                    detail={"violations": below, "bound": lo},
                ))
        if hi is not None:
            above = int((series > hi).sum())
            if above > 0:
                report.findings.append(Finding(
                    severity=Severity.ERROR,
                    check="range_max",
                    column=col,
                    message=f"{above} value(s) in '{col}' above maximum {hi}.",
                    detail={"violations": above, "bound": hi},
                ))


def _check_bucket_end_consistency(
    df: pd.DataFrame, bucket_seconds: int, report: ValidationReport,
) -> None:
    if "bucket_start_ts" not in df.columns or "bucket_end_ts" not in df.columns:
        return
    expected = df["bucket_start_ts"] + timedelta(seconds=bucket_seconds)
    mismatches = int((df["bucket_end_ts"] != expected).sum())
    if mismatches > 0:
        report.findings.append(Finding(
            severity=Severity.ERROR,
            check="bucket_end_consistency",
            message=(
                f"{mismatches} row(s) where bucket_end_ts != "
                f"bucket_start_ts + {bucket_seconds}s."
            ),
            detail={"mismatches": mismatches},
        ))


def _check_monotonic_timestamps(df: pd.DataFrame, report: ValidationReport) -> None:
    if "user_id" not in df.columns or "session_id" not in df.columns:
        return
    for (uid, sid), group in df.groupby(["user_id", "session_id"], sort=False):
        ts = group["bucket_start_ts"].values
        if len(ts) < 2:
            continue
        non_mono = int((ts[1:] <= ts[:-1]).sum())
        if non_mono > 0:
            report.findings.append(Finding(
                severity=Severity.ERROR,
                check="monotonic_timestamps",
                message=(
                    f"Non-monotonic bucket_start_ts in "
                    f"user_id={uid}, session_id={sid}: "
                    f"{non_mono} violation(s)."
                ),
                detail={"user_id": uid, "session_id": sid, "violations": non_mono},
            ))


def _check_session_boundaries(df: pd.DataFrame, report: ValidationReport) -> None:
    """Warn if different session_ids appear for contiguous timestamps."""
    if "user_id" not in df.columns or "session_id" not in df.columns:
        return
    for uid, group in df.groupby("user_id", sort=False):
        group = group.sort_values("bucket_start_ts")
        sids = group["session_id"].values
        ts = group["bucket_start_ts"].values
        if len(sids) < 2:
            continue
        for i in range(1, len(sids)):
            if sids[i] == sids[i - 1]:
                continue
            gap = (pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds()
            if gap <= 60:
                report.findings.append(Finding(
                    severity=Severity.WARNING,
                    check="session_boundary",
                    message=(
                        f"Session change with only {gap}s gap at "
                        f"user_id={uid}, ts={ts[i]}."
                    ),
                    detail={"user_id": uid, "gap_seconds": gap},
                ))
                break


def _check_distributions(df: pd.DataFrame, report: ValidationReport) -> None:
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.std() == 0:
            report.findings.append(Finding(
                severity=Severity.WARNING,
                check="constant_column",
                column=col,
                message=f"Column '{col}' has zero variance (constant).",
            ))
            continue
        mode_frac = float(series.value_counts(normalize=True).iloc[0])
        if mode_frac > 0.9:
            report.findings.append(Finding(
                severity=Severity.WARNING,
                check="dominant_value",
                column=col,
                message=(
                    f"Column '{col}' has a dominant value "
                    f"({mode_frac:.1%} identical)."
                ),
                detail={"mode_fraction": round(mode_frac, 4)},
            ))


def _check_class_balance(df: pd.DataFrame, report: ValidationReport) -> None:
    if "label" not in df.columns:
        return
    dist = df["label"].value_counts(normalize=True)
    for label, frac in dist.items():
        if frac < 0.05:
            report.findings.append(Finding(
                severity=Severity.WARNING,
                check="class_balance",
                column="label",
                message=(
                    f"Class '{label}' has only {frac:.1%} representation "
                    f"(below 5% threshold)."
                ),
                detail={"label": label, "fraction": round(float(frac), 4)},
            ))
