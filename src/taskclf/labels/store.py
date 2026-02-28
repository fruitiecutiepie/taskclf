"""Label span I/O, validation, and synthetic label generation."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS, DEFAULT_DUMMY_ROWS
from taskclf.core.store import read_parquet, write_parquet
from taskclf.core.types import LabelSpan

# Deterministic app_id -> label mapping aligned with features/build._DUMMY_APPS.
_APP_LABEL_MAP: Final[dict[str, str]] = {
    "com.apple.Terminal": "Build",
    "org.mozilla.firefox": "ReadResearch",
    "com.microsoft.VSCode": "Build",
    "com.apple.mail": "Communicate",
    "us.zoom.xos": "Meet",
    "com.tinyspeck.slackmacgap": "Communicate",
    "com.google.Chrome": "ReadResearch",
    "com.jetbrains.intellij": "Build",
    "com.apple.finder": "BreakIdle",
    "com.apple.Notes": "Write",
}

_DUMMY_APPS_ORDER: Final[list[str]] = [
    "com.apple.Terminal",
    "org.mozilla.firefox",
    "com.microsoft.VSCode",
    "com.apple.mail",
    "us.zoom.xos",
    "com.tinyspeck.slackmacgap",
    "com.google.Chrome",
    "com.jetbrains.intellij",
    "com.apple.finder",
    "com.apple.Notes",
]


def write_label_spans(spans: Sequence[LabelSpan], path: Path) -> Path:
    """Serialize *spans* to a parquet file at *path*.

    Args:
        spans: Label span instances to persist.
        path: Destination parquet file path.

    Returns:
        The *path* that was written.
    """
    df = pd.DataFrame([s.model_dump() for s in spans])
    return write_parquet(df, path)


def read_label_spans(path: Path) -> list[LabelSpan]:
    """Deserialize label spans from a parquet file.

    Args:
        path: Path to an existing parquet file written by
            :func:`write_label_spans`.

    Returns:
        List of validated ``LabelSpan`` instances.
    """
    df = read_parquet(path)
    return [LabelSpan.model_validate(row) for row in df.to_dict(orient="records")]


def import_labels_from_csv(path: Path) -> list[LabelSpan]:
    """Read label spans from a CSV file and validate each row.

    Required columns: ``start_ts``, ``end_ts``, ``label``, ``provenance``.
    Optional columns: ``user_id``, ``confidence``.

    Timestamps are parsed via ``pd.to_datetime`` so ISO-8601 and common
    date-time formats are accepted.

    Args:
        path: Path to an existing CSV file.

    Returns:
        List of validated ``LabelSpan`` instances.

    Raises:
        ValueError: If required columns are missing or any row fails
            ``LabelSpan`` validation.
    """
    df = pd.read_csv(path)

    required = {"start_ts", "end_ts", "label", "provenance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df["end_ts"] = pd.to_datetime(df["end_ts"])

    has_user_id = "user_id" in df.columns
    has_confidence = "confidence" in df.columns

    spans: list[LabelSpan] = []
    for _i, row in df.iterrows():
        kwargs: dict = {
            "start_ts": row["start_ts"].to_pydatetime(),
            "end_ts": row["end_ts"].to_pydatetime(),
            "label": row["label"],
            "provenance": row["provenance"],
        }
        if has_user_id and pd.notna(row["user_id"]):
            kwargs["user_id"] = str(row["user_id"])
        if has_confidence and pd.notna(row["confidence"]):
            kwargs["confidence"] = float(row["confidence"])
        spans.append(LabelSpan(**kwargs))
    return spans


def append_label_span(span: LabelSpan, path: Path) -> Path:
    """Append a single label span to an existing (or new) parquet file.

    Validates that the new span does not overlap any existing span
    belonging to the same user.

    Args:
        span: The label span to append.
        path: Parquet file to read-append-write.

    Returns:
        The *path* that was written.

    Raises:
        ValueError: If the span overlaps an existing span for the
            same user.
    """
    existing: list[LabelSpan] = []
    if path.exists():
        existing = read_label_spans(path)

    for ex in existing:
        if span.user_id is not None and ex.user_id is not None and span.user_id != ex.user_id:
            continue
        if span.user_id is None and ex.user_id is not None:
            continue
        if span.user_id is not None and ex.user_id is None:
            continue
        if span.start_ts < ex.end_ts and ex.start_ts < span.end_ts:
            raise ValueError(
                f"New span [{span.start_ts}, {span.end_ts}) overlaps "
                f"existing span [{ex.start_ts}, {ex.end_ts}) "
                f"for user {span.user_id!r}"
            )

    existing.append(span)
    return write_label_spans(existing, path)


def _same_user(a: LabelSpan, b: LabelSpan) -> bool:
    """True when both spans belong to the same user (or both have no user)."""
    if a.user_id is None and b.user_id is None:
        return True
    if a.user_id is not None and b.user_id is not None:
        return a.user_id == b.user_id
    return False


def extend_and_append_label_span(span: LabelSpan, path: Path) -> Path:
    """Append *span* and extend the previous label to fill the gap.

    Finds the most recent existing label for the same ``user_id`` (by
    ``start_ts``) and sets its ``end_ts`` to ``span.start_ts``, creating
    contiguous label coverage.  If the previous label would overlap the
    new span it is truncated instead.

    Falls back to :func:`append_label_span` when there is no previous
    label to extend.

    Args:
        span: The new label span to append.
        path: Parquet file to read-modify-write.

    Returns:
        The *path* that was written.

    Raises:
        ValueError: If any overlap remains after extension.
    """
    existing: list[LabelSpan] = []
    if path.exists():
        existing = read_label_spans(path)

    prev: LabelSpan | None = None
    prev_idx: int | None = None
    for i, ex in enumerate(existing):
        if not _same_user(ex, span):
            continue
        if ex.start_ts >= span.start_ts:
            continue
        if prev is None or ex.start_ts > prev.start_ts:
            prev = ex
            prev_idx = i

    if prev is not None and prev_idx is not None:
        updated = prev.model_copy(update={"end_ts": span.start_ts})
        existing[prev_idx] = updated

    existing.append(span)

    for i, a in enumerate(existing):
        for j, b in enumerate(existing):
            if i >= j:
                continue
            if not _same_user(a, b):
                continue
            if a.start_ts < b.end_ts and b.start_ts < a.end_ts:
                raise ValueError(
                    f"Span [{a.start_ts}, {a.end_ts}) overlaps "
                    f"[{b.start_ts}, {b.end_ts}) for user {a.user_id!r}"
                )

    return write_label_spans(existing, path)


def generate_label_summary(
    features_df: pd.DataFrame,
    start_ts: dt.datetime,
    end_ts: dt.datetime,
) -> dict:
    """Summarise feature rows within a time range for display in CLI / UI.

    Returns a dict with top apps, aggregated interaction stats, and
    session count.  Respects privacy: no raw titles.

    Args:
        features_df: Feature DataFrame with ``bucket_start_ts`` column.
        start_ts: Start of summary window (inclusive).
        end_ts: End of summary window (exclusive).

    Returns:
        Dict with keys ``top_apps``, ``mean_keys_per_min``,
        ``mean_clicks_per_min``, ``mean_scroll_per_min``,
        ``total_buckets``, ``session_count``.
    """
    mask = (features_df["bucket_start_ts"] >= start_ts) & (
        features_df["bucket_start_ts"] < end_ts
    )
    window = features_df.loc[mask]

    if window.empty:
        return {
            "top_apps": [],
            "mean_keys_per_min": None,
            "mean_clicks_per_min": None,
            "mean_scroll_per_min": None,
            "total_buckets": 0,
            "session_count": 0,
        }

    top_apps: list[dict] = []
    if "app_id" in window.columns:
        counts = window["app_id"].value_counts().head(5)
        top_apps = [
            {"app_id": app, "buckets": int(cnt)}
            for app, cnt in counts.items()
        ]

    def _safe_mean(col: str) -> float | None:
        if col in window.columns:
            vals = window[col].dropna()
            if not vals.empty:
                return round(float(vals.mean()), 2)
        return None

    session_count = 0
    if "session_id" in window.columns:
        session_count = int(window["session_id"].nunique())

    return {
        "top_apps": top_apps,
        "mean_keys_per_min": _safe_mean("keys_per_min"),
        "mean_clicks_per_min": _safe_mean("clicks_per_min"),
        "mean_scroll_per_min": _safe_mean("scroll_events_per_min"),
        "total_buckets": len(window),
        "session_count": session_count,
    }


def generate_dummy_labels(date: dt.date, n_rows: int = DEFAULT_DUMMY_ROWS) -> list[LabelSpan]:
    """Create synthetic label spans aligned to the dummy feature timestamps.

    Each span covers exactly one minute bucket, mirroring the timestamps
    generated by ``features.build.generate_dummy_features`` so that every
    feature row has a covering label.

    Args:
        date: Calendar date to generate spans for.
        n_rows: Number of one-minute spans to create.

    Returns:
        List of ``LabelSpan`` instances with provenance ``"synthetic"``.
    """
    spans: list[LabelSpan] = []
    for i in range(n_rows):
        hour = 9 + (i * 8 // max(n_rows, 1))
        minute = (i * 7) % 60
        start = dt.datetime(date.year, date.month, date.day, hour, minute)
        end = start + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)

        app_id = _DUMMY_APPS_ORDER[i % len(_DUMMY_APPS_ORDER)]
        label = _APP_LABEL_MAP[app_id]

        spans.append(
            LabelSpan(
                start_ts=start,
                end_ts=end,
                label=label,
                provenance="synthetic",
            )
        )
    return spans
