"""Block-to-window label projection following time_spec.md Section 6.

Manual labeling is done in time blocks (LabelSpan instances).
This module projects those blocks onto fixed-width feature windows
using strict containment rules so only cleanly-labeled windows enter
the training set.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.core.types import LabelSpan


def project_blocks_to_windows(
    features_df: pd.DataFrame,
    label_spans: Sequence[LabelSpan],
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> pd.DataFrame:
    """Assign labels from *label_spans* to feature windows using strict containment.

    Projection rules (per ``time_spec.md`` Section 6):

    1. A window is labeled only when its **entire** ``[bucket_start_ts,
       bucket_end_ts)`` interval falls within a single labeled block.
    2. Windows overlapping **multiple** labeled blocks are **dropped**.
    3. Windows that only **partially** overlap a block are **dropped**.
    4. Unlabeled windows are **dropped** (not used in supervised training).
    5. When a span carries a ``user_id``, it only matches feature rows
       with the same ``user_id``.

    Args:
        features_df: Feature DataFrame.  Must contain ``bucket_start_ts``
            and ``bucket_end_ts`` columns, plus ``user_id``.
        label_spans: Label spans (blocks) to project.
        bucket_seconds: Window width in seconds (used to derive
            ``bucket_end_ts`` if the column is missing).

    Returns:
        A copy of *features_df* containing only the windows that satisfy
        the strict containment rule, with an added ``label`` column.
    """
    if features_df.empty or not label_spans:
        result = features_df.copy()
        result["label"] = pd.Series(dtype="object")
        return result.iloc[0:0].reset_index(drop=True)

    df = features_df.copy()

    if "bucket_end_ts" not in df.columns:
        df["bucket_end_ts"] = df["bucket_start_ts"] + pd.Timedelta(seconds=bucket_seconds)

    df["bucket_start_ts"] = pd.to_datetime(df["bucket_start_ts"], utc=True)
    df["bucket_end_ts"] = pd.to_datetime(df["bucket_end_ts"], utc=True)

    spans_data = [
        {
            "span_start": pd.Timestamp(s.start_ts, tz="UTC"),
            "span_end": pd.Timestamp(s.end_ts, tz="UTC"),
            "span_label": s.label,
            "span_user_id": s.user_id,
        }
        for s in label_spans
    ]

    labels: list[str | None] = [None] * len(df)
    multi_overlap: list[bool] = [False] * len(df)

    for idx in range(len(df)):
        w_start = df["bucket_start_ts"].iat[idx]
        w_end = df["bucket_end_ts"].iat[idx]
        row_user = df["user_id"].iat[idx] if "user_id" in df.columns else None

        covering: list[str] = []
        for sp in spans_data:
            if sp["span_user_id"] is not None and sp["span_user_id"] != row_user:
                continue
            if sp["span_start"] <= w_start and w_end <= sp["span_end"]:
                covering.append(sp["span_label"])

        if not covering:
            pass
        else:
            unique_labels = set(covering)
            if len(unique_labels) == 1:
                labels[idx] = covering[0]
            else:
                multi_overlap[idx] = True

    df["label"] = labels
    df["_multi_overlap"] = multi_overlap

    result = df[df["label"].notna() & ~df["_multi_overlap"]].copy()
    result = result.drop(columns=["_multi_overlap"]).reset_index(drop=True)
    return result
