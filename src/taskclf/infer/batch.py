"""Batch inference: predict, smooth, and segmentize over a feature DataFrame."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS, DEFAULT_SMOOTH_WINDOW
from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.smooth import Segment, rolling_majority, segmentize
from taskclf.train.lgbm import FEATURE_COLUMNS


def predict_labels(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> list[str]:
    """Run the model on *features_df* and return predicted label strings.

    Args:
        model: Trained LightGBM booster.
        features_df: Feature DataFrame with ``FEATURE_COLUMNS``.
        label_encoder: Encoder fitted on the canonical label vocabulary.

    Returns:
        Predicted label per row.
    """
    x = features_df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=np.float64)
    proba = model.predict(x)
    pred_indices = proba.argmax(axis=1)
    return list(label_encoder.inverse_transform(pred_indices))


def run_batch_inference(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    *,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> tuple[list[str], list[Segment]]:
    """Predict, smooth, and segmentize a batch of feature rows.

    Args:
        model: Trained LightGBM booster.
        features_df: Feature DataFrame (must contain ``FEATURE_COLUMNS``
            and ``bucket_start_ts``).
        smooth_window: Window size for rolling-majority smoothing.
        bucket_seconds: Width of each time bucket in seconds.

    Returns:
        A ``(smoothed_labels, segments)`` tuple where *smoothed_labels*
        has one entry per row in *features_df* and *segments* are the
        merged contiguous spans.
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))

    raw_labels = predict_labels(model, features_df, le)
    smoothed = rolling_majority(raw_labels, window=smooth_window)

    bucket_starts: list[datetime] = [
        pd.Timestamp(ts).to_pydatetime()
        for ts in features_df["bucket_start_ts"].values
    ]
    segments = segmentize(bucket_starts, smoothed, bucket_seconds=bucket_seconds)

    return smoothed, segments


def write_predictions_csv(
    features_df: pd.DataFrame,
    labels: Sequence[str],
    path: Path,
) -> Path:
    """Write per-bucket predictions to a CSV file.

    Args:
        features_df: Source feature DataFrame.
        labels: Predicted (smoothed) label per row.
        path: Destination CSV path.

    Returns:
        The *path* that was written.
    """
    out = pd.DataFrame({
        "bucket_start_ts": features_df["bucket_start_ts"].values,
        "predicted_label": labels,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def write_segments_json(segments: Sequence[Segment], path: Path) -> Path:
    """Write segments to a JSON file.

    Args:
        segments: Segment instances from :func:`run_batch_inference`.
        path: Destination JSON path.

    Returns:
        The *path* that was written.
    """
    records = []
    for seg in segments:
        d = asdict(seg)
        d["start_ts"] = seg.start_ts.isoformat()
        d["end_ts"] = seg.end_ts.isoformat()
        records.append(d)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))
    return path


def read_segments_json(path: Path) -> list[Segment]:
    """Read segments from a JSON file written by :func:`write_segments_json`.

    Args:
        path: Path to an existing segments JSON file.

    Returns:
        List of ``Segment`` instances.
    """
    records = json.loads(path.read_text())
    return [
        Segment(
            start_ts=datetime.fromisoformat(r["start_ts"]),
            end_ts=datetime.fromisoformat(r["end_ts"]),
            label=r["label"],
            bucket_count=r["bucket_count"],
        )
        for r in records
    ]
