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

from taskclf.core.defaults import (
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_SMOOTH_WINDOW,
    MIXED_UNKNOWN,
)
from taskclf.core.types import LABEL_SET_V1
from taskclf.infer.smooth import Segment, rolling_majority, segmentize
from taskclf.train.lgbm import FEATURE_COLUMNS, encode_categoricals


def predict_proba(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    cat_encoders: dict[str, LabelEncoder] | None = None,
) -> np.ndarray:
    """Return raw probability matrix for *features_df*.

    Args:
        model: Trained LightGBM booster.
        features_df: Feature DataFrame with ``FEATURE_COLUMNS``.
        cat_encoders: Pre-fitted categorical encoders for string columns.

    Returns:
        Probability matrix of shape ``(n_rows, n_classes)``.
    """
    feat_df = features_df[FEATURE_COLUMNS].copy()
    feat_df, _ = encode_categoricals(feat_df, cat_encoders)
    x = feat_df.fillna(0).to_numpy(dtype=np.float64)
    return model.predict(x)


def predict_labels(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    *,
    reject_threshold: float | None = None,
) -> list[str]:
    """Run the model on *features_df* and return predicted label strings.

    When *reject_threshold* is set, any prediction whose maximum
    probability falls below the threshold is replaced with
    ``Mixed/Unknown``.

    Args:
        model: Trained LightGBM booster.
        features_df: Feature DataFrame with ``FEATURE_COLUMNS``.
        label_encoder: Encoder fitted on the canonical label vocabulary.
        cat_encoders: Pre-fitted categorical encoders for string columns.
        reject_threshold: If given, predictions with
            ``max(proba) < reject_threshold`` become ``Mixed/Unknown``.

    Returns:
        Predicted label per row.
    """
    proba = predict_proba(model, features_df, cat_encoders)
    pred_indices = proba.argmax(axis=1)
    labels = list(label_encoder.inverse_transform(pred_indices))

    if reject_threshold is not None:
        confidences = proba.max(axis=1)
        labels = [
            MIXED_UNKNOWN if conf < reject_threshold else lbl
            for lbl, conf in zip(labels, confidences)
        ]

    return labels


def run_batch_inference(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    *,
    cat_encoders: dict[str, LabelEncoder] | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    reject_threshold: float | None = None,
) -> tuple[list[str], list[Segment], np.ndarray, np.ndarray]:
    """Predict, smooth, and segmentize a batch of feature rows.

    Args:
        model: Trained LightGBM booster.
        features_df: Feature DataFrame (must contain ``FEATURE_COLUMNS``
            and ``bucket_start_ts``).
        cat_encoders: Pre-fitted categorical encoders for string columns.
        smooth_window: Window size for rolling-majority smoothing.
        bucket_seconds: Width of each time bucket in seconds.
        reject_threshold: If given, predictions with
            ``max(proba) < reject_threshold`` become ``Mixed/Unknown``
            before smoothing.

    Returns:
        A ``(smoothed_labels, segments, confidences, is_rejected)`` tuple.
        *confidences* is ``max(proba)`` per row and *is_rejected* is a
        boolean array indicating which rows fell below the threshold
        (all ``False`` when *reject_threshold* is ``None``).
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))

    proba = predict_proba(model, features_df, cat_encoders)
    confidences = proba.max(axis=1)
    is_rejected = (
        confidences < reject_threshold
        if reject_threshold is not None
        else np.zeros(len(confidences), dtype=bool)
    )

    pred_indices = proba.argmax(axis=1)
    raw_labels: list[str] = list(le.inverse_transform(pred_indices))
    raw_labels = [
        MIXED_UNKNOWN if rej else lbl
        for lbl, rej in zip(raw_labels, is_rejected)
    ]

    smoothed = rolling_majority(raw_labels, window=smooth_window)

    bucket_starts: list[datetime] = [
        pd.Timestamp(ts).to_pydatetime()
        for ts in features_df["bucket_start_ts"].values
    ]
    segments = segmentize(bucket_starts, smoothed, bucket_seconds=bucket_seconds)

    return smoothed, segments, confidences, is_rejected


def write_predictions_csv(
    features_df: pd.DataFrame,
    labels: Sequence[str],
    path: Path,
    *,
    confidences: np.ndarray | None = None,
    is_rejected: np.ndarray | None = None,
) -> Path:
    """Write per-bucket predictions to a CSV file.

    Args:
        features_df: Source feature DataFrame.
        labels: Predicted (smoothed) label per row.
        path: Destination CSV path.
        confidences: Optional max-probability per row.
        is_rejected: Optional boolean rejection flag per row.

    Returns:
        The *path* that was written.
    """
    data: dict[str, object] = {
        "bucket_start_ts": features_df["bucket_start_ts"].values,
        "predicted_label": labels,
    }
    if confidences is not None:
        data["confidence"] = np.round(confidences, 4)
    if is_rejected is not None:
        data["is_rejected"] = is_rejected

    out = pd.DataFrame(data)
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
