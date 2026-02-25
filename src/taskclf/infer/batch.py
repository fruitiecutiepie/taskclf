"""Batch inference: predict, smooth, and segmentize over a feature DataFrame."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
from taskclf.infer.calibration import Calibrator, CalibratorStore, IdentityCalibrator
from taskclf.infer.smooth import Segment, merge_short_segments, rolling_majority, segmentize
from taskclf.infer.taxonomy import TaxonomyConfig, TaxonomyResolver
from taskclf.train.lgbm import FEATURE_COLUMNS, encode_categoricals


@dataclass(frozen=True)
class BatchInferenceResult:
    """Container for batch inference outputs.

    Always contains core prediction fields.  Taxonomy-mapped fields
    (``mapped_labels``, ``mapped_probs``) are ``None`` when no taxonomy
    config was provided.
    """

    raw_labels: list[str]
    smoothed_labels: list[str]
    segments: list[Segment]
    confidences: np.ndarray
    is_rejected: np.ndarray
    core_probs: np.ndarray
    mapped_labels: list[str] | None = field(default=None)
    mapped_probs: list[dict[str, float]] | None = field(default=None)


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
    taxonomy: TaxonomyConfig | None = None,
    calibrator: Calibrator | None = None,
    calibrator_store: CalibratorStore | None = None,
) -> BatchInferenceResult:
    """Predict, smooth, segmentize, and apply hysteresis merging.

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
        taxonomy: Optional taxonomy config.  When provided, core labels
            are mapped to user-defined buckets and ``mapped_labels`` /
            ``mapped_probs`` are populated on the result.
        calibrator: Optional probability calibrator.  When provided,
            raw model probabilities are calibrated before the reject
            decision.
        calibrator_store: Optional per-user calibrator store.  When
            provided, per-user calibration is applied using the
            ``user_id`` column in *features_df*.  Takes precedence
            over *calibrator*.

    Returns:
        A :class:`BatchInferenceResult` with core predictions, segments
        (hysteresis-merged), and optional taxonomy-mapped outputs.
    """
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))

    proba = predict_proba(model, features_df, cat_encoders)

    if calibrator_store is not None and "user_id" in features_df.columns:
        proba = calibrator_store.calibrate_batch(
            proba, list(features_df["user_id"].values),
        )
    else:
        cal = calibrator or IdentityCalibrator()
        proba = cal.calibrate(proba)
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
    segments = merge_short_segments(segments, bucket_seconds=bucket_seconds)

    mapped_labels: list[str] | None = None
    mapped_probs_list: list[dict[str, float]] | None = None
    if taxonomy is not None:
        resolver = TaxonomyResolver(taxonomy)
        results = resolver.resolve_batch(pred_indices, proba, is_rejected=is_rejected)
        mapped_labels = [r.mapped_label for r in results]
        mapped_probs_list = [r.mapped_probs for r in results]

    return BatchInferenceResult(
        raw_labels=raw_labels,
        smoothed_labels=smoothed,
        segments=segments,
        confidences=confidences,
        is_rejected=is_rejected,
        core_probs=proba,
        mapped_labels=mapped_labels,
        mapped_probs=mapped_probs_list,
    )


def write_predictions_csv(
    features_df: pd.DataFrame,
    labels: Sequence[str],
    path: Path,
    *,
    confidences: np.ndarray | None = None,
    is_rejected: np.ndarray | None = None,
    mapped_labels: Sequence[str] | None = None,
    core_probs: np.ndarray | None = None,
) -> Path:
    """Write per-bucket predictions to a CSV file.

    Args:
        features_df: Source feature DataFrame.
        labels: Predicted (smoothed) label per row.
        path: Destination CSV path.
        confidences: Optional max-probability per row.
        is_rejected: Optional boolean rejection flag per row.
        mapped_labels: Optional taxonomy-mapped label per row.
        core_probs: Optional probability matrix ``(n_rows, n_classes)``.

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
    if mapped_labels is not None:
        data["mapped_label"] = list(mapped_labels)
    if core_probs is not None:
        import json as _json

        data["core_probs"] = [
            _json.dumps([round(float(p), 4) for p in row])
            for row in core_probs
        ]

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
