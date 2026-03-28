"""Interval-aware aggregation strategies for multi-bucket predictions.

Given a list of per-bucket :class:`~taskclf.infer.prediction.WindowPrediction`
objects spanning a time interval, these functions reduce them to a single
``(label, confidence)`` pair suitable for a tray suggestion.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from taskclf.infer.prediction import WindowPrediction


def majority_vote(labels: list[str]) -> str:
    """Return the most frequent label. Ties broken by first occurrence."""
    if not labels:
        raise ValueError("labels must be non-empty")
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def confidence_weighted_vote(labels: list[str], confidences: list[float]) -> str:
    """Return the label with the highest total confidence weight.

    Each bucket's confidence is accumulated for its predicted label.
    The label with the greatest total weight wins.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    if len(labels) != len(confidences):
        raise ValueError("labels and confidences must have the same length")
    weights: dict[str, float] = {}
    for label, conf in zip(labels, confidences):
        weights[label] = weights.get(label, 0.0) + conf
    return max(weights, key=weights.__getitem__)


def highest_total_probability(proba_matrix: np.ndarray, label_names: list[str]) -> str:
    """Return the label with the highest summed probability across buckets.

    Args:
        proba_matrix: Shape ``(N, C)`` where N is the number of buckets
            and C is the number of classes.
        label_names: Ordered class names matching columns of *proba_matrix*.
    """
    if proba_matrix.ndim != 2:
        raise ValueError("proba_matrix must be 2-dimensional")
    if proba_matrix.shape[1] != len(label_names):
        raise ValueError(
            f"proba_matrix has {proba_matrix.shape[1]} columns "
            f"but {len(label_names)} label names were provided"
        )
    col_sums = proba_matrix.sum(axis=0)
    return label_names[int(np.argmax(col_sums))]


def aggregate_interval(
    predictions: list[WindowPrediction],
    strategy: str = "majority",
) -> tuple[str, float]:
    """Aggregate per-bucket predictions into a single interval result.

    Args:
        predictions: One :class:`WindowPrediction` per bucket in the interval.
        strategy: ``"majority"``, ``"confidence_weighted"``, or
            ``"highest_probability"``.

    Returns:
        ``(label, confidence)`` where *confidence* is the mean confidence
        of buckets that predicted the winning label.

    Raises:
        ValueError: If *predictions* is empty or *strategy* is unknown.
    """
    if not predictions:
        raise ValueError("predictions must be non-empty")

    labels = [p.core_label_name for p in predictions]
    confidences = [p.confidence for p in predictions]

    if strategy == "majority":
        winner = majority_vote(labels)
    elif strategy == "confidence_weighted":
        winner = confidence_weighted_vote(labels, confidences)
    elif strategy == "highest_probability":
        label_names = list(predictions[0].mapped_probs.keys())
        proba_matrix = np.array([list(p.mapped_probs.values()) for p in predictions])
        winner = highest_total_probability(proba_matrix, label_names)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    winner_confidences = [c for lbl, c in zip(labels, confidences) if lbl == winner]
    agg_confidence = (
        sum(winner_confidences) / len(winner_confidences)
        if winner_confidences
        else confidences[0]
    )

    return (winner, round(agg_confidence, 6))
