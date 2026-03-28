"""Tests for interval-aware aggregation strategies.

Covers:
- AGG-001: Aggregation over N buckets uses all N, not just the last.
- AGG-002: 3x Coding, 2x Writing -> majority vote returns "Coding".
- AGG-003: High-confidence minority wins confidence-weighted vote.
- AGG-004: Single-bucket interval returns same result as direct prediction.
- P4-001: Different strategies produce different results on mixed input.
- EXP-B: Compare 3+ strategies, assert per-strategy accuracy is returned.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from taskclf.infer.aggregation import (
    aggregate_interval,
    confidence_weighted_vote,
    highest_total_probability,
    majority_vote,
)
from taskclf.infer.prediction import WindowPrediction

LABEL_NAMES = [
    "BreakIdle",
    "Build",
    "Communicate",
    "Debug",
    "Design",
    "Meet",
    "Review",
    "Write",
]


def _make_prediction(
    label: str,
    confidence: float,
    *,
    label_id: int | None = None,
    probs: list[float] | None = None,
) -> WindowPrediction:
    """Build a minimal WindowPrediction for testing."""
    if label_id is None:
        label_id = LABEL_NAMES.index(label)
    if probs is None:
        probs = [0.0] * 8
        probs[label_id] = confidence
        remaining = 1.0 - confidence
        others = [i for i in range(8) if i != label_id]
        for i in others:
            probs[i] = round(remaining / len(others), 6)
        diff = round(1.0 - sum(probs), 6)
        probs[others[0]] += diff
    mapped_probs = {name: p for name, p in zip(LABEL_NAMES, probs)}
    return WindowPrediction(
        user_id="test-user",
        bucket_start_ts=datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
        core_label_id=label_id,
        core_label_name=label,
        core_probs=probs,
        confidence=confidence,
        is_rejected=False,
        mapped_label_name=label,
        mapped_probs=mapped_probs,
        model_version="test-hash",
    )


class TestMajorityVote:
    def test_agg002_majority_coding_wins(self) -> None:
        """AGG-002: 3x Coding, 2x Writing -> 'Coding'."""
        labels = ["Build", "Build", "Build", "Write", "Write"]
        assert majority_vote(labels) == "Build"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            majority_vote([])


class TestConfidenceWeightedVote:
    def test_agg003_high_confidence_minority_wins(self) -> None:
        """AGG-003: High-confidence minority can win over low-confidence majority."""
        labels = ["Build", "Build", "Build", "Write", "Write"]
        confidences = [0.3, 0.3, 0.3, 0.95, 0.95]
        assert confidence_weighted_vote(labels, confidences) == "Write"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            confidence_weighted_vote([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            confidence_weighted_vote(["Build"], [0.5, 0.5])


class TestHighestTotalProbability:
    def test_summed_column_wins(self) -> None:
        proba = np.array(
            [
                [0.6, 0.4],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
        )
        assert highest_total_probability(proba, ["A", "B"]) == "B"

    def test_wrong_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            highest_total_probability(np.array([0.5, 0.5]), ["A", "B"])

    def test_column_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            highest_total_probability(np.array([[0.5, 0.5]]), ["A"])


class TestAggregateInterval:
    def test_agg001_uses_all_buckets(self) -> None:
        """AGG-001: Aggregation over N buckets uses all N, not just the last."""
        preds = [
            _make_prediction("Build", 0.9),
            _make_prediction("Build", 0.85),
            _make_prediction("Write", 0.7),
        ]
        label, _ = aggregate_interval(preds, strategy="majority")
        assert label == "Build"
        last_only = preds[-1].core_label_name
        assert last_only == "Write"

    def test_agg004_single_bucket_same_as_direct(self) -> None:
        """AGG-004: Single-bucket interval equals direct prediction."""
        pred = _make_prediction("Debug", 0.88)
        label, confidence = aggregate_interval([pred], strategy="majority")
        assert label == pred.core_label_name
        assert confidence == pytest.approx(pred.confidence, abs=1e-5)

    def test_p4001_strategies_differ_on_mixed_input(self) -> None:
        """P4-001: Majority, confidence-weighted, and highest-probability differ."""
        preds = [
            _make_prediction("Build", 0.3),
            _make_prediction("Build", 0.3),
            _make_prediction("Build", 0.3),
            _make_prediction("Write", 0.95),
            _make_prediction("Write", 0.95),
        ]
        majority_label, _ = aggregate_interval(preds, strategy="majority")
        weighted_label, _ = aggregate_interval(preds, strategy="confidence_weighted")
        assert majority_label == "Build"
        assert weighted_label == "Write"
        assert majority_label != weighted_label

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_interval([], strategy="majority")

    def test_unknown_strategy_raises(self) -> None:
        pred = _make_prediction("Build", 0.8)
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            aggregate_interval([pred], strategy="nonexistent")

    def test_expb_three_strategies_return_results(self) -> None:
        """EXP-B: All three strategies return valid (label, confidence) tuples."""
        preds = [
            _make_prediction("Build", 0.6),
            _make_prediction("Write", 0.8),
            _make_prediction("Build", 0.5),
        ]
        results = {}
        for strat in ("majority", "confidence_weighted", "highest_probability"):
            label, conf = aggregate_interval(preds, strategy=strat)
            assert isinstance(label, str)
            assert 0.0 <= conf <= 1.0
            results[strat] = label
        assert len(results) == 3
