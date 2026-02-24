"""Tests for labels.projection: strict block-to-window label projection.

Covers: full containment, partial overlap drop, multi-block overlap drop,
user_id filtering, empty inputs, and the complete round-trip expectation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.core.types import LabelSpan
from taskclf.labels.projection import project_blocks_to_windows


def _make_features(
    starts: list[dt.datetime],
    user_id: str = "u1",
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> pd.DataFrame:
    rows = []
    for s in starts:
        rows.append({
            "user_id": user_id,
            "bucket_start_ts": s,
            "bucket_end_ts": s + dt.timedelta(seconds=bucket_seconds),
            "session_id": "sess-1",
        })
    return pd.DataFrame(rows)


class TestFullContainment:
    """Windows fully inside a single block are labeled."""

    def test_single_block_covers_all_windows(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0, t0 + dt.timedelta(minutes=1)])
        span = LabelSpan(
            start_ts=t0 - dt.timedelta(minutes=1),
            end_ts=t0 + dt.timedelta(minutes=5),
            label="Build",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 2
        assert list(result["label"]) == ["Build", "Build"]

    def test_block_exactly_matches_window(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        span = LabelSpan(
            start_ts=t0,
            end_ts=t0 + dt.timedelta(seconds=60),
            label="Debug",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 1
        assert result["label"].iloc[0] == "Debug"


class TestPartialOverlap:
    """Windows only partially overlapping a block are dropped."""

    def test_window_starts_before_block(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        span = LabelSpan(
            start_ts=t0 + dt.timedelta(seconds=30),
            end_ts=t0 + dt.timedelta(minutes=5),
            label="Build",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 0

    def test_window_ends_after_block(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        span = LabelSpan(
            start_ts=t0 - dt.timedelta(minutes=5),
            end_ts=t0 + dt.timedelta(seconds=30),
            label="Build",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 0


class TestMultiBlockOverlap:
    """Windows overlapping multiple blocks are dropped."""

    def test_two_blocks_cover_same_window(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        spans = [
            LabelSpan(
                start_ts=t0,
                end_ts=t0 + dt.timedelta(minutes=2),
                label="Build",
                provenance="manual",
            ),
            LabelSpan(
                start_ts=t0,
                end_ts=t0 + dt.timedelta(minutes=2),
                label="Debug",
                provenance="manual",
            ),
        ]
        result = project_blocks_to_windows(features, spans)
        assert len(result) == 0


class TestUserIdFiltering:
    """Spans with user_id only match features from the same user."""

    def test_different_user_not_matched(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0], user_id="user-A")
        span = LabelSpan(
            start_ts=t0,
            end_ts=t0 + dt.timedelta(minutes=2),
            label="Build",
            provenance="manual",
            user_id="user-B",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 0

    def test_same_user_matched(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0], user_id="user-A")
        span = LabelSpan(
            start_ts=t0,
            end_ts=t0 + dt.timedelta(minutes=2),
            label="Build",
            provenance="manual",
            user_id="user-A",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 1

    def test_span_without_user_matches_any(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0], user_id="user-X")
        span = LabelSpan(
            start_ts=t0,
            end_ts=t0 + dt.timedelta(minutes=2),
            label="Write",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 1


class TestUnlabeled:
    """Unlabeled windows are dropped."""

    def test_no_covering_span(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        span = LabelSpan(
            start_ts=t0 + dt.timedelta(hours=1),
            end_ts=t0 + dt.timedelta(hours=2),
            label="Build",
            provenance="manual",
        )
        result = project_blocks_to_windows(features, [span])
        assert len(result) == 0


class TestEmptyInputs:
    def test_empty_features(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 11, 0),
            label="Build",
            provenance="manual",
        )
        result = project_blocks_to_windows(pd.DataFrame(), [span])
        assert len(result) == 0

    def test_empty_spans(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features([t0])
        result = project_blocks_to_windows(features, [])
        assert len(result) == 0


class TestRoundTrip:
    """Project then verify label assignment matches expectations."""

    def test_mixed_scenario(self) -> None:
        t0 = dt.datetime(2025, 6, 15, 10, 0)
        features = _make_features(
            [t0 + dt.timedelta(minutes=i) for i in range(5)]
        )
        spans = [
            LabelSpan(
                start_ts=t0,
                end_ts=t0 + dt.timedelta(minutes=2),
                label="Build",
                provenance="manual",
            ),
            LabelSpan(
                start_ts=t0 + dt.timedelta(minutes=3),
                end_ts=t0 + dt.timedelta(minutes=5),
                label="Debug",
                provenance="manual",
            ),
        ]
        result = project_blocks_to_windows(features, spans)

        assert len(result) == 4
        labels = list(result["label"])
        assert labels == ["Build", "Build", "Debug", "Debug"]
