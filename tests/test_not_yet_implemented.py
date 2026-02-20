"""Placeholder tests for features not yet implemented.

Each test is skipped with a reason pointing to the stub module that needs
implementation.  Remove the skip marker once the feature is built.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# TC-CORE-005: raw title opt-in policy (config gating)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once title_policy config is implemented in core/types.py")
def test_tc_core_005_raw_title_opt_in() -> None:
    """TC-CORE-005: allow raw title only when config title_policy=raw_opt_in."""


# ---------------------------------------------------------------------------
# TC-CORE-011: salted hashing for window titles
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once salted hashing is implemented in core/hashing.py")
def test_tc_core_011_different_salts_yield_different_hashes() -> None:
    """TC-CORE-011: different salts produce different hashes."""


# ---------------------------------------------------------------------------
# TC-TIME-*: bucketization / sessionization (core/time.py is a stub)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once core/time.py bucketization is implemented")
def test_tc_time_001_bucket_alignment() -> None:
    """TC-TIME-001: e.g. 12:00:37 -> 12:00:00 for 60s buckets."""


@pytest.mark.skip(reason="TODO: remove .skip once core/time.py bucketization is implemented")
def test_tc_time_002_boundary_on_exact_bucket() -> None:
    """TC-TIME-002: timestamp exactly on boundary stays stable."""


@pytest.mark.skip(reason="TODO: remove .skip once core/time.py bucketization is implemented")
def test_tc_time_003_day_rollover() -> None:
    """TC-TIME-003: 23:59:30 -> next day handled correctly."""


@pytest.mark.skip(reason="TODO: remove .skip once core/time.py bucketization is implemented")
def test_tc_time_004_dst_transition() -> None:
    """TC-TIME-004: DST transitions don't create duplicate buckets."""


# ---------------------------------------------------------------------------
# TC-FEAT-*: advanced feature computation (features/windows.py, features/text.py stubs)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once features/windows.py rolling-window aggregations are implemented")
def test_tc_feat_002_app_switch_count_last_5m() -> None:
    """TC-FEAT-002: app switch counts in last 5 minutes match expected."""


@pytest.mark.skip(reason="TODO: remove .skip once features/build.py handles real idle events")
def test_tc_feat_003_idle_segments_produce_zero_active() -> None:
    """TC-FEAT-003: idle segments produce active_seconds=0 and correct idle flags."""


@pytest.mark.skip(reason="TODO: remove .skip once features/text.py title featurization is implemented")
def test_tc_feat_004_title_featurization_uses_hash_only() -> None:
    """TC-FEAT-004: window title featurization uses hash/tokenization only."""


@pytest.mark.skip(reason="TODO: remove .skip once features/windows.py rolling windows are implemented")
def test_tc_feat_005_rolling_window_start_of_day() -> None:
    """TC-FEAT-005: rolling-window features are consistent at start-of-day."""


# ---------------------------------------------------------------------------
# TC-INF-*: smoothing and segmentization (infer/smooth.py is a stub)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="TODO: remove .skip once infer/smooth.py smoothing is implemented")
def test_tc_inf_001_rolling_majority_reduces_spikes() -> None:
    """TC-INF-001: rolling majority smoothing reduces short spikes."""


@pytest.mark.skip(reason="TODO: remove .skip once infer/smooth.py segmentization is implemented")
def test_tc_inf_002_segmentization_merges_adjacent() -> None:
    """TC-INF-002: segmentization merges adjacent identical labels."""


@pytest.mark.skip(reason="TODO: remove .skip once infer/smooth.py segmentization is implemented")
def test_tc_inf_003_segments_ordered_nonoverlapping_full_coverage() -> None:
    """TC-INF-003: segments are strictly ordered, non-overlapping, cover all predicted buckets."""


@pytest.mark.skip(reason="TODO: remove .skip once infer/smooth.py segmentization is implemented")
def test_tc_inf_004_segment_durations_match_bucket_counts() -> None:
    """TC-INF-004: segment durations match bucket counts * bucket_size."""
