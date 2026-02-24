"""Tests for per-user/day reject rate grouping and drift detection.

Covers: TC-REJECT-008 (grouping logic), TC-REJECT-009 (drift flags),
TC-REJECT-010 (edge cases).
"""

from __future__ import annotations

from datetime import datetime

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.metrics import reject_rate_by_group


# ---------------------------------------------------------------------------
# TC-REJECT-008: grouping logic
# ---------------------------------------------------------------------------

class TestRejectRateByGroupBasic:
    def test_single_user_single_day(self) -> None:
        labels = ["Build", MIXED_UNKNOWN, "Build", "Build"]
        user_ids = ["u1"] * 4
        timestamps = [
            datetime(2025, 6, 14, 9, 0),
            datetime(2025, 6, 14, 9, 1),
            datetime(2025, 6, 14, 9, 2),
            datetime(2025, 6, 14, 9, 3),
        ]

        result = reject_rate_by_group(labels, user_ids, timestamps)

        assert "u1|2025-06-14" in result["per_group"]
        grp = result["per_group"]["u1|2025-06-14"]
        assert grp["total"] == 4
        assert grp["rejected"] == 1
        assert abs(grp["reject_rate"] - 0.25) < 0.01

    def test_multiple_users_multiple_days(self) -> None:
        labels = ["Build", MIXED_UNKNOWN, "Write", MIXED_UNKNOWN]
        user_ids = ["u1", "u1", "u2", "u2"]
        timestamps = [
            datetime(2025, 6, 14, 9, 0),
            datetime(2025, 6, 14, 9, 1),
            datetime(2025, 6, 15, 10, 0),
            datetime(2025, 6, 15, 10, 1),
        ]

        result = reject_rate_by_group(labels, user_ids, timestamps)

        assert len(result["per_group"]) == 2
        assert "u1|2025-06-14" in result["per_group"]
        assert "u2|2025-06-15" in result["per_group"]

    def test_global_reject_rate_matches(self) -> None:
        labels = [MIXED_UNKNOWN, "Build", "Build", "Build"]
        user_ids = ["u1"] * 4
        timestamps = [datetime(2025, 6, 14, h, 0) for h in range(9, 13)]

        result = reject_rate_by_group(labels, user_ids, timestamps)
        assert abs(result["global_reject_rate"] - 0.25) < 0.01


# ---------------------------------------------------------------------------
# TC-REJECT-009: drift flags
# ---------------------------------------------------------------------------

class TestRejectRateByGroupDrift:
    def test_spike_is_flagged(self) -> None:
        labels = (
            ["Build"] * 20                          # u1 day1: 0% reject
            + [MIXED_UNKNOWN] * 5 + ["Build"] * 5   # u2 day2: 50% reject (spike)
        )
        user_ids = ["u1"] * 20 + ["u2"] * 10
        timestamps = (
            [datetime(2025, 6, 14, 9, i) for i in range(20)]
            + [datetime(2025, 6, 15, 9, i) for i in range(10)]
        )

        result = reject_rate_by_group(labels, user_ids, timestamps)

        assert "u2|2025-06-15" in result["drift_flags"]

    def test_no_spike_when_uniform(self) -> None:
        labels = [MIXED_UNKNOWN, "Build"] * 5
        user_ids = ["u1"] * 10
        timestamps = [datetime(2025, 6, 14, 9, i) for i in range(10)]

        result = reject_rate_by_group(labels, user_ids, timestamps)
        assert result["drift_flags"] == []


# ---------------------------------------------------------------------------
# TC-REJECT-010: edge cases
# ---------------------------------------------------------------------------

class TestRejectRateByGroupEdge:
    def test_empty_inputs(self) -> None:
        result = reject_rate_by_group([], [], [])
        assert result["global_reject_rate"] == 0.0
        assert result["per_group"] == {}
        assert result["drift_flags"] == []

    def test_all_rejected(self) -> None:
        labels = [MIXED_UNKNOWN] * 5
        user_ids = ["u1"] * 5
        timestamps = [datetime(2025, 6, 14, 9, i) for i in range(5)]

        result = reject_rate_by_group(labels, user_ids, timestamps)
        assert result["global_reject_rate"] == 1.0
        grp = result["per_group"]["u1|2025-06-14"]
        assert grp["reject_rate"] == 1.0

    def test_none_rejected(self) -> None:
        labels = ["Build"] * 5
        user_ids = ["u1"] * 5
        timestamps = [datetime(2025, 6, 14, 9, i) for i in range(5)]

        result = reject_rate_by_group(labels, user_ids, timestamps)
        assert result["global_reject_rate"] == 0.0
        assert result["drift_flags"] == []
