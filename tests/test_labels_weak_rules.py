"""Tests for labels.weak_rules: heuristic weak-labeling rules.

Covers: WeakRule construction/validation, match_rule priority and edge
cases, apply_weak_rules span merging and gap handling,
build_default_rules validity.

TC-WEAK-RULE-001..004 (WeakRule),
TC-WEAK-MATCH-001..005 (match_rule),
TC-WEAK-APPLY-001..008 (apply_weak_rules),
TC-WEAK-BUILD-001..002 (build_default_rules).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.core.types import LABEL_SET_V1
from taskclf.labels.weak_rules import (
    APP_CATEGORY_RULES,
    APP_ID_RULES,
    DOMAIN_CATEGORY_RULES,
    WeakRule,
    apply_weak_rules,
    build_default_rules,
    match_rule,
)

_BASE_DATE = dt.date(2025, 6, 15)


def _ts(hour: int, minute: int) -> dt.datetime:
    return dt.datetime(_BASE_DATE.year, _BASE_DATE.month, _BASE_DATE.day, hour, minute)


def _feature_row(
    bucket_start: dt.datetime,
    *,
    app_id: str = "com.example.unknown",
    app_category: str = "other",
    domain_category: str = "non_browser",
) -> dict:
    """Minimal feature row dict with the columns weak rules inspect."""
    return {
        "bucket_start_ts": bucket_start,
        "bucket_end_ts": bucket_start + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS),
        "app_id": app_id,
        "app_category": app_category,
        "domain_category": domain_category,
    }


def _features_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# WeakRule construction / validation
# ---------------------------------------------------------------------------


class TestWeakRule:
    def test_valid_construction(self) -> None:
        """TC-WEAK-RULE-001: valid rule builds without error."""
        rule = WeakRule(name="test", field="app_id", pattern="com.x.Y", label="Build")
        assert rule.name == "test"
        assert rule.field == "app_id"
        assert rule.pattern == "com.x.Y"
        assert rule.label == "Build"
        assert rule.confidence is None

    def test_with_confidence(self) -> None:
        """TC-WEAK-RULE-002: confidence is stored."""
        rule = WeakRule(
            name="test", field="app_id", pattern="com.x.Y", label="Build", confidence=0.7
        )
        assert rule.confidence == pytest.approx(0.7)

    def test_invalid_label_rejected(self) -> None:
        """TC-WEAK-RULE-003: label not in LABEL_SET_V1 raises ValueError."""
        with pytest.raises(ValueError, match="unknown label"):
            WeakRule(name="bad", field="app_id", pattern="com.x.Y", label="InvalidLabel")

    def test_frozen(self) -> None:
        """TC-WEAK-RULE-004: frozen dataclass prevents mutation."""
        rule = WeakRule(name="test", field="app_id", pattern="com.x.Y", label="Build")
        with pytest.raises(AttributeError):
            rule.label = "Debug"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# match_rule
# ---------------------------------------------------------------------------


class TestMatchRule:
    def test_app_id_match(self) -> None:
        """TC-WEAK-MATCH-001: exact app_id match returns label."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        row = _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")
        result = match_rule(row, rules)
        assert result == ("Build", "vscode")

    def test_app_category_match(self) -> None:
        """TC-WEAK-MATCH-002: app_category match returns label."""
        rules = [WeakRule(name="chat", field="app_category", pattern="chat", label="Communicate")]
        row = _feature_row(_ts(10, 0), app_category="chat")
        result = match_rule(row, rules)
        assert result == ("Communicate", "chat")

    def test_domain_category_match(self) -> None:
        """TC-WEAK-MATCH-003: domain_category match returns label."""
        rules = [
            WeakRule(
                name="code_host", field="domain_category", pattern="code_hosting", label="Build"
            )
        ]
        row = _feature_row(_ts(10, 0), domain_category="code_hosting")
        result = match_rule(row, rules)
        assert result == ("Build", "code_host")

    def test_first_match_wins(self) -> None:
        """TC-WEAK-MATCH-004: when multiple rules match, first wins."""
        rules = [
            WeakRule(name="by_app_id", field="app_id", pattern="com.google.Chrome", label="ReadResearch"),
            WeakRule(name="by_domain", field="domain_category", pattern="code_hosting", label="Build"),
        ]
        row = _feature_row(
            _ts(10, 0), app_id="com.google.Chrome", domain_category="code_hosting"
        )
        result = match_rule(row, rules)
        assert result is not None
        assert result[1] == "by_app_id"

    def test_no_match_returns_none(self) -> None:
        """TC-WEAK-MATCH-005: unmatched row returns None."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        row = _feature_row(_ts(10, 0), app_id="com.unknown.App")
        assert match_rule(row, rules) is None

    def test_missing_field_returns_none(self) -> None:
        """Row missing the field key does not crash."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        result = match_rule({"bucket_start_ts": _ts(10, 0)}, rules)
        assert result is None

    def test_empty_rules_returns_none(self) -> None:
        """Empty rules list returns None."""
        row = _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")
        assert match_rule(row, []) is None


# ---------------------------------------------------------------------------
# apply_weak_rules
# ---------------------------------------------------------------------------


class TestApplyWeakRules:
    def test_single_row_produces_one_span(self) -> None:
        """TC-WEAK-APPLY-001: one matching row → one span."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        df = _features_df([_feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")])
        spans = apply_weak_rules(df, rules=rules)
        assert len(spans) == 1
        assert spans[0].label == "Build"
        assert spans[0].provenance == "weak:vscode"
        assert spans[0].start_ts == _ts(10, 0)
        assert spans[0].end_ts == _ts(10, 0) + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)

    def test_consecutive_same_label_merged(self) -> None:
        """TC-WEAK-APPLY-002: consecutive same-label buckets merge."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [
            _feature_row(_ts(10, i), app_id="com.microsoft.VSCode")
            for i in range(3)
        ]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert len(spans) == 1
        assert spans[0].start_ts == _ts(10, 0)
        assert spans[0].end_ts == _ts(10, 2) + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)

    def test_different_labels_produce_separate_spans(self) -> None:
        """TC-WEAK-APPLY-003: label change creates a new span."""
        rules = [
            WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build"),
            WeakRule(name="slack", field="app_id", pattern="com.tinyspeck.slackmacgap", label="Communicate"),
        ]
        rows = [
            _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 1), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 2), app_id="com.tinyspeck.slackmacgap"),
            _feature_row(_ts(10, 3), app_id="com.tinyspeck.slackmacgap"),
        ]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert len(spans) == 2
        assert spans[0].label == "Build"
        assert spans[1].label == "Communicate"

    def test_no_matches_empty_result(self) -> None:
        """TC-WEAK-APPLY-004: no matches → empty list."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [_feature_row(_ts(10, 0), app_id="com.unknown.App")]
        assert apply_weak_rules(_features_df(rows), rules=rules) == []

    def test_empty_dataframe(self) -> None:
        """TC-WEAK-APPLY-005: empty DataFrame → empty list."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        assert apply_weak_rules(pd.DataFrame(), rules=rules) == []

    def test_user_id_propagated(self) -> None:
        """TC-WEAK-APPLY-006: user_id is attached to all spans."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [_feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")]
        spans = apply_weak_rules(_features_df(rows), rules=rules, user_id="test-user")
        assert spans[0].user_id == "test-user"

    def test_confidence_propagated(self) -> None:
        """TC-WEAK-APPLY-007: confidence from rule is attached to spans."""
        rules = [
            WeakRule(
                name="vscode",
                field="app_id",
                pattern="com.microsoft.VSCode",
                label="Build",
                confidence=0.6,
            )
        ]
        rows = [_feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert spans[0].confidence == pytest.approx(0.6)

    def test_gap_breaks_span(self) -> None:
        """TC-WEAK-APPLY-008: non-contiguous buckets produce separate spans."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [
            _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 5), app_id="com.microsoft.VSCode"),
        ]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert len(spans) == 2

    def test_unmatched_row_in_middle_breaks_span(self) -> None:
        """Unmatched row between two matching rows creates two spans."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [
            _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 1), app_id="com.unknown.App"),
            _feature_row(_ts(10, 2), app_id="com.microsoft.VSCode"),
        ]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert len(spans) == 2
        assert spans[0].end_ts == _ts(10, 0) + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)
        assert spans[1].start_ts == _ts(10, 2)

    def test_lockscreen_category_produces_break_idle(self) -> None:
        """Lockscreen app_category is auto-labeled BreakIdle."""
        rows = [
            _feature_row(_ts(10, 0), app_category="lockscreen"),
            _feature_row(_ts(10, 1), app_category="lockscreen"),
            _feature_row(_ts(10, 2), app_category="lockscreen"),
        ]
        spans = apply_weak_rules(_features_df(rows))
        assert len(spans) == 1
        assert spans[0].label == "BreakIdle"
        assert "lockscreen" in spans[0].provenance

    def test_default_rules_used_when_none(self) -> None:
        """When rules=None, build_default_rules() is used."""
        rows = [_feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")]
        spans = apply_weak_rules(_features_df(rows))
        assert len(spans) == 1
        assert spans[0].label == "Build"

    def test_provenance_format(self) -> None:
        """Provenance is 'weak:<rule_name>'."""
        rules = [
            WeakRule(
                name="app_id:com.microsoft.VSCode",
                field="app_id",
                pattern="com.microsoft.VSCode",
                label="Build",
            )
        ]
        rows = [_feature_row(_ts(10, 0), app_id="com.microsoft.VSCode")]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert spans[0].provenance == "weak:app_id:com.microsoft.VSCode"

    def test_unsorted_input_sorted_by_bucket_start(self) -> None:
        """Input rows not sorted by time are sorted before processing."""
        rules = [WeakRule(name="vscode", field="app_id", pattern="com.microsoft.VSCode", label="Build")]
        rows = [
            _feature_row(_ts(10, 2), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 0), app_id="com.microsoft.VSCode"),
            _feature_row(_ts(10, 1), app_id="com.microsoft.VSCode"),
        ]
        spans = apply_weak_rules(_features_df(rows), rules=rules)
        assert len(spans) == 1
        assert spans[0].start_ts == _ts(10, 0)
        assert spans[0].end_ts == _ts(10, 2) + dt.timedelta(seconds=DEFAULT_BUCKET_SECONDS)


# ---------------------------------------------------------------------------
# build_default_rules
# ---------------------------------------------------------------------------


class TestBuildDefaultRules:
    def test_non_empty(self) -> None:
        """TC-WEAK-BUILD-001: returns a non-empty list."""
        rules = build_default_rules()
        assert len(rules) > 0

    def test_all_labels_valid(self) -> None:
        """TC-WEAK-BUILD-002: every rule has a valid label."""
        for rule in build_default_rules():
            assert rule.label in LABEL_SET_V1, f"Rule {rule.name} has invalid label {rule.label}"

    def test_count_matches_maps(self) -> None:
        """Total rules equals sum of all three maps."""
        rules = build_default_rules()
        expected = len(APP_ID_RULES) + len(APP_CATEGORY_RULES) + len(DOMAIN_CATEGORY_RULES)
        assert len(rules) == expected

    def test_app_id_rules_first(self) -> None:
        """App ID rules come before category and domain rules."""
        rules = build_default_rules()
        app_id_count = len(APP_ID_RULES)
        for rule in rules[:app_id_count]:
            assert rule.field == "app_id"

    def test_domain_rules_last(self) -> None:
        """Domain category rules come last."""
        rules = build_default_rules()
        domain_count = len(DOMAIN_CATEGORY_RULES)
        for rule in rules[-domain_count:]:
            assert rule.field == "domain_category"
