"""Heuristic weak-labeling rules that map feature rows to task-type labels.

Weak rules provide an automated, low-confidence alternative to manual
labeling.  Each rule inspects a single feature column (e.g. ``app_id``,
``app_category``, ``domain_category``) and, when matched, proposes a
:class:`~taskclf.core.types.LabelSpan` with ``provenance="weak:<rule_name>"``.

Rules are evaluated in list order; the **first** match wins.  The default
rule list is ordered by specificity: ``app_id`` rules first, then
``app_category``, then ``domain_category``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Final, Sequence

import pandas as pd

from taskclf.core.defaults import DEFAULT_BUCKET_SECONDS
from taskclf.core.types import LABEL_SET_V1, LabelSpan

# ---------------------------------------------------------------------------
# Rule data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeakRule:
    """A single heuristic labeling rule.

    Attributes:
        name: Human-readable identifier (also used in ``provenance``).
        field: Feature column to inspect (e.g. ``"app_id"``).
        pattern: Value that the column must equal for the rule to fire.
        label: Task-type label to assign (must be in ``LABEL_SET_V1``).
        confidence: Optional confidence score attached to produced spans.
    """

    name: str
    field: str
    pattern: str
    label: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.label not in LABEL_SET_V1:
            raise ValueError(
                f"WeakRule {self.name!r}: unknown label {self.label!r}; "
                f"must be one of {sorted(LABEL_SET_V1)}"
            )


# ---------------------------------------------------------------------------
# Built-in rule maps
# ---------------------------------------------------------------------------

APP_ID_RULES: Final[dict[str, str]] = {
    "com.apple.Terminal": "Build",
    "com.microsoft.VSCode": "Build",
    "com.jetbrains.intellij": "Build",
    "com.googlecode.iterm2": "Build",
    "org.mozilla.firefox": "ReadResearch",
    "com.google.Chrome": "ReadResearch",
    "com.apple.Safari": "ReadResearch",
    "com.apple.mail": "Communicate",
    "com.tinyspeck.slackmacgap": "Communicate",
    "us.zoom.xos": "Meet",
    "com.apple.Notes": "Write",
    "com.apple.finder": "BreakIdle",
}

APP_CATEGORY_RULES: Final[dict[str, str]] = {
    "meeting": "Meet",
    "chat": "Communicate",
    "email": "Communicate",
    "editor": "Build",
    "terminal": "Build",
    "devtools": "Debug",
    "docs": "Write",
    "design": "Write",
    "media": "BreakIdle",
    "file_manager": "BreakIdle",
}

DOMAIN_CATEGORY_RULES: Final[dict[str, str]] = {
    "code_hosting": "Build",
    "email_web": "Communicate",
    "chat": "Communicate",
    "social": "BreakIdle",
    "video": "BreakIdle",
    "news": "ReadResearch",
    "docs": "ReadResearch",
    "search": "ReadResearch",
    "productivity": "Write",
}


# ---------------------------------------------------------------------------
# Rule construction helpers
# ---------------------------------------------------------------------------


def build_default_rules() -> list[WeakRule]:
    """Build the default rule list ordered by specificity.

    Order: ``app_id`` rules, then ``app_category``, then
    ``domain_category``.  Within each group the iteration order of the
    corresponding dict is preserved.

    Returns:
        List of :class:`WeakRule` instances.
    """
    rules: list[WeakRule] = []
    for app_id, label in APP_ID_RULES.items():
        rules.append(WeakRule(name=f"app_id:{app_id}", field="app_id", pattern=app_id, label=label))
    for cat, label in APP_CATEGORY_RULES.items():
        rules.append(
            WeakRule(name=f"app_category:{cat}", field="app_category", pattern=cat, label=label)
        )
    for dom, label in DOMAIN_CATEGORY_RULES.items():
        rules.append(
            WeakRule(
                name=f"domain_category:{dom}", field="domain_category", pattern=dom, label=label
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_rule(
    row: dict[str, Any],
    rules: Sequence[WeakRule],
) -> tuple[str, str] | None:
    """Match a single feature row against *rules* (first match wins).

    Args:
        row: Feature row as a dict (column name -> value).
        rules: Ordered sequence of rules to evaluate.

    Returns:
        ``(label, rule_name)`` of the first matching rule, or ``None``
        if no rule fires.
    """
    for rule in rules:
        value = row.get(rule.field)
        if value is not None and value == rule.pattern:
            return rule.label, rule.name
    return None


# ---------------------------------------------------------------------------
# Bulk application
# ---------------------------------------------------------------------------


def apply_weak_rules(
    features_df: pd.DataFrame,
    rules: Sequence[WeakRule] | None = None,
    user_id: str | None = None,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[LabelSpan]:
    """Apply weak rules to every row in *features_df* and merge spans.

    Consecutive buckets (ordered by ``bucket_start_ts``) that receive
    the **same label** are merged into a single :class:`LabelSpan`.
    A new span starts whenever the label changes or there is a gap
    between buckets.

    Args:
        features_df: DataFrame with at least ``bucket_start_ts`` and
            ``bucket_end_ts`` columns plus the feature columns
            referenced by *rules*.
        rules: Rule list to apply.  Defaults to
            :func:`build_default_rules`.
        user_id: Optional user id attached to every produced span.
        bucket_seconds: Expected bucket duration; used for gap detection.

    Returns:
        List of :class:`LabelSpan` with
        ``provenance="weak:<rule_name>"``.
    """
    if rules is None:
        rules = build_default_rules()

    if features_df.empty:
        return []

    df = features_df.sort_values("bucket_start_ts").reset_index(drop=True)

    spans: list[LabelSpan] = []
    current_label: str | None = None
    current_rule_name: str | None = None
    current_confidence: float | None = None
    span_start: dt.datetime | None = None
    span_end: dt.datetime | None = None
    expected_gap = dt.timedelta(seconds=bucket_seconds)

    for _, row_series in df.iterrows():
        row = row_series.to_dict()
        bucket_start = row["bucket_start_ts"]
        bucket_end = row.get("bucket_end_ts", bucket_start + expected_gap)

        match = match_rule(row, rules)
        if match is None:
            if current_label is not None:
                spans.append(
                    LabelSpan(
                        start_ts=span_start,  # type: ignore[arg-type]
                        end_ts=span_end,  # type: ignore[arg-type]
                        label=current_label,
                        provenance=f"weak:{current_rule_name}",
                        user_id=user_id,
                        confidence=current_confidence,
                    )
                )
                current_label = None
                current_rule_name = None
                current_confidence = None
                span_start = None
                span_end = None
            continue

        label, rule_name = match
        confidence = next((r.confidence for r in rules if r.name == rule_name), None)

        is_contiguous = (
            span_end is not None and bucket_start <= span_end + dt.timedelta(seconds=1)
        )

        if label == current_label and is_contiguous:
            span_end = bucket_end
        else:
            if current_label is not None:
                spans.append(
                    LabelSpan(
                        start_ts=span_start,  # type: ignore[arg-type]
                        end_ts=span_end,  # type: ignore[arg-type]
                        label=current_label,
                        provenance=f"weak:{current_rule_name}",
                        user_id=user_id,
                        confidence=current_confidence,
                    )
                )
            current_label = label
            current_rule_name = rule_name
            current_confidence = confidence
            span_start = bucket_start
            span_end = bucket_end

    if current_label is not None:
        spans.append(
            LabelSpan(
                start_ts=span_start,  # type: ignore[arg-type]
                end_ts=span_end,  # type: ignore[arg-type]
                label=current_label,
                provenance=f"weak:{current_rule_name}",
                user_id=user_id,
                confidence=current_confidence,
            )
        )

    return spans
