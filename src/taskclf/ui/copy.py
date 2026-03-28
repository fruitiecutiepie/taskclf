"""Centralized user-facing copy strings for all UI surfaces.

All text shown to the user in notifications, live status, and gap-fill
prompts is defined here so that labeling conventions (Decision 6) stay
consistent and are easy to update in one place.
"""

from __future__ import annotations


def transition_suggestion_text(label: str, start: str, end: str) -> str:
    """Action-oriented transition prompt with a concrete time range."""
    return f"Was this {label}? {start}\u2013{end}"


def live_status_text(label: str) -> str:
    """Passive present-tense status for the current bucket."""
    return f"Now: {label}"


def gap_fill_prompt(duration_str: str) -> str:
    """Prompt surfaced at idle return or session start."""
    return f"You have {duration_str} unlabeled. Review?"


def gap_fill_detail(start: str, end: str) -> str:
    """Detail line for the gap-fill review surface."""
    return f"Review unlabeled: {start}\u2013{end}"
