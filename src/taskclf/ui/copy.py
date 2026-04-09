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


def activity_source_setup_title() -> str:
    """Title shown when the configured activity source is unavailable."""
    return "Activity source unavailable"


def activity_source_setup_message() -> str:
    """Non-blocking guidance shown when activity summaries cannot be loaded."""
    return (
        "Manual labeling still works, but activity summaries and automatic "
        "activity tracking are unavailable until this source is set up."
    )


def activity_source_setup_steps(endpoint: str) -> list[str]:
    """Step-by-step setup guidance for the current activity source endpoint."""
    return [
        "Install and start ActivityWatch.",
        f"Confirm the local server is reachable at {endpoint}.",
        "If you use a custom host, update aw_host in config.toml and restart taskclf.",
    ]


def activity_source_setup_help_url() -> str:
    """Public help page for activity-source setup."""
    return "https://activitywatch.net/"
