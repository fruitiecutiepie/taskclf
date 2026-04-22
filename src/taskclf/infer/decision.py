"""Deterministic rule engine for Mode precedence and Tie-break logic."""

from __future__ import annotations

from typing import Sequence

from taskclf.core.types import (
    AxisDecision,
    CollaborationMode,
    CrossDomainLabel,
    EvidenceSnapshot,
    InteractionStyle,
    Mode,
    OutputDomain,
    Subtype,
    SupportState,
)

# --- Constants & Thresholds ---
CONFIDENCE_NO_EVIDENCE = 0.1
CONFIDENCE_IDLE = 0.9
CONFIDENCE_ATTEND = 0.85
CONFIDENCE_COORDINATE = 0.8
CONFIDENCE_PRODUCE = 0.85
CONFIDENCE_CONSUME = 0.8
CONFIDENCE_FALLBACK = 0.3

PRODUCE_MIN_KEYS = 10
CONSUME_MAX_KEYS = 10

COMMUNICATION_APPS = frozenset(
    [
        "com.tinyspeck.slackmacgap",
        "com.apple.mail",
        "com.microsoft.teams",
    ]
)

EDITOR_TERMINAL_APPS = frozenset(
    [
        "com.microsoft.VSCode",
        "com.apple.Terminal",
        "com.jetbrains.intellij",
    ]
)

BROWSER_APP_SUBSTRINGS = ("chrome", "firefox", "safari")
BROWSER_URL_CATEGORY_SUBSTRING = "browser"


def evaluate_mode(snapshots: Sequence[EvidenceSnapshot]) -> AxisDecision[Mode]:
    """Evaluate Mode using the 5-rule precedence and tie-break logic.

    Order: Idle -> Attend -> Coordinate -> Produce -> Consume.

    Args:
        snapshots: Raw evidence slices for the inference window.

    Returns:
        AxisDecision containing the assigned Mode and reason codes.
    """
    if not snapshots:
        return AxisDecision[Mode](
            value=Mode.Idle,
            confidence=CONFIDENCE_NO_EVIDENCE,
            reason_codes=["no_evidence"],
            alternatives=[],
        )

    # Aggregate signals across the snapshots
    total_duration_ms = sum(s.foreground_duration_ms for s in snapshots)
    total_keys = sum(s.key_events for s in snapshots)
    total_scroll = sum(s.scroll_events for s in snapshots)

    # Note: total_pointer is extracted but currently unused for Mode classification.
    # It is highly ambiguous across Produce/Consume/Coordinate tasks, but it is
    # critical for the upstream low_interaction_idle_signal and future InteractionStyle evaluations.
    _total_pointer = sum(s.pointer_events for s in snapshots)  # noqa: F841

    app_ids = set()
    for s in snapshots:
        app_ids.update(s.app_ids)

    meeting_signal = any(s.meeting_signal for s in snapshots)
    active_call = any(s.active_call for s in snapshots)
    active_mic = any(s.active_mic for s in snapshots)

    is_idle_run = all(s.low_interaction_idle_signal for s in snapshots)

    # -------------------------------------------------------------------------
    # Precedence Rules & Heuristics
    # -------------------------------------------------------------------------

    # 1. Idle check
    # Restorative, non-task-directed, or no meaningful evidence of directed work.
    if is_idle_run and total_duration_ms > 0:
        return AxisDecision[Mode](
            value=Mode.Idle,
            confidence=CONFIDENCE_IDLE,
            reason_codes=["low_interaction_idle_signal"],
        )

    # 2. Attend check
    # Live synchronous session.
    if meeting_signal or active_call or active_mic:
        return AxisDecision[Mode](
            value=Mode.Attend,
            confidence=CONFIDENCE_ATTEND,
            reason_codes=["meeting_signal", "active_call"],
            alternatives=[Mode.Coordinate],
        )

    # 3. Coordinate check
    # Slack, email, Jira, scheduling.
    is_comm_heavy = any(app in COMMUNICATION_APPS for app in app_ids)
    if is_comm_heavy:
        return AxisDecision[Mode](
            value=Mode.Coordinate,
            confidence=CONFIDENCE_COORDINATE,
            reason_codes=["communication_app_active"],
            alternatives=[Mode.Produce],
        )

    # 4. Produce check
    # Coding, writing, synthesizing. High typing/shortcuts.
    is_editor_terminal = any(app in EDITOR_TERMINAL_APPS for app in app_ids)
    if is_editor_terminal and total_keys > PRODUCE_MIN_KEYS:
        return AxisDecision[Mode](
            value=Mode.Produce,
            confidence=CONFIDENCE_PRODUCE,
            reason_codes=["editor_active", "high_typing"],
            alternatives=[Mode.Consume],
        )

    # 5. Consume check
    # Reading docs, watching, inspecting. High scroll, low typing.
    is_browser = any(
        BROWSER_URL_CATEGORY_SUBSTRING in s.browser_url_category.lower()
        for s in snapshots
        if s.browser_url_category
    ) or any(sub in app.lower() for app in app_ids for sub in BROWSER_APP_SUBSTRINGS)

    if is_browser and total_scroll > 0 and total_keys < CONSUME_MAX_KEYS:
        return AxisDecision[Mode](
            value=Mode.Consume,
            confidence=CONFIDENCE_CONSUME,
            reason_codes=["browser_active", "high_scroll", "low_typing"],
            alternatives=[Mode.Produce],
        )

    # Fallback to MixedUnknown handling via alternatives/confidence
    return AxisDecision[
        Mode
    ](
        value=Mode.Idle,  # Default fallback if literally nothing matched but not strict idle
        confidence=CONFIDENCE_FALLBACK,
        reason_codes=["no_strong_signal"],
        alternatives=[Mode.Produce, Mode.Consume],
    )


def restrict_subtype_for_mode(
    mode: Mode, requested_subtype: Subtype | None
) -> Subtype | None:
    """Enforce the Subtype policy based on the dominant Mode."""
    if requested_subtype is None:
        return None

    policy = {
        Mode.Produce: {
            Subtype.Build,
            Subtype.Debug,
            Subtype.Write,
            Subtype.Review,
            Subtype.Analyze,
            Subtype.Plan,
            Subtype.Admin,
        },
        Mode.Consume: {
            Subtype.ReadResearch,
            Subtype.Learn,
            Subtype.ExploreReference,
            Subtype.Review,
            Subtype.Monitor,
            Subtype.Analyze,
        },
        Mode.Coordinate: {
            Subtype.Communicate,
            Subtype.Plan,
            Subtype.Admin,
            Subtype.Review,
        },
        Mode.Attend: {Subtype.Meet, Subtype.Learn, Subtype.Communicate},
        Mode.Idle: {Subtype.BreakIdle},
    }

    allowed = policy.get(mode, set())
    if requested_subtype in allowed:
        return requested_subtype

    return None


def resolve_cross_domain_label(
    snapshots: Sequence[EvidenceSnapshot],
    requested_subtype: Subtype | None = None,
    requested_interaction: InteractionStyle | None = None,
    requested_collaboration: CollaborationMode | None = None,
    requested_domain: OutputDomain | None = None,
    escalation_threshold: float = 0.5,
) -> CrossDomainLabel:
    """Resolve a complete CrossDomainLabel from evidence with escalation checks.

    If the resolved Mode's confidence falls below the escalation_threshold,
    the support_state is escalated to MixedUnknown as a diagnostic trigger.
    """
    mode_decision = evaluate_mode(snapshots)

    valid_subtype = restrict_subtype_for_mode(mode_decision.value, requested_subtype)

    # Simple defaults for optional axes if none requested (in a real pipeline these might be inferred)
    interaction_decision = None
    if requested_interaction:
        interaction_decision = AxisDecision[InteractionStyle](
            value=requested_interaction, confidence=1.0
        )

    collaboration_decision = None
    if requested_collaboration:
        collaboration_decision = AxisDecision[CollaborationMode](
            value=requested_collaboration, confidence=1.0
        )

    domain_decision = None
    if requested_domain:
        domain_decision = AxisDecision[OutputDomain](
            value=requested_domain, confidence=1.0
        )

    subtype_decision = None
    if valid_subtype:
        subtype_decision = AxisDecision[Subtype](value=valid_subtype, confidence=1.0)

    support_state = SupportState.Supported
    if mode_decision.confidence < escalation_threshold:
        support_state = SupportState.MixedUnknown
    elif mode_decision.confidence < 0.7:
        support_state = SupportState.WeakEvidence

    return CrossDomainLabel(
        mode=mode_decision,
        subtype=subtype_decision,
        interaction_style=interaction_decision,
        collaboration_mode=collaboration_decision,
        output_domain=domain_decision,
        support_state=support_state,
    )
