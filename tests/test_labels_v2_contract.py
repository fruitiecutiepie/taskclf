from __future__ import annotations

import datetime as dt

import pandas as pd

from taskclf.cli.migrate_labels import migrate_row
from taskclf.core.types import (
    ActivitySurface,
    ArtifactTouch,
    CollaborationSurface,
    EvidenceSnapshot,
    IntentBasis,
    Mode,
    ModeSource,
    SupportState,
    SyncPresence,
    Subtype,
)
from taskclf.infer.decision import derive_observed_label, resolve_semantic_label


def _snapshot(**overrides: object) -> EvidenceSnapshot:
    snapshot = EvidenceSnapshot(
        bucket_start_ts=dt.datetime(2026, 4, 25, 0, 0, tzinfo=dt.timezone.utc),
        bucket_end_ts=dt.datetime(2026, 4, 25, 0, 1, tzinfo=dt.timezone.utc),
        app_ids=["com.apple.Safari"],
        window_title_hashes=["abc123"],
        foreground_duration_ms=60000,
        key_events=1,
        pointer_events=2,
        scroll_events=5,
        app_switch_count=0,
        active_call=False,
        active_mic=False,
        active_camera=False,
        browser_url_category="browser_docs",
        file_types=None,
        meeting_signal=False,
        build_or_run_signal=None,
        test_signal=None,
        low_interaction_idle_signal=False,
    )
    if not overrides:
        return snapshot

    return EvidenceSnapshot.model_validate(snapshot.model_dump() | overrides)


def test_derive_observed_label_for_live_call_marks_sync_presence() -> None:
    observed = derive_observed_label(
        [
            _snapshot(
                app_ids=["us.zoom.xos"],
                browser_url_category=None,
                scroll_events=0,
                active_call=True,
                meeting_signal=True,
            )
        ]
    )

    assert observed.activity_surface == ActivitySurface.Call
    assert observed.artifact_touch == ArtifactTouch.None_
    assert observed.sync_presence == SyncPresence.LiveHumanSession
    assert observed.collaboration_surface == CollaborationSurface.SyncVoiceVideo


def test_resolve_semantic_label_for_browser_read_uses_inferred_context() -> None:
    label = resolve_semantic_label([_snapshot()])

    assert label.mode.value == Mode.Consume
    assert label.support_state == SupportState.Supported
    assert label.intent_basis == IntentBasis.InferredFromContext
    assert label.mode_source == ModeSource.DeterministicRule


def test_migrate_row_emits_semantic_envelope_with_manual_provenance() -> None:
    envelope = migrate_row(
        pd.Series(
            {
                "label": "Build",
                "start_ts": "2026-04-25T00:00:00Z",
                "end_ts": "2026-04-25T00:03:00Z",
                "confidence": 0.9,
                "provenance": "manual",
            }
        ),
        "2026-04-25.1",
    )

    assert envelope.observed is None
    assert envelope.semantic.mode.value == Mode.Produce
    assert envelope.semantic.subtype is not None
    assert envelope.semantic.subtype.value == Subtype.Build
    assert envelope.semantic.intent_basis == IntentBasis.UserDeclared
    assert envelope.semantic.mode_source == ModeSource.UserOverride

    dumped = envelope.model_dump(mode="json", exclude_none=True)
    assert "label" not in dumped
    assert "semantic" in dumped


def test_migrate_row_mixed_unknown_stays_conservative() -> None:
    envelope = migrate_row(
        pd.Series({"label": "Mixed/Unknown", "confidence": 1.0}),
        "2026-04-25.1",
    )

    assert envelope.semantic.support_state == SupportState.MixedUnknown
    assert envelope.semantic.intent_basis == IntentBasis.Unknown
    assert envelope.semantic.mode_source == ModeSource.DeterministicRule
