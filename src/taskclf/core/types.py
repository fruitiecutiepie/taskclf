"""Core data contracts: Event protocol, feature rows, and label spans."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Final, Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    create_model,
    field_validator,
    model_validator,
)

from taskclf.core.defaults import (
    DEFAULT_TITLE_CHAR3_SKETCH_BUCKETS,
    DEFAULT_TITLE_TOKEN_SKETCH_BUCKETS,
)
from taskclf.core.time import ts_utc_aware_get


@runtime_checkable
class Event(Protocol):
    """Normalized activity event contract.

    Any adapter event type that exposes these read-only attributes
    satisfies the protocol -- no inheritance required.  For example,
    :class:`~taskclf.adapters.activitywatch.types.AWEvent` is a valid
    ``Event`` without importing or subclassing this protocol.
    """

    @property
    def timestamp(self) -> datetime: ...
    @property
    def duration_seconds(self) -> float: ...
    @property
    def app_id(self) -> str: ...
    @property
    def window_title_hash(self) -> str: ...
    @property
    def is_browser(self) -> bool: ...
    @property
    def is_editor(self) -> bool: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def app_category(self) -> str: ...


class CoreLabel(StrEnum):
    """Canonical task-type labels (v1).

    Member ordering matches ``schema/labels_v1.json`` label IDs.
    Do NOT reorder or remove members without a version bump.
    """

    Build = "Build"
    Debug = "Debug"
    Review = "Review"
    Write = "Write"
    ReadResearch = "ReadResearch"
    Communicate = "Communicate"
    Meet = "Meet"
    BreakIdle = "BreakIdle"


LABEL_SET_V1: Final[frozenset[str]] = frozenset(CoreLabel)


T = TypeVar("T", bound=str)


class Mode(StrEnum):
    Produce = "Produce"
    Consume = "Consume"
    Coordinate = "Coordinate"
    Attend = "Attend"
    Idle = "Idle"


class Subtype(StrEnum):
    Build = "Build"
    Debug = "Debug"
    Write = "Write"
    Review = "Review"
    ReadResearch = "ReadResearch"
    Communicate = "Communicate"
    Meet = "Meet"
    Admin = "Admin"
    Plan = "Plan"
    BreakIdle = "BreakIdle"
    Analyze = "Analyze"
    Learn = "Learn"
    ExploreReference = "ExploreReference"
    Monitor = "Monitor"


class InteractionStyle(StrEnum):
    Active = "Active"
    Passive = "Passive"
    Mixed = "Mixed"
    Idle = "Idle"


class CollaborationMode(StrEnum):
    Solo = "Solo"
    AsyncCollab = "AsyncCollab"
    SyncCollab = "SyncCollab"
    Unknown = "Unknown"


class OutputDomain(StrEnum):
    Code = "Code"
    Writing = "Writing"
    Research = "Research"
    Admin = "Admin"
    Communication = "Communication"
    Design = "Design"
    Analysis = "Analysis"
    Operations = "Operations"
    Unknown = "Unknown"


class SupportState(StrEnum):
    Supported = "Supported"
    WeakEvidence = "WeakEvidence"
    Rejected = "Rejected"
    MixedUnknown = "MixedUnknown"


class IntentBasis(StrEnum):
    ObservedOnly = "ObservedOnly"
    InferredFromContext = "InferredFromContext"
    UserDeclared = "UserDeclared"
    Unknown = "Unknown"


class ModeSource(StrEnum):
    DeterministicRule = "DeterministicRule"
    ProbabilisticModel = "ProbabilisticModel"
    UserOverride = "UserOverride"


class ActivitySurface(StrEnum):
    Edit = "Edit"
    Read = "Read"
    Watch = "Watch"
    Message = "Message"
    Call = "Call"
    Search = "Search"
    IdleLike = "IdleLike"


class ArtifactTouch(StrEnum):
    None_ = "None"
    ReadOnly = "ReadOnly"
    Modified = "Modified"
    Created = "Created"


class SyncPresence(StrEnum):
    None_ = "None"
    LiveHumanSession = "LiveHumanSession"
    LiveStream = "LiveStream"


class CollaborationSurface(StrEnum):
    None_ = "None"
    AsyncText = "AsyncText"
    SyncVoiceVideo = "SyncVoiceVideo"


class AxisDecision(BaseModel, Generic[T]):
    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[T] | None = None
    reason_codes: list[str] | None = None


class UserOverride(BaseModel):
    active: bool = True
    override_mode: Mode | None = None
    override_subtype: Subtype | None = None
    note: str | None = None


class ObservedLabel(BaseModel):
    activity_surface: ActivitySurface
    artifact_touch: ArtifactTouch
    sync_presence: SyncPresence
    collaboration_surface: CollaborationSurface


class SemanticLabel(BaseModel):
    mode: AxisDecision[Mode]
    subtype: AxisDecision[Subtype] | None = None
    interaction_style: AxisDecision[InteractionStyle] | None = None
    collaboration_mode: AxisDecision[CollaborationMode] | None = None
    output_domain: AxisDecision[OutputDomain] | None = None
    support_state: SupportState
    intent_basis: IntentBasis = IntentBasis.Unknown
    mode_source: ModeSource = ModeSource.DeterministicRule
    user_override: UserOverride | None = None


class SoftwareLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "Implement", "DebugIncident", "Refactor", "ReviewCode", "RunTests", "InspectLogs"
    )
    artifact_scope: str | None = (
        None  # e.g. "LocalFile", "MultiFile", "Service", "Unknown"
    )


class ResearchLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "LiteratureReview", "NoteSynthesis", "CitationHunt", "ExperimentReview", "SourceScreening"
    )


class DesignLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "InspirationGathering", "Sketch", "IterateVisual", "ReviewVisual", "Prototype"
    )


class EducationLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "LectureWatching", "StudyReading", "PracticeExercise", "Revision", "AssessmentWork"
    )


class OperationsLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "IncidentMonitoring", "Triage", "RunbookExecution", "SystemCheck", "CapacityReview"
    )


class AnalysisLabels(BaseModel):
    activity: str | None = (
        None  # e.g. "SpreadsheetModeling", "DataCleaning", "DashboardInspection", "Reconciliation", "QuantReview"
    )


class PluginPayload(BaseModel):
    namespace: str
    data: (
        SoftwareLabels
        | ResearchLabels
        | DesignLabels
        | EducationLabels
        | OperationsLabels
        | AnalysisLabels
        | dict[str, Any]
    )


class LabelEnvelope(BaseModel):
    taxonomy_version: str = Field(default="labels_v2")
    rule_version: str
    generated_at: str
    evidence_window_ms: int
    inference_window_ms: int
    observed: ObservedLabel | None = None
    semantic: SemanticLabel
    plugins: list[PluginPayload] | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_label_field(cls, values: object) -> object:
        if isinstance(values, dict) and "semantic" not in values and "label" in values:
            updated = dict(values)
            updated["semantic"] = updated.pop("label")
            return updated
        return values

    @property
    def label(self) -> SemanticLabel:
        """Backward-compatible alias for older callers."""
        return self.semantic


class EvidenceSnapshot(BaseModel, frozen=True):
    """Raw observational signals for a short time slice (e.g. 15-60s)."""

    bucket_start_ts: datetime = Field(description="Start of the slice (UTC).")
    bucket_end_ts: datetime = Field(description="End of the slice (UTC, exclusive).")

    app_ids: list[str] = Field(description="List of app IDs active in this window.")
    window_title_hashes: list[str] = Field(description="List of window title hashes.")
    foreground_duration_ms: int = Field(
        description="Milliseconds of foreground activity."
    )
    key_events: int = Field(default=0, description="Number of keystrokes.")
    pointer_events: int = Field(default=0, description="Number of mouse clicks/moves.")
    scroll_events: int = Field(default=0, description="Number of scroll events.")
    app_switch_count: int = Field(default=0, description="Number of app switches.")
    active_call: bool = Field(
        default=False, description="True if an active call is detected."
    )
    active_mic: bool = Field(default=False, description="True if mic is active.")
    active_camera: bool = Field(default=False, description="True if camera is active.")

    browser_url_category: str | None = None
    file_types: list[str] | None = None
    meeting_signal: bool | None = None
    build_or_run_signal: bool | None = None
    test_signal: bool | None = None
    low_interaction_idle_signal: bool | None = None

    @field_validator("bucket_start_ts", "bucket_end_ts", mode="before")
    @classmethod
    def _ensure_aware_utc(cls, v: object) -> object:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v


class TitlePolicy(StrEnum):
    """Controls whether raw window titles may appear in a :class:`FeatureRow`.

    ``HASH_ONLY`` (default)
        All ``raw_*`` fields are rejected — the standard privacy mode.

    ``RAW_WINDOW_TITLE_OPT_IN``
        ``raw_window_title`` is accepted (but still excluded from
        ``model_dump()`` so it can never leak into ``data/processed/``).
        All other ``raw_*`` fields remain prohibited.

    Pass the policy via Pydantic validation context::

        FeatureRow.model_validate(data, context={"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN})
    """

    HASH_ONLY = "hash_only"
    RAW_WINDOW_TITLE_OPT_IN = "raw_window_title_opt_in"


_PROHIBITED_FIELD_PREFIXES = ("raw_",)
TITLE_TOKEN_SKETCH_FIELDS: Final[tuple[str, ...]] = tuple(
    f"title_token_sketch_{i:03d}" for i in range(DEFAULT_TITLE_TOKEN_SKETCH_BUCKETS)
)
TITLE_CHAR3_SKETCH_FIELDS: Final[tuple[str, ...]] = tuple(
    f"title_char3_sketch_{i:03d}" for i in range(DEFAULT_TITLE_CHAR3_SKETCH_BUCKETS)
)
TITLE_SCALAR_FEATURE_FIELDS: Final[tuple[str, ...]] = (
    "title_char_count",
    "title_token_count",
    "title_unique_token_ratio",
    "title_digit_ratio",
    "title_separator_count",
)
V3_ONLY_FEATURE_FIELDS: Final[frozenset[str]] = frozenset(
    TITLE_TOKEN_SKETCH_FIELDS + TITLE_CHAR3_SKETCH_FIELDS + TITLE_SCALAR_FEATURE_FIELDS
)


class FeatureRowBase(BaseModel, frozen=True):
    """One bucketed observation (typically 60 s).

    All persisted feature rows carry schema metadata so downstream
    consumers can detect silent drift.

    Fields are grouped into four sections:

    - **meta** — ``bucket_start_ts``, ``schema_version``, ``schema_hash``,
      ``source_ids``.
    - **context** — ``app_id``, ``app_category``, ``window_title_hash``,
      ``is_browser``, ``is_editor``, ``is_terminal``,
      ``app_switch_count_last_5m``, ``app_foreground_time_ratio``,
      ``app_change_count``, ``app_dwell_time_seconds``,
      ``app_entropy_5m``, ``app_entropy_15m``,
      ``top2_app_concentration_15m``, ``idle_return_indicator``.
    - **keyboard / mouse** — nullable until the corresponding collector is
      wired (``keys_per_min``, ``backspace_ratio``, ``shortcut_rate``,
      ``clicks_per_min``, ``scroll_events_per_min``, ``mouse_distance``).
    - **activity occupancy** — nullable; derived from ``aw-watcher-input``
      (``active_seconds_keyboard``, ``active_seconds_mouse``,
      ``active_seconds_any``, ``max_idle_run_seconds``, ``event_density``).
    - **temporal** — ``hour_of_day``, ``day_of_week``, ``session_length_so_far``.

    A pre-validator rejects any field whose name starts with ``raw_`` to
    enforce the privacy invariant (no raw keystrokes / titles).  The
    single exception is ``raw_window_title``, which is accepted when
    validation context carries ``title_policy=TitlePolicy.RAW_WINDOW_TITLE_OPT_IN``.
    Even then, the field is excluded from ``model_dump()`` so it cannot
    leak into persisted datasets.
    """

    # -- identity --
    user_id: str = Field(description="Random UUID identifying the user (not PII).")
    device_id: str | None = Field(
        default=None, description="Optional device identifier."
    )
    session_id: str = Field(
        description="Deterministic session identifier derived from user_id + session start."
    )

    # -- meta --
    bucket_start_ts: datetime = Field(description="Start of the 60 s bucket (UTC).")
    bucket_end_ts: datetime = Field(
        description="End of the 60 s bucket (UTC, exclusive)."
    )
    schema_version: str = Field(description="Schema version tag, e.g. 'v1'.")
    schema_hash: str = Field(description="Deterministic hash of the column registry.")
    source_ids: list[str] = Field(
        min_length=1, description="Collector IDs that contributed to this row."
    )

    # -- context --
    app_id: str = Field(
        description="Reverse-domain app identifier, e.g. 'com.apple.Terminal'."
    )
    app_category: str = Field(
        description="Semantic app category, e.g. 'editor', 'chat', 'meeting'."
    )
    window_title_hash: str = Field(description="Hashed window title (never raw).")
    is_browser: bool = Field(description="True if the foreground app is a web browser.")
    is_editor: bool = Field(description="True if the foreground app is a code editor.")
    is_terminal: bool = Field(
        description="True if the foreground app is a terminal emulator."
    )
    app_switch_count_last_5m: int = Field(
        ge=0, description="Number of unique app switches in the last 5 minutes."
    )
    app_foreground_time_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of the bucket the dominant app was foreground.",
    )
    app_change_count: int = Field(
        ge=0, description="Number of app transitions within this bucket."
    )
    app_dwell_time_seconds: float = Field(
        ge=0.0,
        description="Seconds the dominant app has been foreground continuously across consecutive buckets.",
    )
    app_entropy_5m: float | None = Field(
        default=None,
        ge=0.0,
        description="Shannon entropy of app duration distribution over the last 5 minutes.",
    )
    app_entropy_15m: float | None = Field(
        default=None,
        ge=0.0,
        description="Shannon entropy of app duration distribution over the last 15 minutes.",
    )
    top2_app_concentration_15m: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Combined time share of the two most-used apps over the last 15 minutes.",
    )
    idle_return_indicator: bool = Field(
        default=False,
        description="True if this bucket immediately follows an idle gap (i.e., starts a new session).",
    )

    # -- keyboard (nullable until collector is wired) --
    keys_per_min: float | None = Field(
        default=None, description="Keystrokes per minute."
    )
    backspace_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of keystrokes that are backspace.",
    )
    shortcut_rate: float | None = Field(
        default=None, ge=0.0, description="Keyboard shortcuts per minute."
    )

    # -- mouse (nullable until collector is wired) --
    clicks_per_min: float | None = Field(
        default=None, ge=0.0, description="Mouse clicks per minute."
    )
    scroll_events_per_min: float | None = Field(
        default=None, ge=0.0, description="Scroll events per minute."
    )
    mouse_distance: float | None = Field(
        default=None, ge=0.0, description="Mouse distance in pixels."
    )

    # -- activity occupancy (nullable until input collector is wired) --
    active_seconds_keyboard: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds with keyboard activity within this bucket.",
    )
    active_seconds_mouse: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds with mouse activity within this bucket.",
    )
    active_seconds_any: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds with any input activity within this bucket.",
    )
    max_idle_run_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Longest consecutive idle run (seconds) within this bucket.",
    )
    event_density: float | None = Field(
        default=None,
        ge=0.0,
        description="Input events per active second within this bucket.",
    )

    # -- browser domain (item 38) --
    domain_category: str = Field(
        default="unknown",
        description="Privacy-preserving browser domain category (e.g. 'search', 'docs', 'social'); 'non_browser' for non-browser apps.",
    )

    # -- title clustering (item 39) --
    window_title_bucket: int = Field(
        ge=0, le=255, description="Hash-trick bucket (0-255) of window_title_hash."
    )
    title_repeat_count_session: int = Field(
        ge=0,
        description="Number of times this window_title_hash has appeared in the current session.",
    )

    # -- temporal dynamics: rolling means (item 40) --
    keys_per_min_rolling_5: float | None = Field(
        default=None, ge=0.0, description="5-bucket rolling mean of keys_per_min."
    )
    keys_per_min_rolling_15: float | None = Field(
        default=None, ge=0.0, description="15-bucket rolling mean of keys_per_min."
    )
    mouse_distance_rolling_5: float | None = Field(
        default=None, ge=0.0, description="5-bucket rolling mean of mouse_distance."
    )
    mouse_distance_rolling_15: float | None = Field(
        default=None, ge=0.0, description="15-bucket rolling mean of mouse_distance."
    )

    # -- temporal dynamics: deltas (item 40) --
    keys_per_min_delta: float | None = Field(
        default=None, description="Change in keys_per_min from previous bucket."
    )
    clicks_per_min_delta: float | None = Field(
        default=None, description="Change in clicks_per_min from previous bucket."
    )
    mouse_distance_delta: float | None = Field(
        default=None, description="Change in mouse_distance from previous bucket."
    )

    # -- temporal dynamics: extended switch count (item 40) --
    app_switch_count_last_15m: int = Field(
        ge=0, description="Unique app switches in the last 15 minutes."
    )

    # -- temporal --
    hour_of_day: int = Field(
        ge=0, le=23, description="Hour component of bucket_start_ts (0-23)."
    )
    day_of_week: int = Field(
        ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)."
    )
    session_length_so_far: float = Field(
        ge=0.0, description="Minutes since session start."
    )

    # -- opt-in raw title (excluded from serialization) --
    raw_window_title: str | None = Field(
        default=None,
        exclude=True,
        description="Raw window title; only accepted when title_policy=RAW_WINDOW_TITLE_OPT_IN.",
    )

    # -- underlying evidence --
    evidence_snapshots: list[EvidenceSnapshot] | None = Field(
        default=None,
        exclude=True,
        description="The raw 15s-60s evidence slices that compose this 2m-5m inference window.",
    )

    @field_validator("bucket_start_ts", "bucket_end_ts", mode="before")
    @classmethod
    def _ensure_aware_utc(cls, v: object) -> object:
        """Tag naive datetimes as UTC; convert non-UTC aware datetimes."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    @model_validator(mode="before")
    @classmethod
    def reject_prohibited_fields(cls, values: dict, info: ValidationInfo) -> dict:  # type: ignore[override]
        if isinstance(values, dict):
            ctx = info.context or {}
            title_policy = ctx.get("title_policy", TitlePolicy.HASH_ONLY)
            for key in values:
                for prefix in _PROHIBITED_FIELD_PREFIXES:
                    if key.startswith(prefix):
                        if (
                            key == "raw_window_title"
                            and title_policy == TitlePolicy.RAW_WINDOW_TITLE_OPT_IN
                        ):
                            continue
                        raise ValueError(
                            f"Prohibited field '{key}': fields starting with "
                            f"'{prefix}' must not appear in a FeatureRow"
                        )
        return values

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        exclude = kwargs.pop("exclude", None)
        if self.schema_version == "v2":
            if exclude is None:
                exclude = {"user_id"}
            elif isinstance(exclude, dict):
                exclude = {**exclude, "user_id": True}
            else:
                exclude = set(exclude) | {"user_id"}
        if self.schema_version != "v3":
            if exclude is None:
                exclude = set(V3_ONLY_FEATURE_FIELDS)
            elif isinstance(exclude, dict):
                exclude = {
                    **exclude,
                    **{field: True for field in V3_ONLY_FEATURE_FIELDS},
                }
            else:
                exclude = set(exclude) | set(V3_ONLY_FEATURE_FIELDS)
        return super().model_dump(*args, exclude=exclude, **kwargs)


def _feature_row_v3_field_definitions() -> dict[str, tuple[type, object]]:
    fields: dict[str, tuple[type, object]] = {}
    for name in TITLE_TOKEN_SKETCH_FIELDS:
        fields[name] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                le=1.0,
                description="Keyed token-sketch bucket frequency for the window title.",
            ),
        )
    for name in TITLE_CHAR3_SKETCH_FIELDS:
        fields[name] = (
            float,
            Field(
                default=0.0,
                ge=0.0,
                le=1.0,
                description="Keyed char-3 sketch bucket frequency for the window title.",
            ),
        )
    fields["title_char_count"] = (
        int,
        Field(default=0, ge=0, description="Normalized window-title character count."),
    )
    fields["title_token_count"] = (
        int,
        Field(default=0, ge=0, description="Normalized window-title token count."),
    )
    fields["title_unique_token_ratio"] = (
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Unique-token ratio after title token normalization.",
        ),
    )
    fields["title_digit_ratio"] = (
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Digit-character ratio in the normalized title.",
        ),
    )
    fields["title_separator_count"] = (
        int,
        Field(
            default=0,
            ge=0,
            description="Count of common browser-title separator characters.",
        ),
    )
    return fields


FeatureRow = create_model(
    "FeatureRow",
    __base__=FeatureRowBase,
    **cast(dict[str, Any], _feature_row_v3_field_definitions()),
)


class LabelSpan(BaseModel, frozen=True):
    """A contiguous time span carrying a single task-type label.

    Gold labels and weak labels share this structure; ``provenance``
    distinguishes them (e.g. ``"manual"`` vs ``"weak:app_rule"``).

    Optional ``user_id`` ties the span to a specific user (required for
    multi-user datasets and block-to-window projection).  Optional
    ``confidence`` records the labeler's self-assessed certainty.
    Both default to ``None`` for backward compatibility with existing
    label CSV imports.
    """

    start_ts: datetime = Field(description="Span start (aware UTC, inclusive).")
    end_ts: datetime = Field(description="Span end (aware UTC, exclusive).")
    label: str = Field(description="Task-type label from LABEL_SET_V1.")
    provenance: str = Field(description="Origin tag, e.g. 'manual' or 'weak:app_rule'.")
    user_id: str | None = Field(
        default=None, description="User who created this label."
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Labeler confidence (0-1)."
    )

    @field_validator("start_ts", "end_ts", mode="before")
    @classmethod
    def _normalize_timestamps(cls, v: object) -> object:
        """Normalize label span timestamps to aware UTC."""
        if isinstance(v, datetime):
            return ts_utc_aware_get(v)
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _nan_confidence_to_none(cls, v: object) -> object:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    extend_forward: bool = Field(
        default=False,
        description="When true, this label extends forward until the next label is created.",
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> LabelSpan:
        if self.extend_forward:
            if self.end_ts < self.start_ts:
                raise ValueError(
                    f"end_ts ({self.end_ts}) must not be before "
                    f"start_ts ({self.start_ts})"
                )
        elif self.end_ts <= self.start_ts:
            raise ValueError(
                f"end_ts ({self.end_ts}) must be strictly after "
                f"start_ts ({self.start_ts})"
            )
        if self.label not in LABEL_SET_V1:
            raise ValueError(
                f"Unknown label {self.label!r}; must be one of {sorted(LABEL_SET_V1)}"
            )
        return self
