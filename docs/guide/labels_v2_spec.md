# Labels v2 Technical Specification

Version: 2.1 (Final)
Status: Production
Last Updated: 2026-04-24

This document defines the technical architecture, schema, and inference pipeline for the `labels_v2` ontology. It is intended for engineers implementing the classification system, data pipelines, and domain plugins.

For human annotation guidelines, refer to the [Annotation Playbook](labels_v2_playbook.md).

---

## 1. System Architecture

The `labels_v2` system moves from a flat label set to a layered classification model:

1. **Evidence Layer (15s-60s):** Raw observational signals (`EvidenceSnapshot`).
2. **Observed Layer (15s-60s):** Deterministic normalized facts (`ObservedLabel`) derived from telemetry.
3. **Inference Layer (2m-5m):** Semantic labeling (`SemanticLabel`) with explicit uncertainty and provenance.
4. **Aggregation Layer (15m-90m):** Session rollups using deterministic dominance rules.
5. **Extension Layer:** Namespaced domain plugins for persona-specific activities.

Core rule:
deterministic observed facts are the debugging ground truth.
Semantic labels are deterministic given evidence plus rule version, but they are still interpretation and must remain reversible.

---

## 2. Core Contract

### 2.1 Deterministic Observed Axes

The observed layer captures facts that should be stable under relabeling.

```typescript
type ActivitySurface =
  | "Edit"
  | "Read"
  | "Watch"
  | "Message"
  | "Call"
  | "Search"
  | "IdleLike";

type ArtifactTouch =
  | "None"
  | "ReadOnly"
  | "Modified"
  | "Created";

type SyncPresence =
  | "None"
  | "LiveHumanSession"
  | "LiveStream";

type CollaborationSurface =
  | "None"
  | "AsyncText"
  | "SyncVoiceVideo";

type ObservedLabel = {
  activity_surface: ActivitySurface;
  artifact_touch: ArtifactTouch;
  sync_presence: SyncPresence;
  collaboration_surface: CollaborationSurface;
};
```

Additional deterministic facts such as input activity bands or app categories MAY be persisted alongside this layer, but they must remain observational rather than semantic.

### 2.2 Semantic Axes

The semantic layer is structured as a `SemanticLabel` containing multiple independent axes.

```typescript
type Mode =
  | "Produce"    // Materially advancing an artifact
  | "Consume"    // Understanding, inspecting, gathering information
  | "Coordinate" // Managing tasks, workflow orchestration
  | "Attend"     // Live synchronous participation
  | "Idle";      // Restorative, non-task-directed

type Subtype =
  | "Build" | "Debug" | "Write" | "Review" | "Analyze" | "Plan" | "Admin"
  | "ReadResearch" | "Learn" | "ExploreReference" | "Monitor"
  | "Communicate" | "Meet" | "BreakIdle";

type InteractionStyle =
  | "Active"
  | "Passive"
  | "Mixed"
  | "Idle";

type CollaborationMode =
  | "Solo"
  | "AsyncCollab"
  | "SyncCollab"
  | "Unknown";

type OutputDomain =
  | "Code" | "Writing" | "Research" | "Admin"
  | "Communication" | "Design" | "Analysis" | "Operations" | "Unknown";
```

### 2.3 Uncertainty and Provenance

The system must output semantic labels alongside their uncertainty structure and intent provenance. A single label is insufficient.

```typescript
type AxisDecision<T extends string> = {
  value: T;
  confidence: number; // 0.0 to 1.0
  alternatives?: T[];
  reason_codes?: string[];
};

type SupportState =
  | "Supported"
  | "WeakEvidence"
  | "Rejected"
  | "MixedUnknown";

type IntentBasis =
  | "ObservedOnly"
  | "InferredFromContext"
  | "UserDeclared"
  | "Unknown";

type ModeSource =
  | "DeterministicRule"
  | "ProbabilisticModel"
  | "UserOverride";

type UserOverride = {
  active: boolean;
  override_mode?: Mode;
  override_subtype?: Subtype;
  note?: string;
};
```

### 2.4 Semantic Label

```typescript
type SemanticLabel = {
  mode: AxisDecision<Mode>;
  subtype?: AxisDecision<Subtype>;
  interaction_style?: AxisDecision<InteractionStyle>;
  collaboration_mode?: AxisDecision<CollaborationMode>;
  output_domain?: AxisDecision<OutputDomain>;
  support_state: SupportState;
  intent_basis: IntentBasis;
  mode_source: ModeSource;
  user_override?: UserOverride;
};
```

---

## 3. Evidence Layer and Observed Projection

Semantic output must remain explicitly separated from observed evidence. The `EvidenceSnapshot` represents the raw factual signals collected over a short window (for example, 30 seconds).

```typescript
type EvidenceSnapshot = {
  bucket_start_ts: string;
  bucket_end_ts: string;

  app_ids: string[];
  window_title_hashes: string[];
  foreground_duration_ms: number;

  key_events: number;
  pointer_events: number;
  scroll_events: number;
  app_switch_count: number;

  active_call: boolean;
  active_mic: boolean;
  active_camera: boolean;

  browser_url_category?: string;
  file_types?: string[];
  meeting_signal?: boolean;
  build_or_run_signal?: boolean;
  test_signal?: boolean;
  low_interaction_idle_signal?: boolean;
};
```

The projection from `EvidenceSnapshot` to `ObservedLabel` MUST be deterministic.

Examples:
* active foreground call or meeting -> `sync_presence = LiveHumanSession`
* file changed -> `artifact_touch = Modified`
* outgoing Slack/email message -> `activity_surface = Message`
* browser reading with no edits -> `artifact_touch = ReadOnly`
* no input plus disengaged or locked state -> `activity_surface = IdleLike`

This layer is safe to persist and debug because it does not claim intent.

---

## 4. Inference Pipeline & Decision Protocol

The inference engine consumes `ObservedLabel` plus surrounding `EvidenceSnapshot` context and deterministically outputs a `SemanticLabel`.

### 4.1 Mode Precedence Rules

The engine MUST evaluate `Mode` in the following strict order:

1. **Idle:** If the observed layer is dominantly `IdleLike`, there is no artifact advancement, no meaningful sync presence, and the context indicates disengaged or restorative activity.
2. **Attend:** If `sync_presence = LiveHumanSession` dominates the window.
3. **Coordinate:** If `activity_surface = Message` or `collaboration_surface = AsyncText` dominates the window.
4. **Produce:** If `activity_surface = Edit` and `artifact_touch` is `Modified` or `Created` for the dominant slice.
5. **Consume:** If `Read`, `Watch`, or `Search` dominates without material artifact advancement.

### 4.2 Subtype Restriction Policy

The `Subtype` must be validated against the resolved `Mode`. Invalid combinations must be stripped or escalated.

* **Produce:** Build, Debug, Write, Review, Analyze, Plan, Admin
* **Consume:** ReadResearch, Learn, ExploreReference, Review, Monitor, Analyze
* **Coordinate:** Communicate, Plan, Admin, Review
* **Attend:** Meet, Learn, Communicate
* **Idle:** BreakIdle

### 4.3 Escalation Triggers and Intent Sensitivity

If the inference engine cannot confidently resolve the `Mode` (for example, `confidence < 0.5`), it MUST escalate the `SupportState` to `MixedUnknown`.
If `0.5 <= confidence < 0.7`, or if the label depends on limited contextual interpretation, it MUST escalate to `WeakEvidence`.

Intent-sensitive distinctions such as `Consume` vs `Idle`, `Produce` vs `Consume` during debugging, and `Coordinate` vs `Produce` during collaboration MUST NOT be treated as deterministic ground truth when the observed layer alone does not settle them.

Use provenance fields as follows:
* `intent_basis = "ObservedOnly"` when the semantic decision follows directly from observed invariants
* `intent_basis = "InferredFromContext"` when surrounding context resolves a plausible tie
* `intent_basis = "UserDeclared"` when the user supplied or corrected the label
* `intent_basis = "Unknown"` when the basis cannot be recovered

Set `mode_source` to:
* `DeterministicRule` for rule-based semantic assignment
* `ProbabilisticModel` for model-backed semantic assignment
* `UserOverride` when the user correction is the authoritative semantic result

---

## 5. Domain Plugin Architecture

To support diverse personas without bloating the core ontology, domain-specific activities are handled via namespaced plugins attached to the semantic envelope.

```typescript
type PluginPayload =
  | { namespace: "software"; data: SoftwareLabels }
  | { namespace: "research"; data: ResearchLabels }
  | { namespace: "design"; data: DesignLabels }
  | { namespace: "education"; data: EducationLabels }
  | { namespace: "operations"; data: OperationsLabels }
  | { namespace: "analysis"; data: AnalysisLabels };
```

**Example: Software Plugin**
```typescript
type SoftwareLabels = {
  activity?: "Implement" | "DebugIncident" | "Refactor" | "ReviewCode" | "RunTests" | "InspectLogs";
  artifact_scope?: "LocalFile" | "MultiFile" | "Service" | "Unknown";
};
```

---

## 6. Session Aggregation Contract

Session summaries (15m-90m) are built deterministically from inference windows (2m-5m).
Observed rollups MAY be aggregated in parallel for debugging and future relabeling, but session semantics must never overwrite the per-window observed facts.

**The 60% Dominance Rule:**
* Calculate the total duration of each `Mode` within the session.
* If a single `Mode` occupies `>= 60%` of the total time, the session is assigned that `Mode`.
* Otherwise, the session `Mode` resolves to `"Mixed"`.
* Optional axes should only aggregate if they align strongly with the dominant session mode; otherwise omit them or mark them mixed.

---

## 7. Data Storage & Versioning

Every persisted label record MUST be wrapped in a `LabelEnvelope` to track schema and rule versions.
Native `labels_v2` records SHOULD persist both the deterministic observed layer and the semantic layer; legacy migrations MAY omit `observed` until telemetry backfill exists.

```typescript
type LabelEnvelope = {
  taxonomy_version: "labels_v2";
  rule_version: string;
  generated_at: string;
  evidence_window_ms: number;   // e.g., 30000 (30s)
  inference_window_ms: number;  // e.g., 180000 (3m)
  observed?: ObservedLabel;
  semantic: SemanticLabel;
  plugins?: PluginPayload[];
};
```

### 7.1 Migration from v1

When migrating legacy `LABEL_SET_V1` data:
1. Map the legacy string to the closest `Mode` and `Subtype` tuple (for example, `Build` -> `Produce` + `Build`).
2. Set conservative semantic uncertainty for migrated records. For legacy `Mixed/Unknown`, use `confidence = 0.3` and `support_state = MixedUnknown` to prevent false precision.
3. Set `intent_basis = "Unknown"` unless the migration source is an explicit user-supplied label, in which case use `intent_basis = "UserDeclared"`.
4. Set `mode_source = "UserOverride"` for imported user labels; otherwise use the migration rule path that produced the semantic result.
5. Backfill `observed` from telemetry when available. If telemetry is unavailable, leave `observed` absent and treat the record as semantic-only legacy data rather than a fully native v2 record.
6. Omit optional axes (`interaction_style`, `collaboration_mode`, `output_domain`) unless the migration source genuinely supports them.
7. Stamp migrated semantic decisions with `reason_codes: ["v1_legacy_migration"]`.
