# Labels v2 Technical Specification

Version: 2.0 (Final)
Status: Production
Last Updated: 2026-04-23

This document defines the technical architecture, schema, and inference pipeline for the `labels_v2` ontology. It is intended for engineers implementing the classification system, data pipelines, and domain plugins.

For human annotation guidelines, refer to the [Annotation Playbook](labels_v2_playbook.md).

---

## 1. System Architecture

The `labels_v2` system moves from a flat label set to a multi-resolution, multi-axis classification model:

1. **Evidence Layer (15s–60s):** Raw observational signals (`EvidenceSnapshot`).
2. **Inference Layer (2m–5m):** Semantic labeling (`CrossDomainLabel`) with explicit uncertainty.
3. **Aggregation Layer (15m–90m):** Session rollups using deterministic dominance rules.
4. **Extension Layer:** Namespaced domain plugins for persona-specific activities.

---

## 2. Core Ontology (The Interpretation Layer)

The semantic output is structured as a `CrossDomainLabel` containing multiple independent axes.

### 2.1 Required Core Axis: Mode

The fundamental classification of the activity. This axis is universally applicable and strictly bounded.

```typescript
type Mode =
  | "Produce"    // Materially advancing an artifact
  | "Consume"    // Understanding, inspecting, gathering information
  | "Coordinate" // Managing tasks, workflow orchestration
  | "Attend"     // Live synchronous participation
  | "Idle";      // Restorative, non-task-directed
```

### 2.2 Optional Cross-Domain Axes

These axes provide additional context but do not redefine the `Mode`.

```typescript
type Subtype =
  | "Build" | "Debug" | "Write" | "Review" | "Analyze" | "Plan" | "Admin"
  | "ReadResearch" | "Learn" | "ExploreReference" | "Monitor"
  | "Communicate" | "Meet" | "BreakIdle";

type InteractionStyle =
  | "Active"  // Direct manipulation, typing
  | "Passive" // Watching, listening
  | "Mixed"   // Alternation
  | "Idle";   // Negligible engagement

type CollaborationMode =
  | "Solo"        // No interaction
  | "AsyncCollab" // Comments, tickets, async chat
  | "SyncCollab"  // Live coordinated presence
  | "Unknown";

type OutputDomain =
  | "Code" | "Writing" | "Research" | "Admin"
  | "Communication" | "Design" | "Analysis" | "Operations" | "Unknown";
```

### 2.3 Explicit Uncertainty Model

The system must output the label alongside its uncertainty structure. A single label is insufficient.

```typescript
type AxisDecision<T extends string> = {
  value: T;
  confidence: number; // 0.0 to 1.0
  alternatives?: T[];
  reason_codes?: string[];
};

type SupportState =
  | "Supported"    // Evidence coherently supports the label
  | "WeakEvidence" // Evidence is sparse or indirect
  | "Rejected"     // Candidate considered and rejected
  | "MixedUnknown";// Multiple plausible interpretations remain unresolved
```

---

## 3. The Evidence Layer (Factual Substrate)

Semantic output must be explicitly separated from observed evidence. The `EvidenceSnapshot` represents the raw factual signals collected over a short window (e.g., 30 seconds).

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

---

## 4. Inference Pipeline & Decision Protocol

The inference engine consumes `EvidenceSnapshot` sequences and deterministically outputs a `CrossDomainLabel`.

### 4.1 Mode Precedence Rules

The engine MUST evaluate `Mode` in the following strict order:

1. **Idle:** If `low_interaction_idle_signal` is true across all snapshots.
2. **Attend:** If `meeting_signal`, `active_call`, or `active_mic` is true.
3. **Coordinate:** If communication apps (Slack, Teams, Mail) dominate.
4. **Produce:** If editor/terminal apps dominate AND `key_events` > threshold.
5. **Consume:** If browser dominates AND `scroll_events` > 0 AND `key_events` < threshold.

### 4.2 Subtype Restriction Policy

The `Subtype` must be validated against the resolved `Mode`. Invalid combinations must be stripped or escalated.

* **Produce:** Build, Debug, Write, Review, Analyze, Plan, Admin
* **Consume:** ReadResearch, Learn, ExploreReference, Review, Monitor, Analyze
* **Coordinate:** Communicate, Plan, Admin, Review
* **Attend:** Meet, Learn, Communicate
* **Idle:** BreakIdle

### 4.3 Escalation Triggers

If the inference engine cannot confidently resolve the `Mode` (e.g., `confidence < 0.5`), it MUST escalate the `SupportState` to `MixedUnknown`. If `0.5 <= confidence < 0.7`, it escalates to `WeakEvidence`.

---

## 5. Domain Plugin Architecture

To support diverse personas without bloating the core ontology, domain-specific activities are handled via namespaced plugins.

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

Session summaries (15m–90m) are built deterministically from inference windows (2m–5m).

**The 60% Dominance Rule:**
* Calculate the total duration of each `Mode` within the session.
* If a single `Mode` occupies $\ge 60\%$ of the total time, the session is assigned that `Mode`.
* Otherwise, the session `Mode` resolves to `"Mixed"`.

---

## 7. Data Storage & Versioning

Every persisted label record MUST be wrapped in a `LabelEnvelope` to track schema and rule versions.

```typescript
type LabelEnvelope = {
  taxonomy_version: "labels_v2";
  rule_version: string;
  generated_at: string;
  evidence_window_ms: number;   // e.g., 30000 (30s)
  inference_window_ms: number;  // e.g., 180000 (3m)
  label: CrossDomainLabel;
  plugins?: PluginPayload[];
};
```

### 7.1 Migration from v1

When migrating legacy `LABEL_SET_V1` data:
1. Map the legacy string to the closest `Mode` and `Subtype` tuple (e.g., `Build` → `Produce` + `Build`).
2. Set `confidence = 0.3` and `support_state = MixedUnknown` for legacy `Mixed/Unknown` records to prevent false precision.
3. Omit optional axes (`interaction_style`, `collaboration_mode`, `output_domain`).
4. Stamp the envelope with `reason_codes: ["v1_legacy_migration"]`.
