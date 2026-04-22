# Labels v2 Design

Here is a concrete labels_v2 design that makes ambiguity bounded, extensions scalable, and personas easier to support without exploding the global label set.

## Goals

`labels_v2` should guarantee:
* a small stable core
* explicit decision rules
* explicit uncertainty
* multi-resolution labeling
* domain extensions without core churn
* compatibility with future v3+

The key change is this:
move from “one flat label set” to “core classification + evidence + extensions + aggregation rules”.

---

## 1. Core ontology

### 1.1 Required core axis

```typescript
type Mode =
  | "Produce"
  | "Consume"
  | "Coordinate"
  | "Attend"
  | "Idle";
```

This remains the only required semantic axis.
It should stay frozen unless there is overwhelming evidence that a new universal mode is needed.

### 1.2 Optional cross-domain axes

```typescript
type Subtype =
  | "Build"
  | "Debug"
  | "Write"
  | "Review"
  | "ReadResearch"
  | "Communicate"
  | "Meet"
  | "Admin"
  | "Plan"
  | "BreakIdle"
  | "Analyze"
  | "Learn"
  | "ExploreReference"
  | "Monitor";

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
  | "Code"
  | "Writing"
  | "Research"
  | "Admin"
  | "Communication"
  | "Design"
  | "Analysis"
  | "Operations"
  | "Unknown";

type SupportState =
  | "Supported"
  | "WeakEvidence"
  | "Rejected"
  | "MixedUnknown";
```

**Why these additions**

Compared with the previous set, added:
* Analyze
* Learn
* ExploreReference
* Monitor
* Design
* Analysis
* Operations

These cover the most obvious persona gaps without breaking the shape of the system.

---

## 2. Decision protocol for mode

This is the most important part. You need deterministic precedence rules, not just label descriptions.

### 2.1 Mode decision order

For a given window, assign mode using this order:

**Rule 1 — Idle**
Assign Idle when the activity is primarily:
* restorative,
* non-task-directed,
* disengaged from a recognized obligation or deliverable,
* or no meaningful evidence of directed work is present.

Examples:
* aimless scrolling between work blocks
* staring / absent
* very low interaction with no task continuity

Do not assign Idle just because input is low.

**Rule 2 — Attend**
Assign Attend when the dominant frame is participation in a live synchronous session.

Examples:
* Zoom/Meet/Teams call
* live lecture
* standup
* interview
* pair session over voice/video

Even if the user speaks a lot or takes notes, Attend stays dominant if the live session is the main frame.

**Rule 3 — Coordinate**
Assign Coordinate when the primary purpose is:
* managing people/tasks,
* sending/responding to messages,
* scheduling,
* triaging,
* delegating,
* handoffs,
* workflow orchestration.

Examples:
* Slack/email triage
* Jira/Linear task organization
* rescheduling meetings
* following up on blockers

**Rule 4 — Produce**
Assign Produce when the primary purpose is:
* creating,
* modifying,
* repairing,
* synthesizing,
* or materially advancing an artifact/system/deliverable.

Examples:
* coding
* writing a report
* editing a design
* preparing slides
* spreadsheet transformation
* note synthesis into a document

Debugging falls here if the session is primarily in service of changing/fixing the artifact/system.

**Rule 5 — Consume**
Assign Consume when the primary purpose is:
* reading,
* watching,
* listening,
* inspecting,
* understanding,
* gathering information,
  without material artifact advancement being dominant.

Examples:
* reading docs
* watching recorded course material
* browsing references
* reading papers
* checking dashboards

### 2.2 Tie-break rules

**Produce vs Consume**
If both occur:
* choose Produce when the session materially advances an artifact
* choose Consume when understanding/intake is the main purpose

**Coordinate vs Attend**
* Attend for live synchronous participation
* Coordinate for async or workflow-management activity

**Consume vs Idle**
* Consume if the intake is goal-directed
* Idle if it is restorative / detached / non-directed

**Produce vs Coordinate**
* Produce if communications are in service of creating/editing a main artifact
* Coordinate if communications themselves are the primary work output

---

## 3. Time model

A lot of ambiguity comes from coarse windows. `labels_v2` should formally separate three levels:

### 3.1 Evidence window
Short slices where raw signals are observed.
Suggested: 15s to 60s

### 3.2 Inference window
The unit at which semantic labels are assigned.
Suggested: 2 to 5 minutes
This is where mode, subtype, etc. are produced.

### 3.3 Session aggregate
A longer contiguous block for reporting.
Suggested: 15 to 90 minutes
Built from inference windows. This prevents one giant mixed block from forcing one fake label.

---

## 4. Explicit uncertainty model

Do not output only one label. Output the label plus uncertainty structure.

### 4.1 Core uncertainty fields

```typescript
type AxisDecision<T extends string> = {
  value: T;
  confidence: number; // 0..1
  alternatives?: T[];
  reason_codes?: string[];
};

type ModeDecision = AxisDecision<Mode>;
```

### 4.2 Support state semantics

* **Supported**: evidence coherently supports the assigned label
* **WeakEvidence**: evidence is sparse or indirect
* **Rejected**: candidate label was considered and rejected by stronger evidence
* **MixedUnknown**: multiple plausible interpretations remain unresolved

`SupportState` should not be a vague confidence synonym. It should mean something about evidence structure.

---

## 5. Split “semantic output” from “observed evidence”

This is necessary for scalability and future relabeling.

### 5.1 Evidence layer

Example shape:

```typescript
type EvidenceSnapshot = {
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

This is not user-facing ontology. It is the factual substrate.

### 5.2 Interpretation layer

```typescript
type CrossDomainLabel = {
  mode: AxisDecision<Mode>;
  subtype?: AxisDecision<Subtype>;
  interaction_style?: AxisDecision<InteractionStyle>;
  collaboration_mode?: AxisDecision<CollaborationMode>;
  output_domain?: AxisDecision<OutputDomain>;
  support_state: SupportState;
};
```

Keep them separate.

---

## 6. Subtype policy

The subtype list must not become a dumping ground.

**Rule:**
Subtype sharpens the shape of the activity within the chosen mode, but does not redefine the mode.

Examples:
* mode=Produce, subtype=Build
* mode=Produce, subtype=Debug
* mode=Consume, subtype=Learn
* mode=Consume, subtype=ExploreReference
* mode=Coordinate, subtype=Communicate
* mode=Attend, subtype=Meet
* mode=Idle, subtype=BreakIdle

### 6.1 Recommended allowed subtype families by mode

Not a hard type-level restriction, but a policy table.

**Produce**
* Build, Debug, Write, Review, Analyze, Plan, Admin

**Consume**
* ReadResearch, Learn, ExploreReference, Review, Monitor, Analyze

**Coordinate**
* Communicate, Plan, Admin, Review

**Attend**
* Meet, Learn, Communicate

**Idle**
* BreakIdle

This reduces nonsensical combinations.

---

## 7. Output domain semantics

Define this narrowly:
`output_domain` is the primary artifact or work-product domain being advanced, not the app or medium being used.

Examples:
* coding in a browser editor → Code
* writing a spec in Google Docs → Writing
* literature review notes → Research
* budget workbook cleanup → Analysis
* task triage in Linear → Admin
* customer support replies → Communication
* Figma iteration → Design
* infra dashboard / incident ops → Operations

This prevents the axis from collapsing into “which app is open.”

---

## 8. Interaction style semantics

This axis is observational, not deeply semantic.

Use:
* **Active**: frequent direct manipulation or generation
* **Passive**: mostly intake, playback, or watch state
* **Mixed**: meaningful alternation between active and passive
* **Idle**: negligible engagement

This axis should not override mode.

For example:
* Consume + Active is valid for close reading and note-taking
* Attend + Passive is valid for a lecture
* Produce + Mixed is valid for writing with intermittent rereading

---

## 9. Collaboration mode semantics

Define by interaction topology, not employer/team boundaries.

* **Solo**: no meaningful collaborator interaction during the window
* **AsyncCollab**: interaction via comments/messages/tickets/docs without live co-presence
* **SyncCollab**: live coordinated presence with another person/group
* **Unknown**: collaboration status cannot be determined

This lets external customer chat, live tutoring, or interviews all fit.

---

## 10. Domain plugin architecture

This is what makes the taxonomy scalable across personas.
Do not keep expanding the global core forever.
Add optional namespaced domain layers.

### 10.1 Base plugin container

```typescript
type PluginPayload =
  | { namespace: "software"; data: SoftwareLabels }
  | { namespace: "research"; data: ResearchLabels }
  | { namespace: "design"; data: DesignLabels }
  | { namespace: "education"; data: EducationLabels }
  | { namespace: "operations"; data: OperationsLabels }
  | { namespace: "analysis"; data: AnalysisLabels };
```

### 10.2 Example domain schemas

**Software**
```typescript
type SoftwareLabels = {
  activity?:
    | "Implement"
    | "DebugIncident"
    | "Refactor"
    | "ReviewCode"
    | "RunTests"
    | "InspectLogs";
  artifact_scope?:
    | "LocalFile"
    | "MultiFile"
    | "Service"
    | "Unknown";
};
```

**Research**
```typescript
type ResearchLabels = {
  activity?:
    | "LiteratureReview"
    | "NoteSynthesis"
    | "CitationHunt"
    | "ExperimentReview"
    | "SourceScreening";
};
```

**Design**
```typescript
type DesignLabels = {
  activity?:
    | "InspirationGathering"
    | "Sketch"
    | "IterateVisual"
    | "ReviewVisual"
    | "Prototype";
};
```

**Education**
```typescript
type EducationLabels = {
  activity?:
    | "LectureWatching"
    | "StudyReading"
    | "PracticeExercise"
    | "Revision"
    | "AssessmentWork";
};
```

**Operations**
```typescript
type OperationsLabels = {
  activity?:
    | "IncidentMonitoring"
    | "Triage"
    | "RunbookExecution"
    | "SystemCheck"
    | "CapacityReview";
};
```

**Analysis**
```typescript
type AnalysisLabels = {
  activity?:
    | "SpreadsheetModeling"
    | "DataCleaning"
    | "DashboardInspection"
    | "Reconciliation"
    | "QuantReview";
};
```

These plugins are optional and evolve independently.

---

## 11. Unknown / unresolved handling

You need controlled escape hatches.

### 11.1 Use these values intentionally
* **Unknown**: cannot infer from current evidence
* **MixedUnknown**: multiple plausible states remain unresolved
* **(no subtype at all)**: subtype omitted because not useful or not inferable

### 11.2 Escalation rule
If Unknown or MixedUnknown exceeds a threshold in production, investigate one of:
* insufficient raw signals
* bad decision rules
* missing plugin coverage
* missing subtype
* too-large window size

This makes fallback values diagnostic, not lazy.

---

## 12. Aggregation contract

You need deterministic rules for building session summaries from inference windows.

### 12.1 Suggested aggregation rule
For a session made of multiple inference windows:
* choose the dominant mode if it occupies at least 60%
* otherwise session mode becomes Mixed
* optional axes aggregate only if they align with the dominant mode strongly enough
* if not, omit or mark as mixed

You may want a session-only label type:
```typescript
type SessionMode = Mode | "Mixed";
```

Do not force all aggregates into the same exact ontology as short windows.
That creates false precision.

---

## 13. Annotation playbook

To reduce human ambiguity, create a playbook with:

### 13.1 Canonical examples
For every mode/subtype pair, include:
* positive examples
* near misses
* counterexamples

### 13.2 Boundary examples
Especially:
* debugging
* note-taking while reading
* meetings where the user presents
* social media for work vs for break
* customer support live chat
* spreadsheet work
* design browsing

### 13.3 Dominance checklist
Annotator checklist:
1. Is this task-directed?
2. Is it live synchronous?
3. Is it mainly workflow/people coordination?
4. Is an artifact materially advanced?
5. Otherwise is it intake/understanding?

That alone will improve consistency a lot.

---

## 14. Versioning and migration

Every label record should include:

```typescript
type LabelEnvelope = {
  taxonomy_version: "labels_v2";
  rule_version: string;
  generated_at: string;
  evidence_window_ms: number;
  inference_window_ms: number;
  label: CrossDomainLabel;
  plugins?: PluginPayload[];
};
```

**Migration rules**
When you later move to v3:
* preserve mode whenever possible
* only split subtypes forward, never silently reinterpret old ones
* keep a migration table

Example:
* ReadResearch may later split into LiteratureReview and ReferenceReading
* v2 → v3 migration can map both to ReferenceReading, or preserve legacy if ambiguous

---

## 15. Concrete examples

### Example A — software engineer fixing a bug while on a pair call
```json
{
  "taxonomy_version": "labels_v2",
  "rule_version": "2026-04-13.1",
  "evidence_window_ms": 30000,
  "inference_window_ms": 300000,
  "label": {
    "mode": {
      "value": "Attend",
      "confidence": 0.74,
      "alternatives": ["Produce"],
      "reason_codes": ["active_call", "screen_share_signal", "editor_active"]
    },
    "subtype": {
      "value": "Meet",
      "confidence": 0.68,
      "alternatives": ["Debug"]
    },
    "interaction_style": {
      "value": "Mixed",
      "confidence": 0.86
    },
    "collaboration_mode": {
      "value": "SyncCollab",
      "confidence": 0.97
    },
    "output_domain": {
      "value": "Code",
      "confidence": 0.89
    },
    "support_state": "MixedUnknown"
  },
  "plugins": [
    {
      "namespace": "software",
      "data": {
        "activity": "DebugIncident",
        "artifact_scope": "MultiFile"
      }
    }
  ]
}
```
This shows why plugins and alternatives matter. The core can admit ambiguity without collapsing.

### Example B — student watching a recorded lecture and taking sparse notes
```json
{
  "taxonomy_version": "labels_v2",
  "rule_version": "2026-04-13.1",
  "evidence_window_ms": 30000,
  "inference_window_ms": 300000,
  "label": {
    "mode": {
      "value": "Consume",
      "confidence": 0.91,
      "reason_codes": ["video_playback", "goal_directed_context"]
    },
    "subtype": {
      "value": "Learn",
      "confidence": 0.88
    },
    "interaction_style": {
      "value": "Passive",
      "confidence": 0.72,
      "alternatives": ["Mixed"]
    },
    "collaboration_mode": {
      "value": "Solo",
      "confidence": 0.95
    },
    "output_domain": {
      "value": "Research",
      "confidence": 0.54,
      "alternatives": ["Unknown"]
    },
    "support_state": "Supported"
  },
  "plugins": [
    {
      "namespace": "education",
      "data": {
        "activity": "LectureWatching"
      }
    }
  ]
}
```

### Example C — analyst reconciling a spreadsheet
```json
{
  "taxonomy_version": "labels_v2",
  "rule_version": "2026-04-13.1",
  "evidence_window_ms": 30000,
  "inference_window_ms": 300000,
  "label": {
    "mode": {
      "value": "Produce",
      "confidence": 0.79,
      "alternatives": ["Consume"]
    },
    "subtype": {
      "value": "Analyze",
      "confidence": 0.83
    },
    "interaction_style": {
      "value": "Active",
      "confidence": 0.77
    },
    "collaboration_mode": {
      "value": "Solo",
      "confidence": 0.81
    },
    "output_domain": {
      "value": "Analysis",
      "confidence": 0.92
    },
    "support_state": "Supported"
  },
  "plugins": [
    {
      "namespace": "analysis",
      "data": {
        "activity": "Reconciliation"
      }
    }
  ]
}
```

### Example D — doomscrolling during a break
```json
{
  "taxonomy_version": "labels_v2",
  "rule_version": "2026-04-13.1",
  "evidence_window_ms": 30000,
  "inference_window_ms": 300000,
  "label": {
    "mode": {
      "value": "Idle",
      "confidence": 0.76,
      "alternatives": ["Consume"],
      "reason_codes": ["non_task_context", "social_feed_pattern", "no_artifact_progress"]
    },
    "subtype": {
      "value": "BreakIdle",
      "confidence": 0.89
    },
    "interaction_style": {
      "value": "Passive",
      "confidence": 0.85
    },
    "collaboration_mode": {
      "value": "Unknown",
      "confidence": 0.91
    },
    "output_domain": {
      "value": "Unknown",
      "confidence": 0.98
    },
    "support_state": "Supported"
  }
}
```

---

## 16. What this buys you

This `labels_v2` structure makes the system scalable because:
* the core stays small
* persona differences go into plugin namespaces
* ambiguity is recorded, not denied
* old data can be reinterpreted from evidence
* annotation disagreements become measurable
* you can add domains without rewriting the world

---

## 17. Recommended migration from current scheme

**Keep unchanged**
* mode
* interaction_style
* collaboration_mode
* support_state

**Expand**
* subtype
* output_domain

**Add**
* decision protocol
* uncertainty fields
* reason codes
* plugin namespaces
* aggregation model
* taxonomy version / rule version

---

## 18. Minimal MVP if you want to avoid overbuilding

If you want the smallest strong upgrade, do only this first:
1. freeze mode
2. add deterministic precedence rules
3. add confidence, alternatives, reason_codes
4. add Analyze, Learn, ExploreReference, Monitor
5. define output_domain precisely
6. separate evidence windows from inference windows
7. reserve plugins field even if mostly empty initially

That already gets you most of the benefit.

---

**Bottom line**

The scalable answer is not “invent enough labels that nobody is ever unsure.”

The scalable answer is:
* small universal core
* deterministic assignment rules
* explicit uncertainty
* short-window inference
* domain plugin extensions
* versioned schema
* evidence preserved separately from interpretation

That is the structure that can handle edge cases without collapsing into taxonomy sprawl.
