# Labels v2 Annotation Playbook

Version: 2.1 (Final)
Status: Production
Last Updated: 2026-04-24

This document is the authoritative contract for assigning labels_v2 to inference windows.

It optimizes for:

* Consistency over intuition
* Determinism over interpretation
* Bounded ambiguity over false precision
* Extensibility without core churn

---

## 0. Core Principles (Non-Negotiable)

1. Start from observed facts before interpreting intent.
2. Always assign mode using the precedence rules.
3. Do not mix axes.
    * mode ≠ subtype ≠ output_domain
4. Label the current window, not the whole session.
5. Use uncertainty explicitly. Do not guess.
6. When in doubt, follow tie-break rules and `intent_basis`, not intuition.
7. If a user correction exists, record it as an override rather than pretending the original semantic guess was certain.

---

## 1. Axis Semantics (Strict Separation)

Think in two layers:

* **Observed layer**: what was directly seen?
    * activity surface
    * artifact touch
    * sync presence
    * collaboration surface
* **Semantic layer**: how should the window be interpreted?
    * mode
    * subtype
    * output_domain
    * interaction_style
    * collaboration_mode
    * support_state
    * intent_basis

Observed facts are the stable anchor.
Semantic labels are still deterministic given the same evidence and rule version, but they may be inferred rather than directly known.

Each axis answers a different question:

| Axis | Question |
|---|---|
| mode | What kind of activity is happening? |
| subtype | What shape does the activity take within the mode? |
| output_domain | What artifact/domain is being advanced? |
| interaction_style | How is the user behaviorally engaging? |
| collaboration_mode | Who is involved and how? |
| support_state | How strong is the evidence? |

**Critical rule**

The same output domain can appear across all modes.

Example (Code):

* Reading code → Consume + Code
* Writing code → Produce + Code
* Discussing code in Slack → Coordinate + Code
* Pair programming → Attend + Code

---

## 2. Mode Assignment Protocol (Deterministic)

You MUST follow this exact order.
Apply it to the current window using observed facts first, then use context only when the observed layer does not fully settle the semantic interpretation.

### Step 1 — Idle

Assign Idle if ALL are true:

* observed facts show no meaningful artifact advancement
* there is no meaningful live or async collaboration dominating the window
* the activity appears restorative / disengaged / aimless rather than goal-directed

Examples:
* aimless scrolling between tasks
* idle computer / no interaction
* detached browsing with no task continuity

DO NOT use Idle for:
* reading with low interaction
* watching educational content
* monitoring systems intentionally

If the window only shows `Read` / `Watch` with no artifact touch and the goal-directedness is unclear, treat this as an intent-sensitive boundary with `WeakEvidence` or `MixedUnknown` rather than forcing Idle.

---

### Step 2 — Attend

Assign Attend if:
The dominant frame is participation in a real-time shared session or live event.

This includes:
* meetings (Zoom, Meet, Teams)
* live lectures
* interviews
* pair sessions (voice/video dominant)

**Dominance rule**
Use Attend ONLY if:
* the session defines the structure of the window
* attention is primarily governed by the live interaction

Observed anchors:
* live call in foreground
* shared live presence
* sync voice/video dominates the window

Do NOT use Attend when:
* call is background while user works independently
* watching a live stream passively without participation

---

### Step 3 — Coordinate

Assign Coordinate if the primary work output is coordination:

* managing tasks
* handling communication
* orchestrating workflow
* delegating / triaging / scheduling

Examples:
* Slack/email responses
* Jira/Linear triage
* assigning tasks
* calendar management
* async collaboration

Observed anchors:
* message surface dominates
* async text collaboration dominates
* artifact modification is absent or secondary

---

### Step 4 — Produce

Assign Produce if the window materially advances an artifact/system/deliverable.

This includes:
* creating
* modifying
* fixing
* synthesizing

Examples:
* coding
* writing documents
* building spreadsheets
* editing designs
* transforming data

Observed anchors:
* edit surface dominates
* artifact touch is `Modified` or `Created`

---

### Step 5 — Consume

Assign Consume if:
the dominant activity is understanding, inspecting, or gathering information without material artifact advancement being dominant

Examples:
* reading documentation
* watching tutorials
* browsing references
* reviewing dashboards (without acting)

Observed anchors:
* read / watch / search dominates
* artifact touch is `None` or `ReadOnly`

---

## 3. Tie-Break Rules (Mandatory)

**Produce vs Consume**
* If artifact advancement dominates -> Produce
* If understanding dominates -> Consume
* If the window alternates between both and neither clearly dominates, keep the observed facts fixed and mark the semantic choice with `intent_basis = InferredFromContext`; escalate to `WeakEvidence` or `MixedUnknown` if needed

**Coordinate vs Attend**
* Live synchronous interaction -> Attend
* Async or workflow management -> Coordinate

**Consume vs Idle**
* Goal-directed intake -> Consume
* Non-directed / restorative -> Idle
* If the observed facts only show passive reading/watching with no artifact touch, but the goal-directedness is unclear, do not pretend this is deterministic ground truth

**Produce vs Coordinate**
* Artifact is the output -> Produce
* Communication/workflow is the output -> Coordinate
* If comments, docs, and edits are tightly mixed, keep the observed layer fixed and use uncertainty explicitly

---

## 4. Subtype Rules

**Definition**
Subtype refines activity within a mode.
It MUST NOT redefine the mode.

---

### Allowed patterns (by convention)

**Produce**
* Build
* Debug
* Write
* Review
* Analyze
* Plan
* Admin

**Consume**
* ReadResearch
* Learn
* ExploreReference
* Review
* Monitor
* Analyze

**Coordinate**
* Communicate
* Plan
* Admin
* Review

**Attend**
* Meet
* Learn
* Communicate

**Idle**
* BreakIdle

---

### Subtype definitions (operational)

* **Build**: Direct construction of an artifact.
* **Debug**: Active fix loop: logs / tests / traces tightly coupled to a concrete issue; iterative diagnosis → fix → verify.
* **Write**: Producing structured written content.
* **Review**: Evaluating an artifact for correctness/quality/acceptance.
* **Analyze**: Transforming or reasoning over structured data.
* **Learn**: Structured intake for skill or knowledge acquisition.
* **ExploreReference**: Unstructured browsing for ideas/examples/sources.
* **Monitor**: Observing changing systems/queues without intervention.
* **Communicate**: Direct message exchange.
* **Plan**: Structuring future work.
* **Admin**: Operational overhead (forms, cleanup, formatting, bookkeeping).
* **BreakIdle**: Intentional or default disengagement.

---

## 5. Output Domain Rules

**Definition**
The artifact or work-product domain being advanced.

NOT:
* the app
* the UI
* the surface behavior

---

**Examples**

| Activity | Output Domain |
|---|---|
| coding | Code |
| writing spec | Writing |
| reading papers | Research |
| spreadsheet modeling | Analysis |
| Jira triage | Admin |
| Slack replies | Communication |
| Figma work | Design |
| incident ops | Operations |

---

## 6. Interaction Style

Observational only.

* Active: direct manipulation, typing, creation
* Passive: watching, listening, minimal input
* Mixed: alternation
* Idle: negligible engagement

This axis NEVER overrides mode.

---

## 7. Collaboration Mode

Defined by interaction topology:

* Solo
* AsyncCollab
* SyncCollab
* Unknown

**Rules**
* Text chat → AsyncCollab
* Live call → SyncCollab
* Customer interaction counts as collaboration
* External vs internal does not matter

Start from the observed layer:
* `collaboration_surface = AsyncText` usually maps to `AsyncCollab`
* `collaboration_surface = SyncVoiceVideo` usually maps to `SyncCollab`
* semantic `collaboration_mode` may still be `Unknown` if the observed topology is incomplete

---

## 8. Boundary Cases (Canonical Rulings)

**Debugging**
* Active fix loop with artifact modification -> Produce + Debug
* General research -> Consume
* Search/log reading without clear artifact touch may stay Consume or resolve to `MixedUnknown`; do not force Produce just because the broader session is about fixing something

---

**Note-taking while reading**
* Light notes → Consume
* Heavy synthesis → Produce + Write

---

**Meetings where user presents**
* Still Attend
* speaking/editing does not change mode

---

**Background call while working**
* Call dominant → Attend
* Work dominant → non-Attend mode

---

**Customer support**
* Text → Coordinate
* Voice/video → Attend

---

**Spreadsheet work**
* Transforming data → Produce + Analyze
* Reading only → Consume

---

**Design workflows**
* Browsing inspiration → Consume + ExploreReference
* Creating/editing → Produce
* output_domain = Design

---

**Social media**
* Work-driven -> Consume
* Aimless -> Idle
* If the observed facts are only `Read` plus `artifact_touch = None`, use surrounding windows only to resolve ambiguity; if ambiguity remains, use `intent_basis = InferredFromContext` with `WeakEvidence` or `MixedUnknown`

---

**Search queries**
* Short, tightly coupled -> inherit dominant activity
* Standalone exploration -> Consume + ExploreReference
* The observed fact may simply be `activity_surface = Search`; the semantic label should remain reversible if the search intent is not obvious

---

**Code review / document review**
* Reading only → Consume + Review
* Making decisions/editing → Produce + Review
* Routing/assigning → Coordinate + Review

---

**Async doc collaboration**
* replying / triage -> Coordinate
* editing doc -> Produce
* reading feedback -> Consume
* if comments and edits are tightly interleaved, keep the observed facts stable and use uncertainty instead of inventing a single crisp intent

---

**Monitoring systems**
* watching dashboards → Consume + Monitor
* acting on system → Produce

---

## 9. Uncertainty Rules

Use support_state explicitly

Use `intent_basis` explicitly

* **ObservedOnly**: semantic label follows directly from stable observed facts
* **InferredFromContext**: semantic choice required contextual interpretation
* **UserDeclared**: user explicitly provided or corrected the intent
* **Unknown**: the semantic basis cannot be recovered confidently

**Supported**
Clear, coherent evidence.

**WeakEvidence**
Sparse signals.

**MixedUnknown**
Multiple plausible interpretations.

**Rejected**
Candidate ruled out.

---

### Annotator rules

Use MixedUnknown when:
* two modes are equally plausible
* window is heavily mixed
* evidence conflicts
* observed facts are clear but intent-sensitive interpretation remains unresolved

Use WeakEvidence when:
* label is plausible but under-supported
* context breaks a tie, but only weakly

---

**Do NOT:**
* force a label with false confidence
* ignore ambiguity
* rewrite observed facts to fit a preferred story
* treat inferred intent as deterministic truth

---

## 10. Windowing Rules

Always label inference window only
* do NOT label full session
* do NOT project session intent onto window

**Context usage**
Allowed:
* resolve ambiguity
* set `intent_basis = InferredFromContext` when the observed layer alone is insufficient

Not allowed:
* override clear current evidence
* collapse uncertainty just because the broader session narrative feels obvious

---

## 11. Anti-Patterns (Strictly Forbidden)

* Using output domain as subtype
* Labeling based on app alone
* Using inactivity as proxy for Idle
* Ignoring precedence rules
* Overfitting to session narrative
* Avoiding uncertainty labeling
* Treating semantic labels as stronger than observed facts
* Deleting the override path by hard-coding inferred intent

---

## 12. Final Mental Model

Think in layers:

1. What was directly observed? -> observed layer
2. What is happening? -> mode
3. What shape? -> subtype
4. What artifact? -> output_domain
5. How engaged? -> interaction_style
6. Who is involved? -> collaboration_mode
7. How sure am I? -> support_state
8. How was intent determined? -> intent_basis / override path

---

**Final Note**

This playbook is deliberately strict.

If ambiguity remains:
* use tie-break rules
* use uncertainty fields
* preserve the observed facts
* do not invent interpretations

A good taxonomy does not eliminate ambiguity.
It makes ambiguity explicit, bounded, and consistent.
This playbook achieves that.
