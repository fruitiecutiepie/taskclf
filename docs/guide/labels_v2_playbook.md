# Labels v2 Annotation Playbook

Version: 2.0 (Final)
Status: Production
Last Updated: 2026-04-23

This document is the authoritative contract for assigning labels_v2 to inference windows.

It optimizes for:

* Consistency over intuition
* Determinism over interpretation
* Bounded ambiguity over false precision
* Extensibility without core churn

---

## 0. Core Principles (Non-Negotiable)

1. Always assign mode using the precedence rules.
2. Do not mix axes.
    * mode ≠ subtype ≠ output_domain
3. Label the current window, not the whole session.
4. Use uncertainty explicitly. Do not guess.
5. When in doubt, follow tie-break rules, not intuition.

---

## 1. Axis Semantics (Strict Separation)

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

### Step 1 — Idle

Assign Idle if ALL are true:

* no clear task intent
* activity is restorative / disengaged / aimless
* no meaningful artifact or obligation is being advanced

Examples:
* aimless scrolling between tasks
* idle computer / no interaction
* detached browsing with no task continuity

DO NOT use Idle for:
* reading with low interaction
* watching educational content
* monitoring systems intentionally

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

Do NOT use Attend when:
* call is background while user works independently
* watching a live stream passively without participation

---

### Step 3 — Coordinate

Assign Coordinate if the primary purpose is:

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

---

### Step 4 — Produce

Assign Produce if the primary purpose is:
materially advancing an artifact/system/deliverable

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

---

### Step 5 — Consume

Assign Consume if:
the primary purpose is understanding, inspecting, or gathering information

Examples:
* reading documentation
* watching tutorials
* browsing references
* reviewing dashboards (without acting)

---

## 3. Tie-Break Rules (Mandatory)

**Produce vs Consume**
* If artifact advancement dominates → Produce
* If understanding dominates → Consume

**Coordinate vs Attend**
* Live synchronous interaction → Attend
* Async or workflow management → Coordinate

**Consume vs Idle**
* Goal-directed intake → Consume
* Non-directed / restorative → Idle

**Produce vs Coordinate**
* Artifact is the output → Produce
* Communication/workflow is the output → Coordinate

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

---

## 8. Boundary Cases (Canonical Rulings)

**Debugging**
* Active fix loop → Produce + Debug
* General research → Consume

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
* Work-driven → Consume
* Aimless → Idle
Use surrounding windows ONLY if ambiguous.

---

**Search queries**
* Short, tightly coupled → inherit dominant activity
* Standalone exploration → Consume + ExploreReference

---

**Code review / document review**
* Reading only → Consume + Review
* Making decisions/editing → Produce + Review
* Routing/assigning → Coordinate + Review

---

**Async doc collaboration**
* replying / triage → Coordinate
* editing doc → Produce
* reading feedback → Consume

---

**Monitoring systems**
* watching dashboards → Consume + Monitor
* acting on system → Produce

---

## 9. Uncertainty Rules

Use support_state explicitly

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

Use WeakEvidence when:
* label is plausible but under-supported

---

**Do NOT:**
* force a label with false confidence
* ignore ambiguity

---

## 10. Windowing Rules

Always label inference window only
* do NOT label full session
* do NOT project session intent onto window

**Context usage**
Allowed:
* resolve ambiguity

Not allowed:
* override clear current evidence

---

## 11. Anti-Patterns (Strictly Forbidden)

* Using output domain as subtype
* Labeling based on app alone
* Using inactivity as proxy for Idle
* Ignoring precedence rules
* Overfitting to session narrative
* Avoiding uncertainty labeling

---

## 12. Final Mental Model

Think in layers:

1. What is happening? → mode
2. What shape? → subtype
3. What artifact? → output_domain
4. How engaged? → interaction_style
5. Who is involved? → collaboration_mode
6. How sure am I? → support_state

---

**Final Note**

This playbook is deliberately strict.

If ambiguity remains:
* use tie-break rules
* use uncertainty fields
* do not invent interpretations

A good taxonomy does not eliminate ambiguity.
It makes ambiguity explicit, bounded, and consistent.
This playbook achieves that.
