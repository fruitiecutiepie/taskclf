# Labels v2 Annotation Playbook

Version: 2.0
Status: Draft
Last Updated: 2026-04-22

This playbook serves as the source of truth for annotators assigning `labels_v2` semantic labels to inference windows. The system handles ambiguity through explicit uncertainty, but consistent annotation of the core mode and subtype is critical.

---

## 1. Dominance Checklist

To reduce human ambiguity, always follow this strict precedence checklist when evaluating a window.

1. **Is this task-directed?**
   - If NO (aimless, restorative, or completely disengaged), evaluate for **Idle**.
   - If YES, proceed to 2.
2. **Is it live synchronous?**
   - If YES (active video/voice call, live lecture, pairing session), assign **Attend**.
   - If NO, proceed to 3.
3. **Is it mainly workflow/people coordination?**
   - If YES (triaging Slack, managing Jira, scheduling, delegating), assign **Coordinate**.
   - If NO, proceed to 4.
4. **Is an artifact materially advanced?**
   - If YES (coding, writing a report, designing, spreadsheet modeling), assign **Produce**.
   - If NO, proceed to 5.
5. **Otherwise, is it intake/understanding?**
   - If YES (reading docs, watching recorded material, browsing references), assign **Consume**.

---

## 2. Canonical Examples by Mode

### Mode: Produce

**Primary characteristic:** Creating, modifying, repairing, synthesizing, or materially advancing an artifact/system/deliverable.

* **Subtype: Build**
  * *Positive:* Writing feature code in VS Code.
  * *Near Miss:* Reading stack overflow to figure out how to build the feature. (This is `Consume` -> `ExploreReference`).
  * *Counterexample:* Triaging bug tickets in Jira. (This is `Coordinate` -> `Admin`).
* **Subtype: Debug**
  * *Positive:* Running terminal commands to investigate a failing server, reading logs to fix an immediate issue.
  * *Near Miss:* Reading documentation about a generic error code without an active fix loop. (This is `Consume` -> `Learn`).
* **Subtype: Write**
  * *Positive:* Drafting a design document in Google Docs.
  * *Counterexample:* Sending a long Slack message to a colleague. (This is `Coordinate` -> `Communicate`).
* **Subtype: Analyze**
  * *Positive:* Transforming data in a spreadsheet or Jupyter notebook.
  * *Near Miss:* Checking a dashboard briefly. (This is `Consume` -> `Monitor`).

### Mode: Consume

**Primary characteristic:** Reading, watching, listening, inspecting, understanding, or gathering information without material artifact advancement.

* **Subtype: ReadResearch**
  * *Positive:* Reading an academic paper or deep-dive article.
  * *Near Miss:* Taking heavy synthesized notes while reading. (If the synthesis artifact is the main goal, this is `Produce` -> `Write`).
* **Subtype: Learn**
  * *Positive:* Watching a recorded tutorial or course.
  * *Counterexample:* Attending a live webinar. (This is `Attend` -> `Learn`).
* **Subtype: Monitor**
  * *Positive:* Reviewing operational dashboards or incident queues without taking action.
  * *Near Miss:* Executing runbooks to fix an incident. (This is `Produce` -> `Admin/Operations`).

### Mode: Coordinate

**Primary characteristic:** Managing people/tasks, sending/responding to messages, workflow orchestration.

* **Subtype: Communicate**
  * *Positive:* Toggling between Slack channels to answer questions.
  * *Near Miss:* Chatting with a customer support user in an internal tool. (This is still `Coordinate` -> `Communicate`, unless it is live video/audio).
* **Subtype: Plan**
  * *Positive:* Organizing tasks in Linear or Jira for the upcoming sprint.
  * *Counterexample:* Writing a technical architecture document. (This is `Produce` -> `Write`).

### Mode: Attend

**Primary characteristic:** Participation in a live synchronous session.

* **Subtype: Meet**
  * *Positive:* Active Zoom call with colleagues.
  * *Near Miss:* Listening to a recorded meeting. (This is `Consume` -> `Monitor/Learn`).
  * *Counterexample:* Live pair programming where you are driving the code. (While meeting is active, if the primary frame is the call, it remains `Attend`. If the call is secondary background noise to intense coding, it might flip to `Produce`, but lean toward `Attend` for active pairing).

### Mode: Idle

**Primary characteristic:** Restorative, non-task-directed, or no meaningful evidence of directed work.

* **Subtype: BreakIdle**
  * *Positive:* Computer locked or zero inputs for 5+ minutes.
  * *Near Miss:* Reading a long technical article without scrolling much. (This is `Consume` -> `ReadResearch`, not Idle, as it is task-directed).

---

## 3. Boundary Examples

These are historically difficult edge cases. Apply these rulings consistently:

* **Debugging:**
  If the session is primarily in service of changing/fixing the artifact/system, it is `Produce` -> `Debug`. If it is just gathering info, it is `Consume`.
* **Note-taking while reading:**
  If understanding/intake is the main purpose (light notes), assign `Consume` + `Active` interaction style. If the session materially advances a synthesis artifact (heavy notes), assign `Produce` + `Write`.
* **Meetings where the user presents:**
  Assign `Attend` -> `Meet`. Even if the user speaks a lot or edits slides live, `Attend` stays dominant if the live session is the main frame.
* **Social media for work vs break:**
  If researching competitors on Twitter, assign `Consume`. If doomscrolling aimlessly, assign `Idle` -> `BreakIdle`. Use the context of surrounding windows.
* **Customer support live chat:**
  Text-based async chat is `Coordinate` -> `Communicate`. Only use `Attend` for live synchronous voice/video.
* **Spreadsheet work:**
  If transforming data or building a model, it is `Produce` -> `Analyze`. If just reading rows, it is `Consume`.
* **Design browsing:**
  Browsing Dribbble/Pinterest for ideas is `Consume` -> `ExploreReference`. Iterating in Figma is `Produce` -> `Design` (OutputDomain).
