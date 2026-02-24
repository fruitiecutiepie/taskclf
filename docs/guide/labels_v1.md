# Task Ontology v1 (Core Labels)

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines the universal, user-agnostic core task labels used by the global classifier.

These labels are:

- Stable across users
- Observable from logged signals
- Designed for time tracking
- Separable using current feature set
- Independent from user-specific taxonomies

---

# 1. Core Label Set

The classifier predicts exactly one of the following core labels per window.

Label IDs are stable and MUST NOT be reordered without version bump.

| ID | Label Name        | Description |
|----|------------------|------------|
| 0  | Build            | Writing or implementing code or structured content in an editor or terminal |
| 1  | Debug            | Investigating, debugging, or operational terminal-heavy troubleshooting |
| 2  | Review           | Reviewing code, diffs, or technical material with light edits |
| 3  | Write            | Writing non-code structured content (docs, notes, specs) |
| 4  | ReadResearch     | Consuming information with minimal content production |
| 5  | Communicate      | Asynchronous communication (chat, email, coordination) |
| 6  | Meet             | Synchronous meetings or calls |
| 7  | BreakIdle        | Idle, break, or non-productive activity |

---

# 2. Observable Labeling Rules

Labels must be assigned using observable interaction signals.
Intent or mental state must NOT be used.

Below are operational rules for human labeling consistency.

---

## 0 — Build

Primary characteristics:
- Editor or terminal foreground
- Sustained typing
- High shortcut usage
- Moderate-to-low app switching

Examples:
- Implementing features
- Refactoring
- Writing structured code

Excludes:
- Investigating logs (Debug)
- Reading documentation (ReadResearch)

---

## 1 — Debug

Primary characteristics:
- Terminal-heavy or log-heavy interaction
- Frequent switching between editor/terminal/browser
- Moderate typing
- Higher context switching than Build

Examples:
- Investigating failing test
- Checking logs
- Running commands repeatedly

Excludes:
- Writing new feature code (Build)

---

## 2 — Review

Primary characteristics:
- Reading diffs or technical material
- Light edits
- Moderate scrolling
- Lower sustained typing than Build

Examples:
- Code review
- Reviewing PR
- Reviewing technical material with occasional edits

---

## 3 — Write

Primary characteristics:
- Sustained typing
- Low shortcut rate
- Lower terminal usage
- Structured text writing

Examples:
- Writing documentation
- Writing design notes
- Writing blog posts

Excludes:
- Coding (Build)
- Messaging/chat (Communicate)

---

## 4 — ReadResearch

Primary characteristics:
- High scrolling
- Low typing
- Browser-heavy
- Low shortcut usage

Examples:
- Reading documentation
- Researching topics
- Consuming articles

---

## 5 — Communicate

Primary characteristics:
- Short bursts of typing
- Frequent app switching
- Messaging/email tools
- Moderate click activity

Examples:
- Slack/Teams chat
- Email responses
- Coordination messages

Excludes:
- Long-form writing (Write)

---

## 6 — Meet

Primary characteristics:
- Very low interaction
- Meeting app foreground
- Long uninterrupted window
- Minimal keyboard/mouse activity

Examples:
- Video calls
- Voice calls
- Screen-share sessions

---

## 7 — BreakIdle

Primary characteristics:
- Near-zero interaction
- Idle gap exceeded threshold
- No meaningful task signals

Examples:
- Lunch break
- Walking away
- Computer idle

---

# 3. Ambiguity Policy

The model outputs probabilities for all core labels.

A window may be marked as rejected if:

```

max(core_probabilities) < REJECT_THRESHOLD

```

Rejected windows:
- Are labeled as `Mixed/Unknown` at inference layer
- Are NOT part of training label space
- Are logged for active learning

Training data MUST NOT include Mixed/Unknown as a class.

---

# 4. Label Transition Guidelines

For time tracking stability:

- Adjacent windows with identical predicted labels should be merged.
- Label changes shorter than MIN_BLOCK_DURATION should be smoothed.
- BreakIdle should override other labels if idle threshold exceeded.

---

# 5. Non-Goals

This ontology does NOT:
- Encode user-specific goals
- Encode productivity judgment
- Encode domain-specific categories (e.g., PianoTeaching, Investing)
- Encode emotional states

Personalization occurs via mapping layer, not by changing core labels.

---

# 6. Versioning Rules

Any of the following require version bump:
- Adding/removing a core label
- Reordering label IDs
- Changing semantic definition
- Changing reject policy behavior

Minor clarifications that do not change semantics do not require version bump.
