# TODO — Tray System

Issues identified from user-use-case analysis of `taskclf tray`.

---

## High Priority

### 1. EventBus not shared in native window mode
**File:** `tray.py` — `_start_ui_subprocess()` vs `_start_ui_embedded()`

`--browser` runs FastAPI in-process and shares the tray's `EventBus`.
Without `--browser`, the UI is spawned as a child process (`taskclf ui`) with its own isolated `EventBus`.
The `status`, `tray_state`, `suggest_label`, and `prediction` events from `ActivityMonitor` and `TrayLabeler` never reach the subprocess's web UI.
StatePanel, LiveBadge predictions, suggestion banners, and transition progress bars all show nothing in native mode.

**Fix:** Either forward events over an IPC channel (e.g. Unix socket / localhost TCP) or always run FastAPI in-process regardless of `--browser`.

---

## Medium Priority

### 2. Fake "unknown" prediction published when no model is loaded
**File:** `tray.py:470-479`

When a transition fires without a model, a `prediction` event with `label: "unknown"`, `confidence: 0.0`, `provenance: "model"` is published.
The LiveBadge and StatePanel display this as if the model predicted "unknown" — misleading users into thinking the model ran and failed.

**Fix:** Publish a distinct `transition` event type (or omit the prediction entirely) so the UI can differentiate "no model" from "model predicted unknown".

### 3. "now" preset creates a 1-second label span
**File:** `LabelGrid.tsx:98`

`selectedMinutes = 0` → `start = now - max(0, 1000)` → 1-second span.
A 1-second label is useless for training and pollutes the dataset.

**Fix:** Make "now" mean "since last label end_ts" or "since last transition", falling back to a configurable minimum (e.g. 1 minute).

### 4. No pause/resume capability
**Files:** `tray.py`, `server.py`

No mechanism to pause monitoring. Users must kill and restart the process to stop tracking, losing session state (`poll_count`, `uptime_s`, `transition_count`).

**Fix:** Add a `pause` tray menu item and `POST /api/tray/pause` endpoint that sets `ActivityMonitor._stop` temporarily without clearing state.

### 5. Raw app names exposed in desktop notifications
**File:** `tray.py:488-494`

Notifications display `"{prev_app} → {new_app}"` in plain text. Someone viewing the user's screen can read their app usage. AGENTS.md privacy rules hash window titles but app names in notifications are unprotected.

**Fix:** Add a config option to disable notifications or obfuscate app names (e.g. show only the label category, not the raw app identifier).

### 6. No label overlap guidance in the UI
**Files:** `server.py:195-196`, `LabelGrid.tsx:113`

The server returns HTTP 409 on overlapping spans. The LabelGrid catches this as a generic `"Error: <message>"` with no guidance on which label conflicts or how to resolve it.

**Fix:** Return structured error data (conflicting span timestamps, label) in the 409 response. Show actionable UI: "Overlaps with [Label] from HH:MM–HH:MM. Delete it or adjust the time window."

### 7. Transition detection cold start gap
**File:** `tray.py:192-195`

On startup, the first poll sets `_current_app` but never fires a transition. All activity before the first detected transition is unlabeled. Users who start the tray mid-session have a coverage gap.

**Fix:** Publish an `initial_block` event on first poll (or after N polls confirm the initial app) so the UI can prompt the user to label the pre-start period.

### 8. Suggestion never expires
**File:** `ws.ts:89-90`, `LabelGrid.tsx`

`activeSuggestion` persists until manually dismissed or replaced by a new suggestion. If no new transitions occur, a stale suggestion from hours ago remains visible.

**Fix:** Add a TTL (e.g. 10 minutes) after which the suggestion auto-dismisses, or clear it when the user creates any label.

---

## Low Priority

### 9. `labels_saved_count` is always zero
**File:** `tray.py:386`

`self._labels_saved_count` is initialized to 0 and never incremented. Labels are saved through `POST /api/labels`, which has no back-channel to the tray. The StatePanel's "labels_saved" row always shows `0`.

**Fix:** Either increment the counter via an EventBus listener when labels are saved, or remove the field from `tray_state`.

### 10. `--no-tray` without `--browser` doesn't tell the user where the UI is
**File:** `tray.py:672-683`

Prints `"taskclf running (...), no tray icon. Press Ctrl+C to quit."` but doesn't mention the UI port or URL. The user has a running server but no way to discover it.

**Fix:** Print `"UI available at http://127.0.0.1:{port}"` in the `--no-tray` message.

### 11. `_toggle_window` doesn't toggle in browser mode
**File:** `tray.py:507-513`

In browser mode, clicking "Open Dashboard" always calls `webbrowser.open()`, which opens a new tab every time. Repeated clicks stack tabs. The method name is misleading.

**Fix:** Rename to `_open_dashboard` for clarity, or use a browser-control approach that focuses an existing tab.

### 12. LabelGrid auto-collapse has a forced 1.5s delay
**File:** `LabelGrid.tsx:109-111`

After a successful label, the grid shows a flash message for 1500ms then collapses. No way to dismiss early. In rapid labeling workflows this adds friction.

**Fix:** Allow click-to-dismiss or reduce the delay. Consider keeping the grid open so the user can label again immediately.

### 13. `LabelRecent` component is dead code
**File:** `LabelRecent.tsx`

`LabelRecent` is exported but never imported anywhere — not in `App.tsx`, not in any route. It duplicates `LabelGrid` functionality (creating labels for recent time windows) with a different UX (form with range sliders vs. preset buttons). It also lacks `extend_forward` and confidence controls that `LabelGrid` has.

**Fix:** Remove `LabelRecent.tsx` or wire it into the app as a dedicated "custom time range" labeling view if that use case is needed.

### 14. `extend_forward` publishes a fake "prediction" event
**File:** `server.py:198-206`

When a label is created with `extend_forward: true`, the server publishes a `prediction` event with `provenance: "manual"`. This updates the LiveBadge to show the label — which is the desired visual effect — but it co-opts the `prediction` event type for something that isn't a prediction. The frontend then renders it as "Last Label" (via a provenance check in `StatePanel`) but it still populates `latestPrediction`, which feeds into `ActivityContext` and `LabelHistory` refetch logic. Mixing label events into the prediction channel makes the event semantics fragile.

**Fix:** Introduce a dedicated `label_created` event type. Have the frontend update the LiveBadge from that event type directly, keeping the prediction channel reserved for actual model outputs.

### 15. Candidate duration uses `poll_seconds` instead of actual elapsed time
**File:** `tray.py:199`

`self._candidate_duration += self._poll_seconds` assumes polls happen exactly on schedule. If the system is under load and a poll is delayed (e.g. AW fetch takes 5s on a 30s interval), the candidate duration is still incremented by `poll_seconds`, not the actual wall-clock time since the last poll. This can cause transitions to fire earlier or later than the configured threshold.

**Fix:** Track `_last_poll_time` and compute `elapsed = now - _last_poll_time` for duration accumulation.

### 16. EventBus silently drops slow WebSocket consumers
**File:** `events.py:36-41`

The subscriber queue has `maxsize=256`. When a consumer can't keep up, `put_nowait` raises `QueueFull` and the subscriber is permanently removed (`dead.append(q)` → `discard`). The WebSocket connection stays open but stops receiving events with no error sent to the client. The frontend has no way to detect this — it looks "connected" but receives nothing.

**Fix:** Either send a "backpressure" warning event before dropping, or evict the oldest event from the queue instead of removing the subscriber entirely.

### 17. Naive-UTC timestamps throughout tray and server
**Files:** `tray.py` (5 occurrences), `server.py`, `client.py`, `online.py`, `time.py`

All timestamps use `datetime.now(timezone.utc).replace(tzinfo=None)` — stripping timezone info to produce naive datetimes that are implicitly UTC. The frontend then re-adds `"Z"` when parsing (`iso + "Z"`). This works as long as every layer agrees on the convention, but is fragile: any component that treats a naive datetime as local time will silently produce wrong results. The pattern appears in at least 5 files.

**Fix:** Keep `tzinfo=timezone.utc` on all internal datetimes. Emit ISO strings with the `Z` suffix from the server. Remove the frontend's `iso + "Z"` fallback hack.
