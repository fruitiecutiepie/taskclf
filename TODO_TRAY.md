# TODO — Tray System

Issues identified from user-use-case analysis of `taskclf tray`.

---

## High Priority

### ~~1. EventBus not shared in native window mode~~ DONE

FastAPI now always runs in-process with the tray's shared `EventBus`, regardless of `--browser`. The server startup logic is extracted into `_start_server()` which both `_start_ui_embedded()` and `_start_ui_subprocess()` call. In native window mode, the subprocess now spawns only a lightweight pywebview shell (`python -m taskclf.ui.window --port PORT`) instead of a full `taskclf ui` process — no duplicate `ActivityMonitor` or `EventBus`. All tray events (`status`, `tray_state`, `suggest_label`, `prediction`) now reach the web UI in both modes.
**Note:** The `POST /api/window/toggle` endpoint is unavailable in native window mode (same as browser mode) because the `WindowAPI` is not shared across processes. Window visibility is managed by pywebview directly. The `labels_saved_count` counter (item #9) now works in both modes.

---

## Medium Priority

### ~~2. Fake "unknown" prediction published when no model is loaded~~ DONE

Transitions without a model now publish a `no_model_transition` event instead of a fake `prediction` event.
The frontend can distinguish "no model loaded" from "model predicted unknown".
**Note:** Frontend components (LiveBadge, StatePanel) should be updated to handle the new event type; currently they will simply ignore it (no visual regression, but the "no model" state won't render a badge until the frontend is updated).

### 3. "now" preset creates a 1-second label span
**File:** `LabelGrid.tsx:98`

`selectedMinutes = 0` → `start = now - max(0, 1000)` → 1-second span.
A 1-second label is useless for training and pollutes the dataset.

**Fix:** Make "now" mean "since last label end_ts" or "since last transition", falling back to a configurable minimum (e.g. 1 minute).

### ~~4. No pause/resume capability~~ DONE

`ActivityMonitor` now has `pause()`, `resume()`, and `is_paused` property using a `threading.Event`. When paused, the `run()` loop skips polling and transition detection but still publishes `status` events with `state: "paused"`. Session state (`poll_count`, `transition_count`, `current_app`) is fully preserved across pause/resume cycles.
`TrayLabeler` exposes `_toggle_pause()` and adds a dynamic "Pause"/"Resume" menu item. The `tray_state` event now includes `paused: bool`.
Server-side: `POST /api/tray/pause` toggles pause state; `GET /api/tray/state` returns current state. Both return `"unavailable"` when not connected to a tray.

### ~~5. Raw app names exposed in desktop notifications~~ DONE

`TrayLabeler` now accepts `notifications_enabled` (default `True`) and `privacy_notifications` (default `True`).
When `notifications_enabled=False`, `_send_notification()` is a no-op.
When `privacy_notifications=True` (the default), notification messages show "Activity changed" instead of raw app identifiers. Set to `False` to opt in to showing raw app names.
Both parameters are also exposed on `run_tray()`.

### ~~6. No label overlap guidance in the UI~~ DONE

The server now returns structured 409 responses with an `OverlapErrorDetail` body: `{"error": "...", "conflicting_start_ts": "...", "conflicting_end_ts": "..."}`. Both `POST /api/labels` and `POST /api/notification/accept` parse the overlap error to identify the conflicting (existing) span's timestamps.
**Note:** The frontend (`LabelGrid.tsx`) still shows a generic error message; it should be updated to display the conflicting span details (e.g. "Overlaps with label from HH:MM–HH:MM").

### ~~7. Transition detection cold start gap~~ DONE

`ActivityMonitor` now accepts an `on_initial_app(app, ts)` callback. On the first `check_transition()` call (when `_current_app is None`), the callback fires after setting the initial app. It fires exactly once.
`TrayLabeler` wires this to `_handle_initial_app()`, which publishes an `initial_app` event via the EventBus: `{"type": "initial_app", "app": <app_id>, "ts": <iso_ts>}`.
**Note:** The frontend does not yet handle this event type; it will be silently ignored until a UI component is added to prompt labeling for the pre-start period.

### 8. Suggestion never expires
**File:** `ws.ts:89-90`, `LabelGrid.tsx`

`activeSuggestion` persists until manually dismissed or replaced by a new suggestion. If no new transitions occur, a stale suggestion from hours ago remains visible.

**Fix:** Add a TTL (e.g. 10 minutes) after which the suggestion auto-dismisses, or clear it when the user creates any label.

---

## Low Priority

### ~~9. `labels_saved_count` is always zero~~ DONE

`create_app()` now accepts an `on_label_saved` callback. The server calls it after successful saves in `POST /api/labels` and `POST /api/notification/accept`. `TrayLabeler._start_server()` passes `_on_label_saved` which increments `_labels_saved_count`.
Works in both `--browser` and native window modes since the server always runs in-process (item #1 fixed).

### ~~10. `--no-tray` without `--browser` doesn't tell the user where the UI is~~ DONE

The `--no-tray` message now prints `"UI available at http://127.0.0.1:{port}"`.

### ~~11. `_toggle_window` doesn't toggle in browser mode~~ DONE

Renamed to `_open_dashboard` for accuracy. The menu label already says "Open Dashboard".

### 12. LabelGrid auto-collapse has a forced 1.5s delay
**File:** `LabelGrid.tsx:109-111`

After a successful label, the grid shows a flash message for 1500ms then collapses. No way to dismiss early. In rapid labeling workflows this adds friction.

**Fix:** Allow click-to-dismiss or reduce the delay. Consider keeping the grid open so the user can label again immediately.

### ~~13. `LabelRecent` component is dead code~~ DONE

`LabelRecent.tsx` has been removed. It was never imported in `App.tsx` or any other component and duplicated `LabelGrid` functionality without `extend_forward` or confidence controls.

### ~~14. `extend_forward` publishes a fake "prediction" event~~ DONE

The server now publishes a `label_created` event (with `label`, `confidence`, `ts`, `start_ts`, `extend_forward`) instead of a `prediction` event with `provenance: "manual"`. The `prediction` channel is reserved for actual model outputs.
**Note:** The frontend (LiveBadge, StatePanel) should be updated to handle the `label_created` event type; until then, the badge won't update on `extend_forward` labels (no regression in prediction display).

### ~~15. Candidate duration uses `poll_seconds` instead of actual elapsed time~~ DONE

`ActivityMonitor.check_transition()` now tracks `_last_check_time` and computes actual wall-clock elapsed time instead of assuming `poll_seconds`. An optional `_now` parameter enables deterministic testing.

### ~~16. EventBus silently drops slow WebSocket consumers~~ DONE

`EventBus.publish()` now evicts the oldest event from a full subscriber queue instead of permanently removing the subscriber. The subscriber stays registered and continues receiving new events; only stale events are lost under backpressure.

### 17. Naive-UTC timestamps throughout tray and server
**Files:** `tray.py` (5 occurrences), `server.py`, `client.py`, `online.py`, `time.py`

All timestamps use `datetime.now(timezone.utc).replace(tzinfo=None)` — stripping timezone info to produce naive datetimes that are implicitly UTC. The frontend then re-adds `"Z"` when parsing (`iso + "Z"`). This works as long as every layer agrees on the convention, but is fragile: any component that treats a naive datetime as local time will silently produce wrong results. The pattern appears in at least 5 files.

**Fix:** Keep `tzinfo=timezone.utc` on all internal datetimes. Emit ISO strings with the `Z` suffix from the server. Remove the frontend's `iso + "Z"` fallback hack.
