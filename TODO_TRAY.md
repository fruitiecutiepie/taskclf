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
`ws.ts` now defines `NoModelTransitionEvent` and handles it in the WebSocket message switch. `LiveBadge` shows "No Model" (instead of generic "Unknown Label") when `trayState.model_loaded === false` and no prediction exists.

### ~~3. "now" preset creates a 1-second label span~~ DONE

When `selectedMinutes === 0` ("now"), `labelNow()` now computes the start as the last label's `end_ts` (fetched via `fetchLabels(1)`). If no last label exists, or the span would be < 1 minute, it falls back to 1 minute. The "now" button label dynamically shows the duration since the last label (e.g. "now (12m)", "now (1h 5m)").

### ~~4. No pause/resume capability~~ DONE

`ActivityMonitor` now has `pause()`, `resume()`, and `is_paused` property using a `threading.Event`. When paused, the `run()` loop skips polling and transition detection but still publishes `status` events with `state: "paused"`. Session state (`poll_count`, `transition_count`, `current_app`) is fully preserved across pause/resume cycles.
`TrayLabeler` exposes `_toggle_pause()` and adds a dynamic "Pause"/"Resume" menu item. The `tray_state` event now includes `paused: bool`.
Server-side: `POST /api/tray/pause` toggles pause state; `GET /api/tray/state` returns current state. Both return `"unavailable"` when not connected to a tray.
Frontend: `TrayState` interface in `ws.ts` now includes `paused: boolean`. `StatusEvent.state` union type includes `"paused"`.

### ~~5. Raw app names exposed in desktop notifications~~ DONE

`TrayLabeler` now accepts `notifications_enabled` (default `True`) and `privacy_notifications` (default `True`).
When `notifications_enabled=False`, `_send_notification()` is a no-op.
When `privacy_notifications=True` (the default), notification messages show "Activity changed" instead of raw app identifiers. Set to `False` to opt in to showing raw app names.
Both parameters are also exposed on `run_tray()`.

### ~~6. No label overlap guidance in the UI~~ DONE

The server now returns structured 409 responses with an `OverlapErrorDetail` body: `{"error": "...", "conflicting_start_ts": "...", "conflicting_end_ts": "..."}`. Both `POST /api/labels` and `POST /api/notification/accept` parse the overlap error to identify the conflicting (existing) span's timestamps.
The frontend (`LabelGrid.tsx`) now parses the structured 409 JSON from the error message and displays a user-friendly flash: "Overlaps HH:MM–HH:MM" showing the conflicting span's time range.

### ~~7. Transition detection cold start gap~~ DONE

`ActivityMonitor` now accepts an `on_initial_app(app, ts)` callback. On the first `check_transition()` call (when `_current_app is None`), the callback fires after setting the initial app. It fires exactly once.
`TrayLabeler` wires this to `_handle_initial_app()`, which publishes an `initial_app` event via the EventBus: `{"type": "initial_app", "app": <app_id>, "ts": <iso_ts>}`.
**Note:** The frontend does not yet handle this event type; it will be silently ignored until a UI component is added to prompt labeling for the pre-start period.

### ~~8. Suggestion never expires~~ DONE

The server publishes a `suggestion_cleared` event via EventBus after successful suggestion acceptance (`POST /api/notification/accept`), skip (`POST /api/notification/skip`), and tray-side auto-saves; manual `POST /api/labels` does not clear the banner. The frontend (`ws.ts`) clears on that event and optionally auto-dismisses after `suggestion_banner_ttl_seconds` from `config.toml` (`0` disables). The TTL timer is cancelled when a suggestion is explicitly dismissed, replaced, or cleared by the server.

---

## Low Priority

### ~~9. `labels_saved_count` is always zero~~ DONE

`create_app()` now accepts an `on_label_saved` callback. The server calls it after successful saves in `POST /api/labels` and `POST /api/notification/accept`. `TrayLabeler._start_server()` passes `_on_label_saved` which increments `_labels_saved_count`.
Works in both `--browser` and native window modes since the server always runs in-process (item #1 fixed).

### ~~10. `--no-tray` without `--browser` doesn't tell the user where the UI is~~ DONE

The `--no-tray` message now prints `"UI available at http://127.0.0.1:{port}"`.

### ~~11. `_toggle_window` doesn't toggle in browser mode~~ DONE

Renamed to `_open_dashboard` for accuracy. The menu label already says "Toggle Dashboard".

### ~~12. LabelGrid auto-collapse has a forced 1.5s delay~~ DONE

The success flash now auto-clears after 1.5s but no longer collapses the grid — the grid stays open for rapid consecutive labeling. The flash is also click-to-dismiss for instant clearance.

### ~~13. `LabelRecent` component is dead code~~ DONE

`LabelRecent.tsx` has been removed. It was never imported in `App.tsx` or any other component and duplicated `LabelGrid` functionality without `extend_forward` or confidence controls.

### ~~14. `extend_forward` publishes a fake "prediction" event~~ DONE

The server now publishes a `label_created` event (with `label`, `confidence`, `ts`, `start_ts`, `extend_forward`) instead of a `prediction` event with `provenance: "manual"`. The `prediction` channel is reserved for actual model outputs.
`ws.ts` now defines `LabelCreatedEvent` and maps it to a `Prediction`-compatible object with `provenance: "manual"`, updating the `latestPrediction` signal. LiveBadge and StatePanel update automatically on `extend_forward` labels — the badge shows the created label and StatePanel displays "Last Label" with the correct provenance.

### ~~15. Candidate duration uses `poll_seconds` instead of actual elapsed time~~ DONE

`ActivityMonitor.check_transition()` now tracks `_last_check_time` and computes actual wall-clock elapsed time instead of assuming `poll_seconds`. An optional `_now` parameter enables deterministic testing.

### ~~16. EventBus silently drops slow WebSocket consumers~~ DONE

`EventBus.publish()` now evicts the oldest event from a full subscriber queue instead of permanently removing the subscriber. The subscriber stays registered and continues receiving new events; only stale events are lost under backpressure.

### ~~17. Naive-UTC timestamps throughout tray and server~~ DONE

**Tray/server/online layer — DONE:**
`tray.py` — all 5 `.replace(tzinfo=None)` calls removed; internal datetimes are now timezone-aware UTC. `.isoformat()` emits `+00:00` naturally.
`server.py` — added `_utc_iso()` helper that appends `+00:00` to naive datetimes and passes aware ones through. All `LabelResponse` fields, `label_created` events, queue items, and overlap error details use `_utc_iso()`. `_to_naive_utc()` moved to `core/time.py` as public `to_naive_utc()` for reuse across layers; `server.py` imports it from there. Applied in `create_label`, `update_label`, `delete_label`, `notification_accept`, `feature_summary`, and `aw_live_summary`.
`online.py` — removed `.replace(tzinfo=None)` from `run_online_loop` poll window.
Frontend — removed `iso + "Z"` fallback hack from `LabelGrid.tsx`, `LabelHistory.tsx`, `StatePanel.tsx`. Removed `.slice(0, -1)` Z-stripping from `LabelGrid.tsx` `labelNow()`. Fixed 409 overlap detail unwrapping (`detail.detail ?? detail`) so conflicting span times display correctly.
Tests — `TestUtcHelpers` covers `_utc_iso()` and `to_naive_utc()` edge cases (naive, aware UTC, aware non-UTC). `TestAwareTimestampRoundTrip` verifies Z-suffixed, offset-suffixed, and naive timestamps are all accepted, normalized, persisted, and returned with `+00:00`. Covers `POST`, `PUT`, and `DELETE /api/labels`. `test_core_time.py` covers `to_naive_utc()` (TC-TIME-011 through TC-TIME-013).

**Adapters/CLI layer — DONE:**
`adapters/activitywatch/client.py` — `_parse_timestamp()` now returns timezone-aware UTC datetimes instead of naive UTC. Naive input timestamps are assumed UTC and tagged with `tzinfo=timezone.utc`. Downstream consumers (`align_to_bucket()`, sorting, URL builder) already handle aware timestamps.
`cli/main.py` — timestamps are generated as aware UTC via `datetime.now(timezone.utc)`. Explicit `to_naive_utc()` normalization is applied before `LabelSpan` creation and `generate_label_summary()` calls (which compare against naive-UTC feature DataFrames). The `.replace(tzinfo=None)` pattern is eliminated.
Tests — `test_adapters_aw.py` `test_utc_conversion` assertion updated for aware UTC. All 1336 tests pass, zero cascading failures.

**Core pipeline layer — DONE:**
`core/time.py` — `align_to_bucket()` and `generate_bucket_range()` now return timezone-aware UTC datetimes (`tzinfo=timezone.utc`) instead of naive UTC. Docstrings updated.
`core/types.py` — `FeatureRow` gains a `_ensure_aware_utc` field validator on `bucket_start_ts` and `bucket_end_ts` that auto-tags naive datetimes as UTC and converts non-UTC aware datetimes to UTC. This prevents cascade to downstream code constructing FeatureRows with naive timestamps.
`features/build.py` — `generate_dummy_features()` now produces aware UTC timestamps directly.
`features/windows.py` — `app_switch_count_in_window()` normalises timestamps to POSIX epoch before comparison, preventing `TypeError` when mixing naive event timestamps with aware bucket timestamps.
`labels/store.py` — `generate_label_summary()` detects UTC-aware DataFrame columns and normalises comparison timestamps accordingly.
Tests — `test_core_time.py` assertions updated for aware UTC results. `test_features_from_aw.py` `bucket_start_ts` assertion updated. All 1336 tests pass, zero cascading failures.
