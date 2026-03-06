# ui.server

Web UI for ground-truth label collection and live prediction monitoring.

## Launch

```bash
taskclf ui
taskclf ui --port 8741 --model-dir models/run_20260226
```

For frontend development with hot reload:

```bash
taskclf ui --dev
```

Options:

| Option | Default | Description |
|---|---|---|
| `--port` | `8741` | Port for the web server |
| `--model-dir` | *(none)* | Model bundle for live predictions |
| `--aw-host` | `http://localhost:5600` | ActivityWatch server URL |
| `--poll-seconds` | `60` | Seconds between AW polling |
| `--title-salt` | `taskclf-default-salt` | Salt for hashing window titles |
| `--data-dir` | `data/processed` (ephemeral in `--dev`) | Processed data directory; omit with `--dev` for an auto-cleaned temp dir |
| `--transition-minutes` | `3` | Minutes before suggesting a label change |
| `--dev` | off | Start Vite dev server for frontend hot reload; uses ephemeral data dir unless `--data-dir` is set |

## Panels

- **Label** -- Form with date/time pickers, `CoreLabel` dropdown, confidence slider, and user ID input.
- **Recent** -- Quick-label with preset durations (now / 1 / 5 / 10 / 15 / 30 / 60 min) or a custom duration input supporting seconds, minutes, hours, and days. "now" labels the span from the last label's `end_ts` to the current moment (falling back to 1 minute if no last label exists or the span would be < 1 minute); the button dynamically shows the duration (e.g. "now (12m)"). Other values label the corresponding trailing window. The "Extend until next label" checkbox (on by default) sets `extend_forward=true` on the new label; when the *next* label is created, this label's `end_ts` is automatically stretched to meet the next label's `start_ts`, producing contiguous coverage without gaps. Shows a live ActivityWatch summary when available. A compact "Last: *Label* Nm ago" indicator below the buttons provides continuity context without requiring the full history view. After a successful label, a brief "Saved" flash appears (click to dismiss instantly); the grid stays open for rapid consecutive labeling. On overlap errors (409), the flash shows the conflicting span's time range (e.g. "Overlaps 09:00–10:00").
- **Queue** -- Pending `LabelRequest` items sorted by confidence (lowest first). Shows time range, predicted label, confidence, and reason.

## Live Features

- **Live badge (compact)** -- Header pill showing the current label/app and connection dot. Visible in the collapsed tray window.
- **State panel** -- Tabbed panel with two views, selected via a segmented control at the top:
  - **System tab** -- Internal-state debug panel with a collapsible accordion layout: each section header shows an inline summary badge (e.g., current app, predicted label, connection status) so all states are scannable at a glance. Click any header to expand its detail rows. **Activity Monitor** and **Last Prediction** default to open; all other sections start collapsed. Eight sections:
    - **Activity Monitor** -- summary: `state · current_app`. Details: `state`, `current_app`, `since`, `poll_interval`, `poll_count`, `last_poll` timestamp, `uptime`. When a transition candidate exists: `candidate_app`, `candidate_progress` with duration/threshold/percentage and a visual progress bar.
    - **Last Prediction** -- summary: `mapped_label confidence%`. Details: `label`, `mapped_label`, `confidence` (color-coded green/red at 50% threshold), `ts`, `trigger_app`.
    - **Model** -- summary: `loaded`/`not loaded` (color-coded). Details: `loaded` status, `model_dir`, `schema_hash`, `suggested` label, `suggestion_conf`.
    - **Transitions** -- summary: transition count. Details: `total` count, last transition details: `prev → new` apps, `block` time range, `fired_at` timestamp.
    - **Active Suggestion** -- appears when the model suggests a label on transition. Summary: `suggested confidence%`. Details: `suggested`, `confidence`, `reason`, `old_label`, `block` time range.
    - **ActivityWatch** -- summary: `connected`/`disconnected` (color-coded). Details: AW `connection` status, `host`, `bucket_id`, `last_events` count, and **app distribution** (top 5 apps with event counts from the last poll).
    - **WebSocket** -- summary: connection status (color-coded). Details: `status`, `messages` total, per-type `breakdown` (st/pred/tray/sug), `last_received` timestamp, `reconnects` count, `connected_since`.
    - **Config** -- summary: `dev`/`prod`. Details: `data_dir`, `ui_port`, `dev_mode`, `labels_saved` count.
  - **History tab** -- Recent labels grouped by date with a color-coded timeline strip and per-entry time ranges. Click any label row to expand an inline editor showing the ActivityContext for that time window (apps used, input stats), a label-change grid (current label highlighted), and a delete button with confirmation. Provides a dedicated review surface separate from the quick-label popup, with the full panel height available for browsing.
- **Suggestion banner** -- Appears when the ActivityMonitor detects a task change. Accept or dismiss with one click.

## Architecture

The UI is a SolidJS single-page application served by a FastAPI backend:

- **REST endpoints** (`/api/labels`, `/api/labels/export`, `/api/labels/import`, `/api/labels/stats`, `/api/queue`, `/api/features/summary`, `/api/aw/live`, `/api/config/labels`, `/api/config/user`, `/api/tray/pause`, `/api/tray/state`) handle label CRUD, import/export, stats, queue management, user configuration, tray control, and data queries.
  - `POST /api/labels` accepts an optional `extend_forward` boolean. When true, the label is persisted with `extend_forward=true`; when the *next* label is created for the same user, this label's `end_ts` is automatically stretched to the new label's `start_ts`, producing contiguous coverage. The quick-label UI sets this flag by default. On overlap (409), the response body contains structured conflict details: `{"detail": {"error": "...", "conflicting_start_ts": "...", "conflicting_end_ts": "..."}}` so the frontend can show which existing label conflicts and its time range.
  - `PUT /api/labels` changes the label on an existing span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `DELETE /api/labels` removes a span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `GET /api/labels/export` downloads all label spans as a CSV file (`text/csv`). Returns 404 if no labels file exists or the file contains no spans.
  - `POST /api/labels/import` accepts a multipart CSV file upload (`file`) and an optional `strategy` form field (`"merge"` or `"overwrite"`, default `"merge"`). In merge mode, imported spans are deduplicated against existing labels by `(start_ts, end_ts, user_id)` and overlap-checked; conflicts return 409. In overwrite mode, all existing labels are replaced. Returns `{"status": "ok", "imported": N, "total": M, "strategy": "merge"|"overwrite"}`. Returns 422 on invalid CSV or strategy.
  - `GET /api/labels/stats` returns labeling statistics for a given day. Accepts an optional `date` query parameter (ISO-8601 date string, defaults to today UTC). Returns `{"date": "2026-03-01", "count": 5, "total_minutes": 75.0, "breakdown": {"Build": 45.0, "Meet": 20.0, "Write": 10.0}}`.
  - `POST /api/tray/pause` toggles the monitoring pause state. Returns `{"status": "ok", "paused": true/false}` when connected to a tray, or `{"status": "unavailable", "paused": false}` when no tray callbacks are configured.
  - `GET /api/tray/state` returns the current tray pause state: `{"available": true/false, "paused": true/false}`.
- **WebSocket** (`/ws/predictions`) streams live events from the ActivityMonitor:
  - `status` -- every poll cycle: `state` (`"collecting"` or `"paused"`), `current_app`, `current_app_since`, `candidate_app`, `candidate_duration_s`, `transition_threshold_s`, `poll_seconds`, `poll_count`, `last_poll_ts`, `uptime_s`, `aw_connected`, `aw_bucket_id`, `aw_host`, `last_event_count`, `last_app_counts`. When monitoring is paused, `state` is `"paused"` and polling/transition detection is skipped.
  - `tray_state` -- every poll cycle: `model_loaded`, `model_dir`, `model_schema_hash`, `suggested_label`, `suggested_confidence`, `transition_count`, `last_transition` (with `prev_app`, `new_app`, `block_start`, `block_end`, `fired_at`), `labels_saved_count`, `data_dir`, `ui_port`, `dev_mode`, `paused`.
  - `initial_app` -- once on startup when the first dominant app is detected: `app`, `ts`. Allows the UI to prompt the user to label the pre-start period that would otherwise be unlabeled.
  - `prediction` -- on app transition with model suggestion: `label`, `confidence`, `ts`, `mapped_label`, `current_app`. Reserved for actual model outputs; manual labels no longer use this event type.
  - `no_model_transition` -- on app transition without a loaded model: `current_app`, `ts`, `block_start`, `block_end`. The frontend uses this to distinguish "no model loaded" from "model predicted unknown"; `LiveBadge` shows "No Model" instead of "Unknown Label" when `trayState.model_loaded === false`.
  - `label_created` -- when a label with `extend_forward=true` is created via `POST /api/labels`: `label`, `confidence`, `ts` (end), `start_ts`, `extend_forward`. Replaces the former `prediction` event with `provenance: "manual"`. The frontend maps this to a `Prediction`-compatible object so `LiveBadge` and `StatePanel` update automatically.
  - `suggestion_cleared` -- published after every successful label save (via `POST /api/labels` or `POST /api/notification/accept`): `reason` (e.g. `"label_saved"`). Clients clear the active suggestion on receipt. The frontend also applies a 10-minute TTL: if no new `suggest_label` event replaces it, the suggestion auto-dismisses.
  - `suggest_label` -- on app transition with model suggestion: `suggested`, `confidence`, `reason`, `old_label`, `block_start`, `block_end`.
  - `prompt_label` -- on task transition with labeling prompt: `prev_app`, `new_app`, `block_start`, `block_end`, `duration_min`, `suggested_label`, `suggested_confidence`.
  - `show_label_grid` -- triggered by `POST /api/window/show-label-grid`: `type` (`"show_label_grid"`, no other fields).

  **Backpressure policy:** Each WebSocket subscriber has a 256-event queue. When the queue is full, the oldest event is evicted to make room for the new one. The subscriber is never silently dropped; it continues receiving events at the cost of missing stale ones.

**Timestamp format:** All ISO-8601 timestamps emitted by the server (REST responses and WebSocket events) include an explicit UTC timezone suffix (`+00:00`). Incoming timestamps in request bodies are accepted with or without timezone info; aware timestamps are normalized to naive UTC before Parquet storage for backward compatibility.

## Privacy

The UI never displays raw window titles, keystrokes, or URLs.
Only aggregated metrics and application identifiers are shown.
Desktop notifications redact app names by default (`privacy_notifications=True`);
set to `False` to show raw app identifiers in notifications.
Notifications can be disabled entirely with `notifications_enabled=False`.

::: taskclf.ui.server

---

# ui.tray

System tray labeling app for continuous background labeling.

## Launch

```bash
taskclf tray
taskclf tray --model-dir models/run_20260226
taskclf tray --username alice
taskclf tray --retrain-config configs/retrain.yaml
taskclf tray --dev
```

## Features

- **System tray icon** -- runs persistently in the background via pystray.
- **In-process web UI server** -- the FastAPI server always runs in-process, sharing the tray's `EventBus`. In `--browser` mode the dashboard opens in the default browser; otherwise a lightweight pywebview subprocess provides a native floating window. Both modes receive the same live events (status, tray_state, suggest_label, prediction) because the server and the tray publish/subscribe on the same `EventBus` instance.
- **Activity transition detection** -- polls ActivityWatch and detects when the dominant foreground app changes. A transition fires when the new app persists for >= `--transition-minutes` (default 3). On the first poll, an `initial_app` event is published so the UI can prompt labeling for the pre-start period.
- **Pause/resume** -- monitoring can be paused via the tray menu ("Pause"/"Resume") or the `POST /api/tray/pause` REST endpoint. When paused, polling and transition detection are skipped but session state (poll count, transitions) is preserved. The `status` event emits `state: "paused"` and the `tray_state` event includes `paused: true`.
- **Desktop notifications** -- on each transition, a notification prompts the user to label the completed block. By default, app names are redacted for privacy (`privacy_notifications=True`). Set `privacy_notifications=False` to show raw app identifiers. Notifications can be disabled entirely with `notifications_enabled=False`.
- **Label suggestions** -- when `--model-dir` is provided, the app predicts a label and includes it in the notification. Without a model, all 8 core labels are shown.
- **Quick-label menus** -- right-click the tray icon to label the last 5/10/15/30 minutes with any core label.
- **Label Stats** -- tray menu action that shows a desktop notification summarizing today's labeling progress: total label count, total time, and per-label breakdown (e.g. "Today: 5 labels, 1h 15m -- Build 45m, Debug 20m, Write 10m"). Also available via `GET /api/labels/stats` for programmatic access.
- **Import Labels** -- tray menu action that imports label spans from a CSV file. Opens a file-open dialog (via tkinter) to choose the source CSV, then prompts the user to merge with existing labels or overwrite them. Merge deduplicates by `(start_ts, end_ts, user_id)` and rejects overlapping spans; overwrite replaces all labels. Also available via `POST /api/labels/import` for programmatic access.
- **Export Labels** -- tray menu action that exports all label spans to a CSV file. Opens a save-file dialog (via tkinter) to choose the destination; falls back to `<data_dir>/labels_v1/labels_export.csv` when tkinter is unavailable. Also available via `GET /api/labels/export` for programmatic access.
- **Status** -- tray menu action that shows a desktop notification with connection and session status: ActivityWatch connection state, poll count, transition count, saved labels count, and loaded model name.
- **Open Data Folder** -- tray menu action that opens the data directory in the OS file manager (Finder on macOS, `xdg-open` on Linux). Falls back to a notification showing the path if the file manager cannot be launched.
- **Model submenu** -- a "Model" submenu listing all valid model bundles found in `--models-dir` (default `models/`). The submenu auto-refreshes on every menu open by re-scanning the models directory, so new bundles created by retraining appear immediately without restarting the tray. The currently loaded model shows a check mark (radio-button effect). Clicking a different bundle hot-swaps the in-memory suggester without restarting. A "(No Model)" entry unloads the model entirely. The submenu also contains "Reload Model" to re-read the current bundle from disk and "Check Retrain" to run the retrain eligibility check. When `--models-dir` is missing or empty, a disabled "(no models found)" placeholder is shown.
- **Check Retrain** -- tray menu action (inside the Model submenu) that checks whether retraining is due based on the latest model's age and the configured cadence. Shows a notification with "Retrain recommended" (with model name and creation date) or "Model is current". Uses `--retrain-config` to load a custom retrain YAML config; falls back to defaults when not provided. Disabled when `--models-dir` is not set.
- **Open Web UI** -- menu option to open the labeling web UI in the browser.
- **Event broadcasting** -- publishes `status`, `tray_state`, `initial_app`, `prediction`, `label_created`, and `suggest_label` events to the shared EventBus for connected WebSocket clients.

## Privacy

Same guarantees as the web UI: no raw window titles or keystrokes are displayed or stored.
Desktop notifications redact app names by default.

::: taskclf.ui.tray
