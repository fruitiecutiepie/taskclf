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
- **Recent** -- Quick-label with preset durations (now / 1 / 5 / 10 / 15 / 30 / 60 min) or a custom duration input supporting seconds, minutes, hours, and days. "now" creates a point label at the current moment; other values label the corresponding trailing window. The "Extend until next label" checkbox (on by default) sets `extend_forward=true` on the new label; when the *next* label is created, this label's `end_ts` is automatically stretched to meet the next label's `start_ts`, producing contiguous coverage without gaps. Shows a live ActivityWatch summary when available. A compact "Last: *Label* Nm ago" indicator below the buttons provides continuity context without requiring the full history view.
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

- **REST endpoints** (`/api/labels`, `/api/queue`, `/api/features/summary`, `/api/aw/live`, `/api/config/labels`, `/api/config/user`) handle label CRUD, queue management, user configuration, and data queries.
  - `POST /api/labels` accepts an optional `extend_forward` boolean. When true, the label is persisted with `extend_forward=true`; when the *next* label is created for the same user, this label's `end_ts` is automatically stretched to the new label's `start_ts`, producing contiguous coverage. The quick-label UI sets this flag by default.
  - `PUT /api/labels` changes the label on an existing span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `DELETE /api/labels` removes a span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
- **WebSocket** (`/ws/predictions`) streams live events from the ActivityMonitor:
  - `status` -- every poll cycle: `state`, `current_app`, `current_app_since`, `candidate_app`, `candidate_duration_s`, `transition_threshold_s`, `poll_seconds`, `poll_count`, `last_poll_ts`, `uptime_s`, `aw_connected`, `aw_bucket_id`, `aw_host`, `last_event_count`, `last_app_counts`.
  - `tray_state` -- every poll cycle: `model_loaded`, `model_dir`, `model_schema_hash`, `suggested_label`, `suggested_confidence`, `transition_count`, `last_transition` (with `prev_app`, `new_app`, `block_start`, `block_end`, `fired_at`), `labels_saved_count`, `data_dir`, `ui_port`, `dev_mode`.
  - `prediction` -- on app transition without a suggestion: `label`, `confidence`, `ts`, `mapped_label`, `current_app`.
  - `suggest_label` -- on app transition with model suggestion: `suggested`, `confidence`, `reason`, `old_label`, `block_start`, `block_end`.
  - `prompt_label` -- on task transition with labeling prompt: `prev_app`, `new_app`, `block_start`, `block_end`, `duration_min`, `suggested_label`, `suggested_confidence`.
  - `show_label_grid` -- triggered by `POST /api/window/show-label-grid`: `type` (`"show_label_grid"`, no other fields).

## Privacy

The UI never displays raw window titles, keystrokes, or URLs.
Only aggregated metrics and application identifiers are shown.

::: taskclf.ui.server

---

# ui.tray

System tray labeling app for continuous background labeling.

## Launch

```bash
taskclf tray
taskclf tray --model-dir models/run_20260226
taskclf tray --username alice
taskclf tray --dev
```

## Features

- **System tray icon** -- runs persistently in the background via pystray.
- **Embedded web UI server** -- automatically starts the FastAPI server so "Show/Hide Window" opens the labeling dashboard in a browser without needing a separate `taskclf ui` process.
- **Activity transition detection** -- polls ActivityWatch and detects when the dominant foreground app changes. A transition fires when the new app persists for >= `--transition-minutes` (default 3).
- **Desktop notifications** -- on each transition, a notification prompts the user to label the completed block.
- **Label suggestions** -- when `--model-dir` is provided, the app predicts a label and includes it in the notification. Without a model, all 8 core labels are shown.
- **Quick-label menus** -- right-click the tray icon to label the last 5/10/15/30 minutes with any core label.
- **Open Web UI** -- menu option to open the labeling web UI in the browser.
- **Event broadcasting** -- publishes `status`, `tray_state`, `prediction`, and `suggest_label` events to the shared EventBus for connected WebSocket clients.

## Privacy

Same guarantees as the web UI: no raw window titles or keystrokes are displayed or stored.

::: taskclf.ui.tray
