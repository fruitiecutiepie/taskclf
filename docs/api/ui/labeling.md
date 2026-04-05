# ui.server

Web UI for ground-truth label collection and live prediction monitoring.
The tray-side orchestration classes (`ActivityMonitor`, `_LabelSuggester`,
`TrayLabeler`) are implemented as slotted dataclasses with equivalent behavior.

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
| `--idle-transition-minutes` | `1` | Minutes before a lockscreen/idle transition fires (separate from general transitions) |
| `--dev` | off | Start Vite dev server for frontend hot reload; uses ephemeral data dir unless `--data-dir` is set |

## Panels

- **Label** -- Form with date/time pickers, `CoreLabel` dropdown, confidence slider, and user ID input.
- **Recent** -- Quick-label with preset durations (now / 1 / 5 / 10 / 15 / 30 / 60 min) or a custom duration input supporting seconds, minutes, hours, and days. "now" labels the span from the last label's `end_ts` to the current moment (falling back to 1 minute if no last label exists or the span would be < 1 minute); the button dynamically shows the duration (e.g. "now (12m)"). Other values label the corresponding trailing window. The "Extend until next label" checkbox (on by default) sets `extend_forward=true` on the new label; when the *next* label is created, this label's `end_ts` is automatically stretched to meet the next label's `start_ts`, producing contiguous coverage without gaps. For a zero-duration "now" label with `extend_forward=true`, the server performs a same-timestamp handoff by ending the currently active same-user label at `now`, then creating the new open-ended label at `now`, so no overlap dialog is needed in the common switch-label-now flow. The History view immediately renders the new span as open-ended (`start – Now`, duration `until next label`) so users get instant confirmation that the label was recorded before the next label closes the span. Shows a live ActivityWatch summary when available. A compact "Last: *Label* Nm ago" indicator below the buttons provides continuity context without requiring the full history view. After a successful label, a brief "Saved" flash appears (click to dismiss instantly); the grid stays open for rapid consecutive labeling. On overlap errors, instead of a dead-end flash the grid shows a compact confirmation prompt listing the conflicting label names (e.g. "Overlaps 3 labels: Write, Debug, Review") with the affected time range shown below (e.g. "(10:50–11:20 will be replaced)"), plus Overwrite/Keep All/Cancel buttons. For a single conflict the per-span time range is shown inline; for multiple conflicts a "show details" toggle reveals per-span times. Choosing Overwrite re-submits the label with `overwrite: true`, which truncates, splits, or removes the conflicting existing span(s) to make room for the new label. Choosing Keep All re-submits with `allow_overlap: true`, preserving all existing and new labels on the overlapping time range. If `lastLabel` changes while the prompt is visible (e.g. from another tab or WebSocket), the prompt is automatically dismissed since the conflict data may be stale.
- **Queue** -- Pending `LabelRequest` items sorted by confidence (lowest first). Shows time range, predicted label, confidence, and reason.

## Live Features

- **Live badge (compact)** -- Header pill showing the current label/app and connection dot. Visible in the collapsed tray window.
- **State panel** -- Tabbed panel with three views, selected via a segmented control at the top:
  - **System tab** -- Internal-state debug panel with a collapsible accordion layout: each section header shows an inline summary badge (e.g., current app, predicted label, connection status) so all states are scannable at a glance. Click any header to expand its detail rows. **Activity Monitor** and **Last Prediction** default to open; all other sections start collapsed. Eight sections:
    - **Activity Monitor** -- summary: `state · current_app`. Details: `state`, `current_app`, `since`, `poll_interval`, `poll_count`, `last_poll` timestamp, `uptime`. When a transition candidate exists: `candidate_app`, `candidate_progress` with duration/threshold/percentage and a visual progress bar.
    - **Last Prediction** -- summary: `mapped_label confidence%`. Details: `label`, `mapped_label`, `confidence` (color-coded green/red at 50% threshold), `ts`, `trigger_app`.
    - **Model** -- summary: `loaded`/`not loaded` (color-coded). Details: `loaded` status, `model_dir`, `schema_hash`, `suggested` label, `suggestion_conf`.
    - **Transitions** -- summary: transition count. Details: `total` count, last transition details: `prev → new` apps, `block` time range, `fired_at` timestamp.
    - **Active Suggestion** -- appears when the model suggests a label on transition. Summary: `suggested confidence%`. Details: `suggested`, `confidence`, `reason`, `old_label`, `block` time range.
    - **ActivityWatch** -- summary: `connected`/`disconnected` (color-coded). Details: AW `connection` status, `host`, `bucket_id`, `last_events` count, and **app distribution** (top 5 apps with event counts from the last poll).
    - **WebSocket** -- summary: connection status (color-coded). Details: `status`, `messages` total, per-type `breakdown` (st/pred/tray/sug), `last_received` timestamp, `reconnects` count, `connected_since`.
    - **Config** -- summary: `dev`/`prod`. Details: `data_dir`, `ui_port`, `dev_mode`, `labels_saved` count.
  - **Training tab** -- Model training interface with data readiness checks, a training form (date range, boost rounds, class weight, synthetic toggle), real-time progress via WebSocket, result display (macro/weighted F1), and a model bundle list. See [training.md](training.md) for endpoint details.
  - **History tab** -- Single-day view with date navigation (prev/next arrows and a date picker). Shows a full-day timeline strip (00:00–23:59) with color-coded labeled segments and clickable unlabeled gaps. Click any label row to expand an inline editor showing the ActivityContext for that time window (apps used, input stats), a label-change grid (current label highlighted), and a delete button with confirmation. Label time edits are minute-based and use end-exclusive semantics (`15:00-15:01` means one minute); equal start/end values (`15:00-15:00`) are rejected. Row display stays compact at `HH:MM`, but includes seconds when either boundary has non-zero seconds. Click any unlabeled gap row to expand an inline editor with time range inputs (pre-filled to the gap boundaries, adjustable to label a sub-range), ActivityContext for the selected range, and a label picker grid; selecting a label creates a new label span for the chosen sub-range. Navigating to a past date with no labels shows the entire day as a single labelable gap. Provides a dedicated review surface separate from the quick-label popup, with the full panel height available for browsing.
- **Suggestion banner** -- Appears in the label panel when a new inferred label arrives from `suggest_label`. Saving is explicit two-step confirmation (`Save Suggested Label` -> `Confirm Save`) before anything is written to label history. Dismiss calls the notification-skip flow without creating labels.
- **Auto-save BreakIdle** -- When a completed activity block is detected as idle (lockscreen app was dominant, or the model suggested `BreakIdle`), the tray auto-saves the label with `provenance="auto_idle"` and publishes a `label_created` event. No user confirmation is required since the user was away. Lockscreen/idle transitions use a separate, faster threshold (`idle_transition_minutes`, default 1 min) so breaks are detected quickly without affecting the general transition cadence (`transition_minutes`, default 3 min). The fast threshold applies in both directions: when the lockscreen app becomes dominant, and when the user returns from lockscreen. Matching uses normalized app IDs: `com.apple.loginwindow` (macOS), `com.microsoft.LockApp`/`com.microsoft.LogonUI` (Windows), `org.gnome.ScreenSaver`, `org.gnome.Shell`, `org.i3wm.i3lock`, `org.swaywm.swaylock`, `org.jwz.xscreensaver`, `org.freedesktop.light-locker`, `org.suckless.slock` (Linux).

## Architecture

The UI is a SolidJS single-page application served by a FastAPI backend:

- **REST endpoints** (`/api/labels`, `/api/labels/export`, `/api/labels/import`, `/api/labels/stats`, `/api/queue`, `/api/features/summary`, `/api/aw/live`, `/api/config/labels`, `/api/config/user`, `/api/tray/pause`, `/api/tray/state`, `/api/train/*`) handle label CRUD, import/export, stats, queue management, user configuration, tray control, model training, and data queries. See [training.md](training.md) for the `/api/train/*` endpoints.
  - `GET /api/labels` accepts optional `limit` (default 50, max 500) and `date` (ISO-8601 date string, e.g. `2025-03-07`) query parameters. When `date` is provided, only labels overlapping that day are returned. The History tab uses this to fetch labels for the selected date.
  - `POST /api/labels` accepts optional `extend_forward`, `overwrite`, and `allow_overlap` booleans. `extend_forward` persists the label with `extend_forward=true`; when the *next* label is created for the same user, this label's `end_ts` is automatically stretched to the new label's `start_ts`, producing contiguous coverage. For zero-duration `extend_forward` labels (`start_ts == end_ts`, used by "label from now"), the server first truncates the currently active same-user span at that same timestamp, then appends the new label, ensuring a contiguous no-overlap handoff. The quick-label UI sets this flag by default. When `overwrite` is `true`, conflicting same-user spans are truncated, split, or removed to make room for the new label (no 409 is returned). When `allow_overlap` is `true`, the overlap check is skipped entirely and multiple labels are allowed to coexist on the same time range; this is useful for multi-task periods. When both are `false` (default), an overlap returns 409 with structured conflict details: `{"detail": {"error": "...", "conflicting_start_ts": "...", "conflicting_end_ts": "..."}}` so the frontend can prompt the user to overwrite or keep all.
  - `PUT /api/labels` changes the label on an existing span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `DELETE /api/labels` removes a span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `GET /api/labels/export` downloads all label spans as a CSV file (`text/csv`). Returns 404 if no labels file exists or the file contains no spans.
  - `POST /api/labels/import` accepts a multipart CSV file upload (`file`) and an optional `strategy` form field (`"merge"` or `"overwrite"`, default `"merge"`). In merge mode, imported spans are deduplicated against existing labels by `(start_ts, end_ts, user_id)` and overlap-checked; conflicts return 409. In overwrite mode, all existing labels are replaced. Returns `{"status": "ok", "imported": N, "total": M, "strategy": "merge"|"overwrite"}`. Returns 422 on invalid CSV or strategy.
  - `GET /api/labels/stats` returns labeling statistics for a given day. Accepts an optional `date` query parameter (ISO-8601 date string, defaults to today UTC). Returns `{"date": "2026-03-01", "count": 5, "total_minutes": 75.0, "breakdown": {"Build": 45.0, "Meet": 20.0, "Write": 10.0}}`.
  - `POST /api/tray/pause` toggles the monitoring pause state. Returns `{"status": "ok", "paused": true/false}` when connected to a tray, or `{"status": "unavailable", "paused": false}` when no tray callbacks are configured.
  - `GET /api/tray/state` returns the current tray pause state: `{"available": true/false, "paused": true/false}`.
  - `POST /api/notification/accept` confirms an inferred label suggestion and writes it to `labels.parquet` with `provenance="suggestion"`. Required body: `{"block_start": "...", "block_end": "...", "label": "..."}`.
  - `POST /api/notification/skip` dismisses the current suggestion without saving a label and broadcasts `suggestion_cleared` with reason `"skipped"` so all connected clients clear the prompt.
- **WebSocket** (`/ws/predictions`) streams live events from the ActivityMonitor:
  - `status` -- every poll cycle: `state` (`"collecting"` or `"paused"`), `current_app`, `current_app_since`, `candidate_app`, `candidate_duration_s`, `transition_threshold_s`, `poll_seconds`, `poll_count`, `last_poll_ts`, `uptime_s`, `aw_connected`, `aw_bucket_id`, `aw_host`, `last_event_count`, `last_app_counts`. When monitoring is paused, `state` is `"paused"` and polling/transition detection is skipped.
  - `tray_state` -- every poll cycle: `model_loaded`, `model_dir`, `model_schema_hash`, `suggested_label`, `suggested_confidence`, `transition_count`, `last_transition` (with `prev_app`, `new_app`, `block_start`, `block_end`, `fired_at`), `labels_saved_count`, `data_dir`, `ui_port`, `dev_mode`, `paused`.
  - `initial_app` -- once on startup when the first dominant app is detected: `app`, `ts`. Allows the UI to prompt the user to label the pre-start period that would otherwise be unlabeled.
  - `prediction` -- on app transition with model suggestion: `label`, `confidence`, `ts`, `mapped_label`, `current_app`. Reserved for actual model outputs; manual labels no longer use this event type.
  - `no_model_transition` -- on app transition without a loaded model: `current_app`, `ts`, `block_start`, `block_end`. The frontend uses this to distinguish "no model loaded" from "model predicted unknown"; `LiveBadge` shows "No Model" instead of "Unknown Label" when `trayState.model_loaded === false`.
  - `label_created` -- when a label with `extend_forward=true` is created via `POST /api/labels`, or when the tray auto-saves a `BreakIdle` label (lockscreen/idle detection): `label`, `confidence`, `ts` (end), `start_ts`, `extend_forward`. The frontend maps this to a `Prediction`-compatible object so `LiveBadge` and `StatePanel` update automatically.
  - `suggestion_cleared` -- published after successful label saves (via `POST /api/labels` or `POST /api/notification/accept`), dismissals (`POST /api/notification/skip`), and auto-saved BreakIdle labels: `reason` (e.g. `"label_saved"`, `"skipped"`, `"auto_saved_breakidle"`). Clients clear the active suggestion on receipt. The frontend also applies a 10-minute TTL: if no new `suggest_label` event replaces it, the suggestion auto-dismisses.
  - `suggest_label` -- on app transition with model suggestion: `suggested`, `confidence`, `reason`, `old_label`, `block_start`, `block_end`.
  - `prompt_label` -- on task transition with labeling prompt: `prev_app`, `new_app`, `block_start`, `block_end`, `duration_min`, `suggested_label`, `suggested_confidence`.
  - `label_grid_show` -- triggered by `POST /api/window/show-label-grid`: `type` (`"label_grid_show"`, no other fields).
  - `train_progress` -- during training: `job_id`, `step`, `progress_pct`, `message`.
  - `train_complete` -- on training success: `job_id`, `metrics` (`macro_f1`, `weighted_f1`), `model_dir`.
  - `train_failed` -- on training failure: `job_id`, `error`.
  - `unlabeled_time` -- every poll cycle when unlabeled time exists: `unlabeled_minutes`, `text`, `last_label_end`, `ts`.
  - `gap_fill_prompt` -- at idle return (>5 min), session start, or post-acceptance: `trigger`, `unlabeled_minutes`, `text`, `last_label_end`, `ts`.
  - `gap_fill_escalated` -- when unlabeled time exceeds threshold: `unlabeled_minutes`, `threshold_minutes`.

  **Backpressure policy:** Each WebSocket subscriber has a 256-event queue. When the queue is full, the oldest event is evicted to make room for the new one. The subscriber is never silently dropped; it continues receiving events at the cost of missing stale ones.

**Timestamp format:** All ISO-8601 timestamps emitted by the server (REST responses and WebSocket events) include an explicit UTC timezone suffix (`+00:00`). Incoming timestamps in request bodies are accepted with or without timezone info and normalized to timezone-aware UTC via `ts_utc_aware_get()`. All internal comparisons, filters, and storage operations use aware UTC. Legacy naive timestamps in existing Parquet files are treated as UTC and normalized to aware UTC on read.

## Privacy

The UI never displays raw window titles, keystrokes, or URLs.
Only aggregated metrics and application identifiers are shown.
Transition notifications (web and desktop fallback) redact app names by default
(`privacy_notifications=True`); set to `False` to show raw app identifiers.
Desktop fallback notifications can be disabled entirely with `notifications_enabled=False`.

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
- **In-process web UI server** -- the FastAPI server always runs in-process, sharing the tray's `EventBus`. In `--browser` mode the dashboard opens in the default browser; otherwise a lightweight pywebview subprocess provides a native floating window. Both modes receive the same live events (status, tray_state, suggest_label, prediction) because the server and the tray publish/subscribe on the same `EventBus` instance. To keep cold starts responsive, the HTTP server is started before optional model loading finishes, so the dashboard can appear while suggestions are still warming up. Parquet/`pandas` I/O is loaded lazily inside label, feature-summary, and training-related handlers so the initial import path stays lighter than the full data stack.
- **Activity transition detection** -- polls ActivityWatch and detects when the dominant foreground app changes. A transition fires when the new app persists for >= `--transition-minutes` (default 3). On the first poll, an `initial_app` event is published so the UI can prompt labeling for the pre-start period.
- **Pause/resume** -- monitoring can be paused via the tray menu ("Pause"/"Resume") or the `POST /api/tray/pause` REST endpoint. When paused, polling and transition detection are skipped but session state (poll count, transitions) is preserved. The `status` event emits `state: "paused"` and the `tray_state` event includes `paused: true`.
- **Transition notifications** -- on each transition, a notification prompts the user to label the completed block. The primary channel is the **Web Notifications API** (delivered via the `prompt_label` WebSocket event to the frontend); macOS osascript desktop notifications serve as a fallback when no browser or pywebview client is connected. The browser requests notification permission on the first user interaction (click). By default, app names are redacted for privacy (`privacy_notifications=True`). Set `privacy_notifications=False` to show raw app identifiers. Desktop fallback notifications can be disabled entirely with `notifications_enabled=False`.
- **Label suggestions** -- when `--model-dir` is provided, the app predicts a label and includes it in the notification. Without a model, all 8 core labels are shown. On startup, model loading is deferred until after the embedded UI server begins listening, so `tray_state.model_loaded` may briefly remain `false` during cold-start while the suggester loads in the background. `_LabelSuggester` propagates the stable config-backed `user_id` (from `UserConfig`) to `build_features_from_aw_events` so the model receives the same personalization signal used during training. When no config is available, falls back to `"default-user"`. Input events (keyboard/mouse statistics from `aw-watcher-input`) are also fetched and passed to feature building so that input-derived features (`keys_per_min`, `clicks_per_min`, etc.) are populated rather than left as `None`.
- **Quick-label menus** -- right-click the tray icon to label the last 5/10/15/30 minutes with any core label.
- **Label Stats** -- tray menu action that shows a desktop notification summarizing today's labeling progress: total label count, total time, and per-label breakdown (e.g. "Today: 5 labels, 1h 15m -- Build 45m, Debug 20m, Write 10m"). Also available via `GET /api/labels/stats` for programmatic access.
- **Import Labels** -- tray menu action that imports label spans from a CSV file. Opens a file-open dialog (via tkinter) to choose the source CSV, then prompts the user to merge with existing labels or overwrite them. Falls back to native macOS file dialogs (via osascript) when tkinter is unavailable or fails (e.g. threading conflicts with the pystray event loop). Merge deduplicates by `(start_ts, end_ts, user_id)` and rejects overlapping spans; overwrite replaces all labels. Also available via `POST /api/labels/import` for programmatic access.
- **Export Labels** -- tray menu action that exports all label spans to a CSV file. Opens a save-file dialog (via tkinter) to choose the destination; falls back to `<data_dir>/labels_v1/labels_export.csv` when tkinter is unavailable. Also available via `GET /api/labels/export` for programmatic access.
- **Status** -- tray menu action that shows a desktop notification with connection and session status: ActivityWatch connection state, poll count, transition count, saved labels count, and loaded model name.
- **Open Data Folder** -- tray menu action that opens the data directory in the OS file manager (Finder on macOS, `xdg-open` on Linux). Falls back to a notification showing the path if the file manager cannot be launched.
- **Edit Config** -- tray menu action that opens `config.toml` in the default text editor. All runtime settings (`notifications_enabled`, `privacy_notifications`, `poll_seconds`, `transition_minutes`, `aw_host`, `title_salt`, `ui_port`) are persisted to this TOML file on startup with `#` comments describing each setting. Edited values take effect on next restart; explicit CLI flags always override config file values. Existing `config.json` files are auto-migrated to TOML on first startup.
- **Report Issue** -- tray menu action that opens the GitHub issue tracker (`https://github.com/fruitiecutiepie/taskclf/issues/new`) in the default browser, pre-filled with the `bug_report.yml` template and version/OS diagnostics as query parameters. The user controls what information is submitted; no data is sent automatically.
- **Model submenu** -- a "Model" submenu listing all valid model bundles found in `--models-dir` (default `models/`). The submenu auto-refreshes on every menu open by re-scanning the models directory, so new bundles created by retraining appear immediately without restarting the tray. The currently loaded model shows a check mark (radio-button effect). Clicking a different bundle hot-swaps the in-memory suggester without restarting. A "(No Model)" entry unloads the model entirely. The submenu also contains "Reload Model" to re-read the current bundle from disk and "Check Retrain" to run the retrain eligibility check. When `--models-dir` is missing or empty, a disabled "(no models found)" placeholder is shown.
- **Check Retrain** -- tray menu action (inside the Model submenu) that checks whether retraining is due based on the latest model's age and the configured cadence. Shows a notification with "Retrain recommended" (with model name and creation date) or "Model is current". Uses `--retrain-config` to load a custom retrain YAML config; falls back to defaults when not provided. Disabled when `--models-dir` is not set.
- **Open Web UI** -- menu option to open the labeling web UI in the browser.
- **Shell UI parity** -- The compact badge route in a normal browser tab uses a light solid page background and in-page label grid / state panel (hover and pin) because `Host.invoke` is a no-op without a host bridge. The `?view=label` and `?view=panel` routes match native popup markup. Use the pywebview floating window, `taskclf electron`, or Electron dev with the sidecar backend for full multi-window behavior.
- **Gap-fill surface** -- tracks unlabeled time since the last confirmed label and publishes `unlabeled_time` events every poll cycle (passive badge). Active `gap_fill_prompt` events are published only at idle return (>5 min), session start, or immediately after accepting a transition suggestion. When unlabeled time exceeds `gap_fill_escalation_minutes` (default 480), the tray icon changes to orange (`gap_fill_escalated` event). No popup or notification is sent on escalation.
- **Event broadcasting** -- publishes `status`, `tray_state`, `initial_app`, `prediction`, `label_created`, `suggest_label`, `unlabeled_time`, `gap_fill_prompt`, and `gap_fill_escalated` events to the shared EventBus for connected WebSocket clients.
- **Frontend log channel** -- in frontend dev mode (`--dev`), debug/error messages emitted in the SolidJS app can be forwarded through `window.pywebview.api.frontend_debug_log(...)` and `window.pywebview.api.frontend_error_log(...)`. Debug lines are written at DEBUG level (requires DEBUG logging enabled, e.g. global `--verbose`), while error lines are written at ERROR level.
  The app also installs global `window.onerror` and `unhandledrejection` handlers in dev mode so uncaught frontend failures are captured and forwarded via the same error channel.
- **Crash handler** -- `TrayLabeler.run()` wraps the main loop in a top-level `try/except`. On unhandled exceptions, a crash report is written to `<TASKCLF_HOME>/logs/crash_<YYYYMMDD_HHMMSS>.txt` and a desktop notification is attempted with the crash file path. See [core.crash](../core/crash.md) for details.

## Surface Architecture

The tray implements three distinct UI surfaces with separate code paths,
interaction patterns, and confidence profiles (Decision 6):

| Surface | Method | Event type | Copy function | Trigger |
|---|---|---|---|---|
| Transition suggestion | `_handle_transition` | `prompt_label` | `transition_suggestion_text` | App transition |
| Live status | `_publish_live_status` | `live_status` | `live_status_text` | Every poll cycle |
| Gap-fill indicator | `_publish_unlabeled_time` | `unlabeled_time` | `gap_fill_prompt` | Every poll cycle (passive) |
| Gap-fill prompt | `_publish_gap_fill_prompt` | `gap_fill_prompt` | `gap_fill_prompt` | Idle return / session start / post-acceptance |
| Gap-fill escalation | `_check_escalation` | `gap_fill_escalated` | — | Unlabeled time exceeds threshold |

- **Transition suggestions** aggregate all buckets in the completed interval
  via `infer.aggregation.aggregate_interval` and display an action-oriented
  prompt with a concrete time range (e.g. "Was this Coding? 12:00–12:47").
- **Live status** predicts only the current single bucket and publishes a
  passive present-tense label ("Now: Coding").
- **Gap-fill indicator** is a passive badge showing total unlabeled time
  since the last confirmed label. Published every poll cycle as an
  `unlabeled_time` event. Does not interrupt the user.
- **Gap-fill prompt** is an active prompt published only at three defined
  trigger points: idle return (>5 min idle), session start, or immediately
  after the user accepts a transition suggestion.
- **Gap-fill escalation** fires when unlabeled time exceeds a configurable
  threshold (`gap_fill_escalation_minutes`, default 480 = one active day).
  Changes the tray icon color to orange. No popup or notification is sent.
- Numeric confidence is never shown to the user on transition or live status surfaces.
- All user-facing copy strings are centralized in
  [`ui.copy`](copy.md).

## Gap-fill events

Three new WebSocket event types support the gap-fill surface:

- `unlabeled_time` — published every poll cycle when unlabeled time exists:
  `unlabeled_minutes`, `text` (human-readable badge text), `last_label_end`, `ts`.
- `gap_fill_prompt` — published at idle return, session start, or post-acceptance:
  `trigger` (`"idle_return"`, `"session_start"`, or `"post_acceptance"`),
  `unlabeled_minutes`, `text`, `last_label_end`, `ts`.
- `gap_fill_escalated` — published when unlabeled time exceeds the escalation
  threshold: `unlabeled_minutes`, `threshold_minutes`.

The `gap_fill_escalation_minutes` setting (default 480) controls when
escalation fires. It is a user-facing config unlike the reject threshold.

## Privacy

Same guarantees as the web UI: no raw window titles or keystrokes are displayed or stored.
Desktop notifications redact app names by default.

::: taskclf.ui.tray
