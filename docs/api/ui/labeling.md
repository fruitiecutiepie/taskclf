# ui.server

Web UI for ground-truth label collection and live prediction monitoring.
The shared runtime classes (`ActivityMonitor`, `_LabelSuggester`) live in
`taskclf.ui.runtime`, while tray-specific orchestration stays in
`taskclf.ui.tray.TrayLabeler`.

## Launch

```bash
taskclf ui
taskclf ui --port 8741 --model-dir models/run_20260226
```

For frontend development with hot reload:

```bash
taskclf ui --dev
```

For browser-based full-stack development with frontend HMR plus backend
auto-reload:

```bash
taskclf ui --dev --browser
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
| `--dev` | off | Start Vite dev server for frontend hot reload; in browser mode, the FastAPI backend also runs with auto-reload. Uses an ephemeral data dir unless `--data-dir` is set |

## Panels

- **Label** -- Form with date/time pickers, `CoreLabel` dropdown, confidence slider, and user ID input.
- **Recent** -- Quick-label with preset durations (now / 1 / 5 / 10 / 15 / 30 / 60 min) or a custom duration input supporting seconds, minutes, hours, and days. "now" labels the span from the last label's `end_ts` to the current moment (falling back to 1 minute if no last label exists or the span would be < 1 minute); the button dynamically shows the duration (e.g. "now (12m)"). Other values label the corresponding trailing window. The "Extend until next label" checkbox (on by default) sets `extend_forward=true` on the new label; when the *next* label is created, this label's `end_ts` is automatically stretched to meet the next label's `start_ts`, producing contiguous coverage without gaps. For a zero-duration "now" label with `extend_forward=true`, the server performs a same-timestamp handoff by ending the currently active same-user label at `now`, then creating the new open-ended label at `now`, so no overlap dialog is needed in the common switch-label-now flow. The History view immediately renders the new span as open-ended (`start – Now`, duration `until next label`) so users get instant confirmation that the label was recorded before the next label closes the span. Shows a live ActivityWatch summary when available. The footer switches between a compact "Last: *Label* Nm ago" summary for completed labels and "Current: *Label* since Nm ago" for any label whose `extend_forward` coverage is still active.
  Quick-label keeps those two notions separate: `GET /api/labels/current` detects the active `extend_forward` label that drives the footer and **Stop current label** control, while `GET /api/labels?limit=1` still returns the span with the **latest `end_ts`** first for gap-fill and "last ended" behavior, including when overlapping spans exist (`allow_overlap`). That keeps the stop action visible for running labels even if some completed overlapping span ends later.
  When a current label is active, the footer offers a two-step **Stop current label** action that closes the running span at the moment the user confirms, without deleting the label that has already been recorded. This applies both to zero-duration "from now" labels and to earlier backfilled spans that were saved with `extend_forward=true` and are still the active label. On success, the compact badge immediately drops the stale manual label and falls back to the latest passive live-status label from the model. The gap shortcut is hidden while a current `extend_forward` label is active so the quick-label surface does not imply there is unlabeled time to backfill. When the last **completed** label is shown, the **gap** button (e.g. `gap 5m`) reflects unlabeled time since that label’s `end_ts` and refreshes on a ~30s wall-clock tick so the duration stays current while the window stays open. After a successful label, a brief "Saved" flash appears (click to dismiss instantly); the grid stays open for rapid consecutive labeling. Non-overlap quick-label failures now render a persistent inline error banner with **Copy error** and **Close** actions instead of auto-dismissing after a timeout. On overlap errors, instead of a dead-end flash the grid shows a compact confirmation prompt listing the conflicting label names (e.g. "Overlaps 3 labels: Write, Debug, Review") with the affected time range shown below (e.g. "(10:50–11:20 will be replaced)"), plus Overwrite/Keep All/Cancel buttons. For a single conflict the per-span time range is shown inline; for multiple conflicts a "show details" toggle reveals per-span times. Choosing Overwrite re-submits the label with `overwrite: true`, which truncates, splits, or removes the conflicting existing span(s) to make room for the new label. Choosing Keep All re-submits with `allow_overlap: true`, preserving all existing and new labels on the overlapping time range. If `lastLabel` changes while the prompt is visible (e.g. from another tab or WebSocket), the prompt is automatically dismissed since the conflict data may be stale.
- **Queue** -- Pending `LabelRequest` items sorted by confidence (lowest first). Shows time range, predicted label, confidence, and reason.

## Live Features

- **Live badge (compact)** -- Header pill showing the current label/app and connection dot. Visible in the collapsed tray window. When a `suggest_label` event arrives, the compact badge immediately switches to the suggested label as a frontend-only display override. If the suggestion is skipped, the pill restores the pre-suggestion display; if the suggestion is accepted, the pill keeps the accepted suggestion until a fresher `prediction` or `live_status` update supersedes it.
- **State panel** -- Tabbed panel with three views, selected via a segmented control at the top:
  - **System tab** -- Internal-state debug panel with a collapsible accordion layout: each section header shows an inline summary badge (e.g., current app, predicted label, connection status) so all states are scannable at a glance. Click any header to expand its detail rows. **Activity Monitor** and **Last Prediction** default to open; all other sections start collapsed. Eight sections:
    - **Activity Monitor** -- summary: `state · current_app`. Details: `state`, `current_app`, `since`, `poll_interval`, `poll_count`, `last_poll` timestamp, `uptime`. When a transition candidate exists: `candidate_app`, `candidate_progress` with duration/threshold/percentage and a visual progress bar.
    - **Last Prediction** -- summary: `mapped_label confidence%`. Details: `label`, `mapped_label`, `confidence` (color-coded green/red at 50% threshold), `ts`, `trigger_app`.
    - **Model** -- summary: `loaded`/`not loaded` (color-coded). Details: `loaded` status, `model_dir`, `schema_hash`, `suggested` label, `suggestion_conf`. When a model is loaded and the tray exposes `model_dir`, the panel also shows **bundle-saved** validation metrics (`val_macro_f1`, `val_weighted_f1`, training range, top confusion pairs) from `GET /api/train/models/current/inspect` — static metrics on disk, not a live replay of the test split.
    - **Transitions** -- summary: transition count. Details: `total` count, last transition details: `prev → new` apps, `block` time range, `fired_at` timestamp.
    - **Active Suggestion** -- appears when the model suggests a label on transition. Summary: `suggested confidence%`. Details: `suggested`, `confidence`, `reason`, `old_label`, `block` time range.
    - **Activity Source** -- summary: provider `state` (`checking`, `ready`, `setup required`) with provider-neutral diagnostics. Details: provider name, endpoint, resolved `source_id`, `last_sample_count`, last-sample app distribution, and a setup callout when the configured source is unavailable. The setup callout is intentionally non-blocking: manual label creation, editing, and suggestion accept/skip flows remain enabled even when summaries are unavailable.
    - **WebSocket** -- summary: connection status (color-coded). Details: `status`, `messages` total, per-type `breakdown` (st/pred/tray/sug), `last_received` timestamp, `reconnects` count, `connected_since`.
    - **Config** -- summary: `dev`/`prod`. Details: `data_dir`, `ui_port`, `dev_mode`, `labels_saved` count.
  - **Training tab** -- Model training interface with data readiness checks, a training form (date range, boost rounds, class weight, synthetic toggle), real-time progress via WebSocket, result display (macro/weighted F1), and a model bundle list. Each bundle row can expand **Bundle metrics** to lazy-load the same bundle-saved validation inspection as `GET /api/train/models/{model_id}/inspect`. Validation/start/cancel failures and failed-run messages use the same persistent inline error banner with **Copy error** and **Close** controls. See [training.md](training.md) for endpoint details.
  - **History tab** -- Single-day view with date navigation (prev/next arrows and a date picker). Shows a full-day timeline strip (00:00–23:59) with color-coded labeled segments and clickable unlabeled gaps. Click any label row to expand an inline editor showing the ActivityContext for that time window (apps used, input stats), a label-change grid (current label highlighted), and a delete button with confirmation. Label time edits are minute-based and use end-exclusive semantics (`15:00-15:01` means one minute); equal start/end values (`15:00-15:00`) are rejected. Row display stays compact at `HH:MM`, but includes seconds when either boundary has non-zero seconds. Click any unlabeled gap row to expand an inline editor with time range inputs (pre-filled to the gap boundaries, adjustable to label a sub-range), ActivityContext for the selected range, and a label picker grid; selecting a label creates a new label span for the chosen sub-range. Navigating to a past date with no labels shows the entire day as a single labelable gap. The active date refetches immediately on `labels_changed` WebSocket events, so manual labels, accepted suggestions, edits, deletes, and imports appear without requiring a tab switch. Provides a dedicated review surface separate from the quick-label popup, with the full panel height available for browsing.
- **Suggestion banner** -- When present, it is shown at the **top** of the label column (above the duration/time picker and the rolling activity summary) so the model prompt is visible first. It appears when a new inferred label arrives from `suggest_label`. The banner shows the current label, suggested label, confidence, and the applicable block time range (`block_start` -> `block_end`), using a compact local-time display and adding the date when the suggested range crosses midnight locally. Below that, an inline **activity summary** for that same range reuses the same surface as the main label panel (`ActivitySummary` via `GET /api/activity/summary`): top apps, input-rate hints when available, and bucket/session coverage. If the configured activity source is unavailable, the banner shows the same non-blocking setup callout used elsewhere; if the range is simply empty, it shows `No activity data for this window`. Label names use the same color cues as the label grid. Clicking `Use suggestion` immediately writes the suggested label to label history. When that suggestion lands inside the effective coverage of the current `extend_forward` label, the backend automatically splits the running label into a before-fragment and a resumed open-ended fragment after the suggestion so the active label continues past the suggested block. Other same-user overlaps still use the same **Overwrite All** / **Keep All** / **Cancel** prompt as quick-labeling; retries use `POST /api/notification/accept` with `overwrite` or `allow_overlap` so the span stays `provenance="suggestion"`. `Skip` calls the notification-skip flow without creating labels. Manual quick-label saves (`POST /api/labels`) do **not** dismiss the banner; it clears when the suggestion is accepted, skipped, or cleared by tray-side auto-save paths. The compact badge mirrors the suggestion immediately while the banner is present, reverts to its pre-suggestion display on skip, and stays on the accepted suggestion until a fresher explicit badge signal arrives. Optional client auto-dismiss is configured by `suggestion_banner_ttl_seconds` in `config.toml` (see [core/config.md](../core/config.md) and `GET /api/config/user`); `0` disables the timer. Save/skip failures surface as persistent inline errors with **Copy error** and **Close** actions rather than disappearing on a timer.
- **Auto-save BreakIdle** -- When a completed activity block is detected as idle (lockscreen app was dominant, or the model suggested `BreakIdle`), the tray auto-saves the label with `provenance="auto_idle"` and publishes a `label_created` event. No user confirmation is required since the user was away. Lockscreen/idle transitions use a separate, faster threshold (`idle_transition_minutes`, default 1 min) so breaks are detected quickly without affecting the general transition cadence (`transition_minutes`, default 2 min). The fast threshold applies in both directions: when the lockscreen app becomes dominant, and when the user returns from lockscreen. Matching uses normalized app IDs: `com.apple.loginwindow` (macOS), `com.microsoft.LockApp`/`com.microsoft.LogonUI` (Windows), `org.gnome.ScreenSaver`, `org.gnome.Shell`, `org.i3wm.i3lock`, `org.swaywm.swaylock`, `org.jwz.xscreensaver`, `org.freedesktop.light-locker`, `org.suckless.slock` (Linux).

## Architecture

The UI is a SolidJS single-page application served by a FastAPI backend:

- **REST endpoints** (`/api/labels`, `/api/labels/export`, `/api/labels/import`, `/api/labels/stats`, `/api/queue`, `/api/features/summary`, `/api/activity/summary`, `/api/aw/live`, `/api/config/labels`, `/api/config/user`, `/api/tray/pause`, `/api/tray/state`, `/api/train/*`) handle label CRUD, import/export, stats, queue management, user configuration, tray control, model training, and data queries. See [training.md](training.md) for the `/api/train/*` endpoints (including bundle-only model inspection under `/api/train/models/.../inspect`).
  - `GET /api/labels` accepts optional `limit` (default 50, max 500), `date` (ISO-8601 date string, e.g. `2025-03-07`), and optional `range_start` / `range_end` (ISO-8601 UTC bounds) query parameters. After filters are applied, results are sorted by **`end_ts` descending** (latest-ended first). When `date` is provided, only labels overlapping that day are returned. The History tab uses this to fetch labels for the selected date.
  - `GET /api/labels/current` returns the most recently started label whose `extend_forward` coverage still contains "now", or `null` when none exists. That includes zero-duration "from now" labels (`start_ts == end_ts`) and earlier spans saved with `extend_forward=true` when no later same-user label has ended that coverage yet. Quick-label uses this endpoint for the footer's current badge and stop action, so the active label remains discoverable even when `GET /api/labels` is ordered by latest `end_ts`.
  - `POST /api/labels` accepts optional `extend_forward`, `overwrite`, and `allow_overlap` booleans. `extend_forward` persists the label with `extend_forward=true`; when the *next* label is created for the same user, this label's `end_ts` is automatically stretched to the new label's `start_ts`, producing contiguous coverage. For zero-duration `extend_forward` labels (`start_ts == end_ts`, used by "label from now"), the server first truncates the currently active same-user span at that same timestamp, then appends the new label, ensuring a contiguous no-overlap handoff. The quick-label UI sets this flag by default. Before overlap checks, same-user boundaries within 1 ms are snapped together so timestamps that passed through JavaScript `Date` do not fail with false microsecond overlaps. When `overwrite` is `true`, conflicting same-user spans are truncated, split, or removed to make room for the new label (no 409 is returned). When `allow_overlap` is `true`, the overlap check is skipped entirely and multiple labels are allowed to coexist on the same time range; this is useful for multi-task periods. When both are `false` (default), an overlap returns 409 with structured conflict details: `{"detail": {"error": "...", "conflicting_start_ts": "...", "conflicting_end_ts": "..."}}` so the frontend can prompt the user to overwrite or keep all.
  - `PUT /api/labels` changes the label on an existing span identified by `start_ts` + `end_ts`. Optional `new_start_ts`, `new_end_ts`, and `extend_forward` fields can also change the time range and running/current state. The quick-label "Stop current label" action uses `new_end_ts=<click time>` with `extend_forward=false` to close the active label without deleting it. Returns 404 if no matching span exists.
  - `DELETE /api/labels` removes a span identified by `start_ts` + `end_ts`. Returns 404 if no matching span exists.
  - `GET /api/labels/export` downloads all label spans as a CSV file (`text/csv`). Returns 404 if no labels file exists or the file contains no spans.
  - `POST /api/labels/import` accepts a multipart CSV file upload (`file`) and an optional `strategy` form field (`"merge"` or `"overwrite"`, default `"merge"`). In merge mode, imported spans are deduplicated against existing labels by `(start_ts, end_ts, user_id)` and overlap-checked; conflicts return 409. In overwrite mode, all existing labels are replaced. Returns `{"status": "ok", "imported": N, "total": M, "strategy": "merge"|"overwrite"}`. Returns 422 on invalid CSV or strategy.
  - `GET /api/labels/stats` returns labeling statistics for a given day. Accepts an optional `date` query parameter (ISO-8601 date string, defaults to today UTC). Returns `{"date": "2026-03-01", "count": 5, "total_minutes": 75.0, "breakdown": {"Build": 45.0, "Meet": 20.0, "Write": 10.0}}`.
  - `GET /api/activity/summary` is the frontend-facing activity summary endpoint. Query params: `start`, `end` (ISO-8601). It returns the usual aggregate stats plus `activity_provider`, `recent_apps`, `range_state`, and `message`. `range_state="ok"` means summary data exists, `range_state="no_data"` means the source is reachable but the requested window is empty, and `range_state="provider_unavailable"` means the configured source is not ready and the response includes setup guidance. Manual labeling remains available regardless of `range_state`.
  - `POST /api/tray/pause` toggles the monitoring pause state. Returns `{"status": "ok", "paused": true/false}` when connected to a tray, or `{"status": "unavailable", "paused": false}` when no tray callbacks are configured.
  - `GET /api/tray/state` returns tray availability and pause state. When the tray backend provides `get_tray_state`, the payload may also include `model_dir` and `models_dir` (each a string path or `null`). `models_dir` is set when `--models-dir` is configured so clients (e.g. Electron) can enable **Advanced → Edit Inference Policy**.
  - `POST /api/notification/accept` confirms an inferred label suggestion and writes it to `labels.parquet` with `provenance="suggestion"`. Required body: `{"block_start": "...", "block_end": "...", "label": "..."}`. Optional `overwrite` and `allow_overlap` booleans match `POST /api/labels`. When the suggested block falls inside the effective coverage of the current same-user `extend_forward` label, the server automatically uses overwrite-style splitting so the prior label resumes after the accepted suggestion. Other overlaps still return 409 with the same structured conflict details as manual labeling, and the UI can retry with `overwrite=true` or `allow_overlap=true` without changing provenance.
  - `POST /api/notification/skip` dismisses the current suggestion without saving a label and broadcasts `suggestion_cleared` with reason `"skipped"` so all connected clients clear the prompt.
- **WebSocket** (`/ws/predictions`) streams live events from the ActivityMonitor:
  - `status` -- every poll cycle: `state` (`"collecting"` or `"paused"`), `current_app`, `current_app_since`, `candidate_app`, `candidate_duration_s`, `transition_threshold_s`, `poll_seconds`, `poll_count`, `last_poll_ts`, `uptime_s`, and nested `activity_provider` (`provider_id`, `provider_name`, `state`, `summary_available`, `endpoint`, `source_id`, `last_sample_count`, `last_sample_breakdown`, `setup_title`, `setup_message`, `setup_steps`, `help_url`). Legacy `aw_connected`, `aw_bucket_id`, `aw_host`, `last_event_count`, and `last_app_counts` are still emitted temporarily as compatibility aliases. When monitoring is paused, `state` is `"paused"` and polling/transition detection is skipped.
  - `tray_state` -- every poll cycle: `model_loaded`, `model_dir`, `model_schema_hash`, `suggested_label`, `suggested_confidence`, `transition_count`, `last_transition` (with `prev_app`, `new_app`, `block_start`, `block_end`, `fired_at`), `labels_saved_count`, `data_dir`, `ui_port`, `dev_mode`, `paused`.
  - `initial_app` -- once on startup when the first dominant app is detected: `app`, `ts`. Allows the UI to prompt the user to label the pre-start period that would otherwise be unlabeled.
  - `prediction` -- on app transition with model suggestion: `label`, `confidence`, `ts`, `mapped_label`, `current_app`. Reserved for actual model outputs; manual labels no longer use this event type.
  - `no_model_transition` -- on app transition without a loaded model: `current_app`, `ts`, `block_start`, `block_end`. The frontend uses this to distinguish "no model loaded" from "model predicted unknown"; `LiveBadge` shows "No Model" instead of "Unknown Label" when `trayState.model_loaded === false`.
  - `label_created` -- when a label with `extend_forward=true` is created via `POST /api/labels`, or when the tray auto-saves a `BreakIdle` label (lockscreen/idle detection): `label`, `confidence`, `ts` (end), `start_ts`, `extend_forward`. The frontend maps this to a `Prediction`-compatible object so `LiveBadge` and `StatePanel` update automatically.
  - `label_stopped` -- when an open-ended running label is closed via `PUT /api/labels`: `ts` (the new end timestamp). Clients clear any stale manual badge prediction at or before that timestamp, then fall back to the latest `live_status` label if available.
  - `labels_changed` -- on any label-history mutation that should invalidate day views: `reason`, `ts`. Published after `POST /api/labels`, `PUT /api/labels`, `DELETE /api/labels`, `POST /api/labels/import`, `POST /api/notification/accept`, and tray-side auto-saved `BreakIdle` labels.
  - `suggestion_cleared` -- published after successful suggestion acceptance (`POST /api/notification/accept`), dismissals (`POST /api/notification/skip`), and auto-saved BreakIdle labels: `reason` (e.g. `"label_saved"`, `"skipped"`, `"auto_saved_breakidle"`). Manual `POST /api/labels` saves do **not** publish this event. Clients clear the active suggestion on receipt. The compact badge treats `"skipped"` as a restore-to-previous-display signal; accepted/auto-saved clears keep the assumed suggestion visible until a fresher `prediction` or `live_status` event arrives. The frontend may also auto-dismiss after `suggestion_banner_ttl_seconds` from `config.toml` (via `GET /api/config/user`); `0` disables that timer.
  - `suggest_label` -- on app transition with model suggestion: `suggested`, `confidence`, `reason`, `old_label`, `block_start`, `block_end`.
  - `prompt_label` -- on task transition with labeling prompt: `prev_app`, `new_app`, `block_start`, `block_end`, `duration_min`, `suggested_label`, `suggestion_text`. The structured `block_start` / `block_end` bounds remain UTC; the human-readable `suggestion_text` range is rendered in the user's local timezone for notification display. Frontend notification surfaces may also derive a second exact local-time range line directly from `block_start` / `block_end` for higher-precision display.
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
- **In-process web UI server** -- the FastAPI server always runs in-process, sharing the tray's `EventBus`. In `--browser` mode the dashboard opens in the default browser; otherwise a lightweight pywebview subprocess provides a native floating window. Both modes receive the same live events (`status`, `tray_state`, `prompt_label`, `suggest_label`, `prediction`) because the server and the tray publish/subscribe on the same `EventBus` instance. To keep cold starts responsive, the HTTP server is started before optional model loading finishes, so the dashboard can appear while suggestions are still warming up. Browser-only startup imports the shared `taskclf.ui.runtime` helpers instead of the tray icon module, so it does not pull Pillow/pystray into that import path. Parquet/`pandas` I/O is loaded lazily inside label, feature-summary, and training-related handlers so the initial import path stays lighter than the full data stack.
- **Activity transition detection** -- polls ActivityWatch and detects when the dominant foreground app changes. A transition fires when the new app persists for >= `--transition-minutes` (default matches `DEFAULT_TRANSITION_MINUTES`, currently 2 minutes). On the first poll, an `initial_app` event is published so the UI can prompt labeling for the pre-start period.
- **Pause/resume** -- monitoring can be paused via the tray menu ("Pause"/"Resume") or the `POST /api/tray/pause` REST endpoint. When paused, polling and transition detection are skipped but session state (poll count, transitions) is preserved. The `status` event emits `state: "paused"` and the `tray_state` event includes `paused: true`.
- **Transition notifications** -- on each transition, a notification prompts the user to label the completed block. Plain browser mode uses the **Web Notifications API** (driven by the `prompt_label` WebSocket event and requested when the route mounts). Native shells stay on native notification paths: Electron shows an OS notification, and the legacy pywebview shell forwards the same `prompt_label` event through `WindowAPI.show_transition_notification()` so suggestions still raise a desktop notification even when the embedded webview does not expose the browser Notification API. User-facing time ranges in notification copy are displayed in the local timezone, while the structured event timestamps remain UTC. Notification bodies include an exact local start/end range line derived from `block_start` / `block_end` for precise review. Standalone label/panel windows reuse the same prompt event path as the compact shell, and the frontend de-dupes each transition by block range so multiple subscribed windows do not alert twice for the same prompt. On supported Web Notifications runtimes, transition prompts request persistent display (`requireInteraction`) so they stay visible until dismissed or clicked instead of auto-closing immediately. In the Electron shell, transition prompts use native action buttons on supported platforms: when a model suggestion exists the notification offers `Accept`, `Review`, and `Skip`, while clicking the notification body also opens the labeler; without a suggestion, the native action is `Review`. On macOS the shell falls back to a plain clickable notification by default because action buttons require the packaged app to meet stricter platform conditions. The pywebview native path uses privacy-safe copy (`suggestion_text` when present, otherwise `Activity changed`) plus the exact local range line; it does not expose raw app names. By default, app names are redacted for privacy (`privacy_notifications=True`). Set `privacy_notifications=False` to show raw app identifiers. Desktop fallback notifications can be disabled entirely with `notifications_enabled=False`.
- **Label suggestions** -- when `--model-dir` is provided, the app predicts a label and includes it in the notification. Without a model, all 8 core labels are shown. On startup, model loading is deferred until after the embedded UI server begins listening, so `tray_state.model_loaded` may briefly remain `false` during cold-start while the suggester loads in the background. The shared `_LabelSuggester` runtime helper propagates the stable config-backed `user_id` (from `UserConfig`) to `build_features_from_aw_events` so the model receives the same personalization signal used during training. When no config is available, falls back to `"default-user"`. Input events (keyboard/mouse statistics from `aw-watcher-input`) are also fetched and passed to feature building so that input-derived features (`keys_per_min`, `clicks_per_min`, etc.) are populated rather than left as `None`.
- **Quick labeling** -- preset durations and the label grid live in the web UI (for example the **Recent** panel), not in the pystray right-click menu. Use **Toggle Dashboard** to open the UI.
- **Today's Labels** -- tray menu action that shows a desktop notification summarizing today's labeling progress: total label count, total time, and per-label breakdown (e.g. "Today: 5 labels, 1h 35m -- Build 1h 5m, Debug 20m, Write 10m"). Counts use the UTC calendar day. Also available via `GET /api/labels/stats` for programmatic access.
- **Import Labels** -- tray menu action that imports label spans from a CSV file. Opens a file-open dialog (via tkinter) to choose the source CSV, then prompts the user to merge with existing labels or overwrite them. Falls back to native macOS file dialogs (via osascript) when tkinter is unavailable or fails (e.g. threading conflicts with the pystray event loop). Merge deduplicates by `(start_ts, end_ts, user_id)` and rejects overlapping spans; overwrite replaces all labels. Also available via `POST /api/labels/import` for programmatic access.
- **Export Labels** -- tray menu action that exports all label spans to a CSV file. Opens a save-file dialog (via tkinter) to choose the destination; falls back to `<data_dir>/labels_v1/labels_export.csv` when tkinter is unavailable. Also available via `GET /api/labels/export` for programmatic access.
- **Show Status** -- tray menu action that shows a desktop notification with connection and session status: ActivityWatch connection state, poll count, transition count, saved labels count, and loaded model name.
- **Open Data Folder** -- tray menu action that opens the data directory in the OS file manager (Finder on macOS, `xdg-open` on Linux). Falls back to a notification showing the path if the file manager cannot be launched.
- **Edit Config** -- tray menu action that opens `config.toml` in the default text editor. On first run, if the file is missing, taskclf writes a **full commented starter template** once (all supported keys); existing files are not regenerated on later startups. Resolved runtime settings are **not** rewritten on every launch; values from this file are read at startup, and explicit non-default CLI flags override file values for that run. Changes from the web UI (for example username or suggestion banner TTL via `/api/config/user`) merge into `config.toml`. See [User config template](../../guide/config_template.md) and [`configs/user_config.template.toml`](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml). Existing `config.json` files are auto-migrated to TOML when `config.toml` is absent.
- **Edit Inference Policy** -- under the tray **Advanced** submenu; opens `models/inference_policy.json` in the default text editor (disabled when `--models-dir` is not set). If the file is missing, it is created first only when the currently loaded/resolved model bundle can seed it, reusing `metadata.json`'s advisory `reject_threshold` and auto-attaching a matching `artifacts/calibrator_store` when its `store.json` is explicitly bound to that model. If no model can be resolved, the tray does **not** write a placeholder file; instead it notifies you to use **Prediction Model** or **Open Data Folder** (the `models/` folder sits next to your data directory), and mentions the optional CLI command `taskclf policy create --model-dir models/<run_id>` for users who have the CLI installed. The canonical starter shape still lives in [`configs/inference_policy.template.json`](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/inference_policy.template.json); see [Inference policy template](../../guide/inference_policy_template.md). Unlike `config.toml`, this file is **not** auto-created on first run—only when you use this action with a resolvable model or create it via the CLI. Invalid hand-edits may cause inference to fall back to `active.json` resolution until the file parses again.
- **Report Issue** -- tray menu action that opens the GitHub issue tracker (`https://github.com/fruitiecutiepie/taskclf/issues/new`) in the default browser, pre-filled with the `bug_report.yml` template and version/OS diagnostics as query parameters. The user controls what information is submitted; no data is sent automatically.
- **Prediction Model submenu** -- a **Prediction Model** submenu listing all valid model bundles found in `--models-dir` (default `models/`). The submenu auto-refreshes on every menu open by re-scanning the models directory, so new bundles created by retraining appear immediately without restarting the tray. The currently loaded model shows a check mark (radio-button effect). Clicking a different bundle hot-swaps the in-memory suggester without restarting. A **No Model** entry unloads the model entirely. The submenu also contains **Refresh Model** (reloads via inference policy when configured, otherwise re-reads the active bundle from disk) and **Retrain Status** to run the retrain eligibility check. When `--models-dir` is missing or empty, a disabled **No Models Found** placeholder is shown.
- **Retrain Status** -- tray menu action (inside the Prediction Model submenu) that checks whether retraining is due based on the latest model's age and the configured cadence. Shows a notification with "Retrain recommended" (with model name and creation date) or "Model is current". Uses `--retrain-config` to load a custom retrain YAML config; falls back to defaults when not provided. Disabled when `--models-dir` is not set.
- **Tray menu order (pystray)** -- **Toggle Dashboard**, **Pause** / **Resume**, **Show Status**; then **Today's Labels**, **Import Labels**, **Export Labels**; then **Prediction Model** (submenu), **Open Data Folder**, **Edit Config**, **Advanced** (submenu: **Edit Inference Policy**), **Report Issue**; then **Quit**. Left-click opens or toggles the dashboard depending on mode; right-click shows this menu.
- **Shell UI parity** -- The compact badge route in a normal browser tab uses a light solid page background and right-aligned in-page label/panel popups with hover, pin, and 300 ms delayed hide so it feels closer to the packaged Electron shell. `Host.invoke` is still a no-op without a host bridge. The `?view=label` and `?view=panel` routes match native popup markup. Use the pywebview floating window, `taskclf electron`, or Electron dev with the sidecar backend for full multi-window behavior.
- **Gap-fill surface** -- tracks unlabeled time since the last confirmed label and publishes `unlabeled_time` events every poll cycle (passive badge). Active `gap_fill_prompt` events are published only at idle return (>5 min), session start, or immediately after accepting a transition suggestion. When unlabeled time exceeds `gap_fill_escalation_minutes` (default 480), the tray icon changes to orange (`gap_fill_escalated` event). No popup or notification is sent on escalation.
- **Event broadcasting** -- publishes `status`, `tray_state`, `initial_app`, `prediction`, `label_created`, `labels_changed`, `suggest_label`, `unlabeled_time`, `gap_fill_prompt`, and `gap_fill_escalated` events to the shared EventBus for connected WebSocket clients.
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
  prompt with a concrete local-time range (e.g. "Was this Coding? 12:00–12:47").
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

Shared monitor and suggestion helpers are documented in [ui.runtime](runtime.md).

::: taskclf.ui.tray
