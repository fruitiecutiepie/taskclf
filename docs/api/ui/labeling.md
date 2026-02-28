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
| `--data-dir` | `data/processed` | Processed data directory |
| `--transition-minutes` | `3` | Minutes before suggesting a label change |
| `--dev` | off | Start Vite dev server for frontend hot reload |

## Panels

- **Label** -- Form with date/time pickers, `CoreLabel` dropdown, confidence slider, and user ID input.
- **Recent** -- Quick-label the last N minutes with a slider (1-60 min). Shows a live ActivityWatch summary when available.
- **Queue** -- Pending `LabelRequest` items sorted by confidence (lowest first). Shows time range, predicted label, confidence, and reason.
- **History** -- Recent labels in a sortable table.

## Live Features

- **Live badge** -- Header displays the current predicted label and confidence, updated via WebSocket.
- **Suggestion banner** -- Appears when the ActivityMonitor detects a task change. Accept or dismiss with one click.

## Architecture

The UI is a SolidJS single-page application served by a FastAPI backend:

- **REST endpoints** (`/api/labels`, `/api/queue`, `/api/features/summary`, `/api/aw/live`, `/api/config/labels`) handle label CRUD, queue management, and data queries.
- **WebSocket** (`/ws/predictions`) streams live prediction, suggestion, and status events from the ActivityMonitor.

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
- **Event broadcasting** -- publishes prediction and suggestion events to the shared EventBus for connected WebSocket clients.

## Privacy

Same guarantees as the web UI: no raw window titles or keystrokes are displayed or stored.

::: taskclf.ui.tray
