# ui.labeling

Streamlit labeling UI for ground-truth collection.

## Launch

```bash
streamlit run src/taskclf/ui/labeling.py -- --data-dir data/processed
```

Optional flags:

```bash
streamlit run src/taskclf/ui/labeling.py -- \
  --data-dir data/processed \
  --aw-host http://localhost:5600 \
  --title-salt my-salt
```

## Panels

- **Queue** -- Pending `LabelRequest` items sorted by confidence (lowest first). Shows time range, predicted label, confidence, and reason.
- **New Label** -- Form with date/time pickers, `CoreLabel` dropdown, confidence slider, and user ID input. Displays a feature summary (top apps, input rates) before submission.
- **Label Recent** -- Quick-label the last N minutes. Includes a slider (1-60 min), live ActivityWatch summary (top apps in the window), on-disk feature summary when available, and a label dropdown. Designed for real-time labeling as you work.
- **History** -- Recent labels in a sortable table.

## Privacy

The UI never displays raw window titles, keystrokes, or URLs.
Only aggregated metrics and application identifiers are shown.

::: taskclf.ui.labeling

---

# ui.tray

System tray labeling app for continuous background labeling.

## Launch

```bash
taskclf tray
taskclf tray --model-dir models/run_20260226
```

## Features

- **System tray icon** -- runs persistently in the background via pystray.
- **Activity transition detection** -- polls ActivityWatch and detects when the dominant foreground app changes. A transition fires when the new app persists for >= `--transition-minutes` (default 3).
- **Desktop notifications** -- on each transition, a notification prompts the user to label the completed block.
- **Label suggestions** -- when `--model-dir` is provided, the app predicts a label and includes it in the notification. Without a model, all 8 core labels are shown.
- **Quick-label menus** -- right-click the tray icon to label the last 5/10/15/30 minutes with any core label.
- **Open Streamlit UI** -- menu option to launch the full Streamlit labeling UI.

## Privacy

Same guarantees as the Streamlit UI: no raw window titles or keystrokes are displayed or stored.

::: taskclf.ui.tray
