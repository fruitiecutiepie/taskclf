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
