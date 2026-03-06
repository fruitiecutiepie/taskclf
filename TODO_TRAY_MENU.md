# TODO — Tray Menu Enhancements

Potential additions to the pystray tray menu in `TrayLabeler._build_menu()` (`src/taskclf/ui/tray.py`).

Current menu:
- Open Dashboard (default / left-click)
- Pause / Resume (dynamic)
- Export Labels
- ---
- Quit

---

## High Priority

### 1. Label Stats (notification)

**What:** "Label Stats" menu item that shows a desktop notification summarising today's labeling progress: total label count, time coverage (hours labeled / hours since first label), and per-label distribution (e.g. "Build 45m, Debug 20m, Write 10m").

**Why:** The most common mid-work question is "am I labeling enough?" — answering it currently requires opening the dashboard and navigating to the history tab.

**Implementation:**
- Read spans from `self._labels_path` via `read_label_spans()` (already imported in `_export_labels`).
- Filter to today (UTC date match on `start_ts`).
- Compute per-label minutes: `sum((s.end_ts - s.start_ts).total_seconds() / 60 for s in spans if s.label == label)`.
- Format as a compact notification string, e.g. "Today: 5 labels, 1h 15m — Build 45m, Debug 20m, Write 10m".
- Call `self._notify(summary)`.
- No file dialog, no parameters — a single click.
- Add `GET /api/labels/stats` endpoint in `server.py` returning JSON `{date, count, total_minutes, breakdown: {label: minutes}}` so the dashboard can also surface it.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_label_stats()` callback and menu item
- `src/taskclf/ui/server.py` — add `GET /api/labels/stats` endpoint
- `tests/test_tray.py` — test notification content
- `docs/api/ui/labeling.md` — document menu item and endpoint

---

### 2. Import Labels

**What:** "Import Labels" menu item that opens a file picker for a CSV and imports it via `import_labels_from_csv()` + `write_label_spans()`.

**Why:** Counterpart to the existing "Export Labels". Makes sharing bidirectional from the tray — a collaborator exports their labels, sends the CSV, and you import it without touching the CLI.

**Implementation:**
- Mirror `_export_labels()` structure: tkinter `askopenfilename` with `filetypes=[("CSV files", "*.csv")]`.
- Call `import_labels_from_csv(chosen_path)` to validate.
- **Merge strategy decision needed:** currently `import_labels_from_csv` returns `list[LabelSpan]` and the CLI's `labels import` command overwrites the entire parquet via `write_label_spans()`. For the tray, appending (via `append_label_span` per span) is safer but slower and may conflict. Options:
  - (a) Overwrite (matches CLI behavior, simple, but destructive).
  - (b) Append each span (safe but O(n^2) overlap checks).
  - (c) Merge: read existing + imported, deduplicate by `(start_ts, end_ts, user_id)`, check overlaps, write merged set.
- Notify on success ("Imported N labels from file.csv") or failure ("Import failed: ...").
- Add `POST /api/labels/import` endpoint accepting multipart CSV upload.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_import_labels()` callback and menu item
- `src/taskclf/ui/server.py` — add `POST /api/labels/import` endpoint
- `tests/test_tray.py` — test import success / failure / cancel
- `docs/api/ui/labeling.md` — document menu item and endpoint

---

## Medium Priority

### 3. Switch Model (submenu)

**What:** A "Model" submenu listing available model bundles from the `models/` directory, with the currently loaded one checked. Clicking a different model hot-swaps the suggester.

**Why:** Currently requires restarting the tray with a different `--model-dir`. During train/evaluate cycles you want to test different models' suggestions without restarting.

**Implementation:**
- Use `model_registry.list_bundles(models_dir)` to scan available bundles.
- Build a dynamic `pystray.Menu` submenu with one `MenuItem` per bundle, using `checked=lambda item: item.text == current_model_id` for a radio-button effect.
- On click: call `load_model_bundle(bundle.path)` and replace `self._suggester` with a new `_LabelSuggester`. Update `self._model_dir`, `self._model_schema_hash`.
- Add a "(No Model)" entry to unload the model entirely.
- Fallback: if `models/` doesn't exist or is empty, show a single disabled "(no models)" item.
- The `ActiveModelReloader` in `infer/resolve.py` already handles `active.json`-based hot-reload for the online loop; this would be the manual/interactive equivalent for the tray.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_build_model_submenu()`, `_switch_model()` callback
- `src/taskclf/core/defaults.py` — may need `DEFAULT_MODELS_DIR` imported
- `tests/test_tray.py` — test submenu construction, model swap
- `docs/api/ui/labeling.md` — document submenu

---

### 4. Open Data Directory

**What:** "Open Data Folder" menu item that opens `self._data_dir` in the OS file manager (Finder on macOS, `xdg-open` on Linux).

**Why:** Quick access to parquet files, queue, and exports without remembering or typing the path. Useful for manual inspection, copying files to share, or checking disk usage.

**Implementation:**
- macOS: `subprocess.Popen(["open", str(self._data_dir)])`.
- Linux: `subprocess.Popen(["xdg-open", str(self._data_dir)])`.
- Pattern already exists in `_send_desktop_notification` which switches on `platform.system()`.
- No notification needed; the OS handles feedback.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_open_data_dir()` callback and menu item
- `tests/test_tray.py` — test subprocess call (mock `Popen`)
- `docs/api/ui/labeling.md` — document menu item

---

### 5. Reload Model

**What:** "Reload Model" menu item that re-reads the model bundle from `self._model_dir` (or re-resolves from `active.json`) without restarting the tray.

**Why:** After retraining, the model files on disk change but the in-memory `_LabelSuggester` still uses the old weights. The online inference loop has `ActiveModelReloader` for this, but the tray has no equivalent.

**Implementation:**
- If `self._model_dir` is set, re-run the `_LabelSuggester(model_dir)` constructor in a try/except.
- On success: update `self._suggester`, `self._model_schema_hash`, notify "Model reloaded from {dir}".
- On failure: keep old model, notify "Reload failed: {error}".
- If no `model_dir` was configured, notify "No model directory configured".
- Menu item should be disabled (grayed out) when no model is loaded — use `enabled=lambda _: self._model_dir is not None`.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_reload_model()` callback and menu item
- `tests/test_tray.py` — test reload success / failure / no-model
- `docs/api/ui/labeling.md` — document menu item

---

### 6. Connection Status (notification)

**What:** "Status" menu item that shows a notification with ActivityWatch connection state, poll stats, and labeling stats.

**Why:** Quick health check without opening the dashboard's System tab. Useful when you suspect AW is down or polling stopped.

**Implementation:**
- Gather from `self._monitor`: `is_paused`, `poll_count`, `_current_app`, `_aw_connected` (if exposed).
- Gather from self: `_transition_count`, `_labels_saved_count`, `_model_dir`, `_model_schema_hash`.
- Format as notification, e.g. "AW: connected | Polls: 142 | Transitions: 5 | Labels: 12 | Model: run_20260226".
- Single click, no dialog.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_show_status()` callback and menu item
- `src/taskclf/ui/tray.py` — may need to expose AW connection state from `ActivityMonitor`
- `tests/test_tray.py` — test notification content
- `docs/api/ui/labeling.md` — document menu item

---

## Low Priority

### 7. Check Retrain

**What:** "Check Retrain" menu item that runs the retrain eligibility check and shows a notification with the result.

**Why:** Surfaces `train check-retrain` logic as a one-click check. Only useful if you have a retrain config set up.

**Implementation:**
- Requires a retrain config path (from `configs/retrain.yaml` or a default).
- Call `check_retrain_needed()` from `taskclf.train.retrain`.
- Notify "Retrain recommended: {reason}" or "Model is current (last trained {date})".
- Menu item disabled if no retrain config exists.

**Touches:**
- `src/taskclf/ui/tray.py` — add `_check_retrain()` callback and menu item
- `src/taskclf/ui/tray.py` — accept optional `--retrain-config` parameter
- `tests/test_tray.py` — test notification content
- `docs/api/ui/labeling.md` — document menu item

---

## Proposed Menu Structure

After all items are implemented:

```
Open Dashboard          (default / left-click)
Pause / Resume          (dynamic)
─────────────────────
Label Stats             (notification)
Import Labels           (file picker)
Export Labels           (file picker)
─────────────────────
Model ►                 (submenu)
  ☑ run_20260301
    run_20260226
    run_20260220
    (No Model)
  ─────────────────
  Reload Model
  Check Retrain
─────────────────────
Status                  (notification)
Open Data Folder
─────────────────────
Quit
```
