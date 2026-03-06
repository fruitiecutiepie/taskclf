# TODO — Tray Menu Enhancements

Potential additions to the pystray tray menu in `TrayLabeler._build_menu()` (`src/taskclf/ui/tray.py`).

Current menu:
- Open Dashboard (default / left-click)
- Pause / Resume (dynamic)
- ---
- Label Stats (notification)
- Export Labels
- ---
- Status (notification)
- Open Data Folder
- Reload Model (enabled when model_dir set)
- ---
- Quit

---

## High Priority

### ~~1. Label Stats (notification)~~ DONE

Implemented in `_label_stats()` callback + `GET /api/labels/stats` endpoint.

- Tray menu item shows a desktop notification with today's label count, total time, and per-label breakdown.
- REST endpoint returns JSON `{date, count, total_minutes, breakdown}` with optional `?date=` query parameter.
- Tests: `TestLabelStats` in `tests/test_tray.py`, `TestLabelStats` in `tests/test_ui_server.py`.

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

### ~~4. Open Data Directory~~ DONE

Implemented in `_open_data_dir()` callback.

- macOS: `subprocess.Popen(["open", ...])`, Linux: `subprocess.Popen(["xdg-open", ...])`.
- Falls back to a notification showing the path on error.
- Tests: `TestOpenDataDir` in `tests/test_tray.py`.

---

### ~~5. Reload Model~~ DONE

Implemented in `_reload_model()` callback.

- Re-reads model bundle from `self._model_dir` without restarting.
- Success: updates `_suggester` and `_model_schema_hash`, notifies user.
- Failure: keeps old model, notifies error.
- No model dir: notifies "No model directory configured".
- Menu item disabled (grayed out) when `model_dir is None`.
- Tests: `TestReloadModel` in `tests/test_tray.py`.

---

### ~~6. Connection Status (notification)~~ DONE

Implemented in `_show_status()` callback.

- Shows AW connection state, paused flag, poll count, transition count, saved labels count, model name.
- Format: "AW: connected | Polls: 142 | Transitions: 5 | Labels: 12 | Model: run_20260226".
- Tests: `TestShowStatus` in `tests/test_tray.py`.

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
Label Stats             (notification)       ✅ DONE
Import Labels           (file picker)
Export Labels           (file picker)
─────────────────────
Model ►                 (submenu)
  ☑ run_20260301
    run_20260226
    run_20260220
    (No Model)
  ─────────────────
  Reload Model                               ✅ DONE
  Check Retrain
─────────────────────
Status                  (notification)       ✅ DONE
Open Data Folder                             ✅ DONE
─────────────────────
Quit
```
