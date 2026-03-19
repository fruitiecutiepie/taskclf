# TODO — Tray Menu Enhancements

Potential additions to the pystray tray menu in `TrayLabeler._build_menu()` (`src/taskclf/ui/tray.py`).

Current menu:
- Toggle Dashboard (default / left-click)
- Pause / Resume (dynamic)
- ---
- Label Stats (notification)
- Import Labels (file picker + merge/overwrite)
- Export Labels
- ---
- Model (submenu: bundle list + No Model + Reload Model)
- Status (notification)
- Open Data Folder
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

### ~~2. Import Labels~~ DONE

Implemented in `_import_labels()` callback + `POST /api/labels/import` endpoint.

- Tray menu item opens a file picker for a CSV, then asks the user to choose merge or overwrite strategy via a tkinter messagebox.
- **Merge** deduplicates by `(start_ts, end_ts, user_id)` (imported wins on collision), validates no overlaps in the merged set, writes the result. Uses `merge_label_spans()` in `labels/store.py`.
- **Overwrite** replaces all existing labels with the imported set.
- REST endpoint accepts multipart `file` + `strategy` form field (default `"merge"`). Returns `{status, imported, total, strategy}`. 409 on overlap conflicts during merge, 422 on invalid CSV.
- Tests: `TestImportLabels` in `tests/test_tray.py` (6 tests: merge, overwrite, cancel file, cancel strategy, bad CSV, menu structure), `TestImportLabels` in `tests/test_ui_server.py` (7 tests: merge, overwrite, overlap 409, dedup, invalid CSV, invalid strategy, default strategy).

---

## Medium Priority

### ~~3. Switch Model (submenu)~~ DONE

Implemented in `_build_model_submenu()`, `_switch_model()`, and `_unload_model()` callbacks.

- "Model" submenu built dynamically from `model_registry.list_bundles(models_dir)`. Each valid bundle is a radio-button `MenuItem`; the currently loaded one is checked via `checked=lambda`.
- Clicking a different bundle calls `_switch_model(path)` which instantiates a new `_LabelSuggester`, updates `_model_dir`, `_suggester`, `_model_schema_hash`, and notifies success. On failure, the old model is preserved.
- "(No Model)" entry unloads the model entirely (clears `_suggester`, `_model_dir`, suggestions).
- "Reload Model" moved from top-level menu into the Model submenu.
- Fallback: when `models_dir` is `None`, doesn't exist, is empty, or contains only invalid bundles, shows a disabled "(no models found)" placeholder.
- CLI: `--models-dir` option (default `models/`) passed to `TrayLabeler`.
- Tests: `TestSwitchModel` in `tests/test_tray.py` (12 tests: list valid bundles, exclude invalid, checked state, checked when unloaded, switch success, switch failure, noop if same, unload, no models_dir fallback, empty dir fallback, all-invalid fallback, submenu in main menu). Updated `TestBuildMenuEnhancements` (4 tests: menu items, submenu contains reload, reload disabled/enabled).
- Docs: `docs/api/ui/labeling.md` updated.

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

### ~~7. Check Retrain~~ DONE

Implemented in `_check_retrain()` callback + "Check Retrain" menu item in Model submenu.

- Tray menu item (inside Model submenu) checks whether retraining is due based on the latest model's age and configured cadence. Shows a desktop notification with "Retrain recommended (cadence: Nd, last: model_name created date)" or "Model is current (model_name, created date)". When no models exist, notifies "Retrain recommended: no models found".
- Uses `check_retrain_due()` and `find_latest_model()` from `taskclf.train.retrain`.
- Optional `--retrain-config` CLI option loads a custom retrain YAML config; falls back to `RetrainConfig()` defaults when not provided.
- Menu item disabled when `--models-dir` is not set.
- CLI: `--retrain-config` option added to `taskclf tray` command.
- Tests: `TestCheckRetrain` in `tests/test_tray.py` (8 tests: no models_dir, retrain due no models, retrain due with model, model is current, with config file, exception handling, submenu enabled, submenu disabled). Updated `TestBuildMenuEnhancements` and `TestSwitchModel` to verify "Check Retrain" in submenu.
- Docs: `docs/api/ui/labeling.md` updated.

---

## Known Gaps

### Model submenu does not refresh after retrain

pystray builds the menu once at startup. After `taskclf train retrain` creates a
new bundle (e.g. `models/run_20260307_143022/`), the Model submenu still shows
the old list. The user must restart the tray to see the new model.

**Note:** "Reload Model" does not help here — it re-reads the *currently loaded*
bundle's directory from disk, but a retrain produces a *new* directory.

See `TODO_DYNAMIC_MODEL_MENU.md` for the proposed fix.

---

## Proposed Menu Structure

After all items are implemented:

```
Toggle Dashboard          (default / left-click)
Pause / Resume          (dynamic)
─────────────────────
Label Stats             (notification)       ✅ DONE
Import Labels           (file picker)         ✅ DONE
Export Labels           (file picker)
─────────────────────
Model ►                 (submenu)             ✅ DONE
  ☑ run_20260301
    run_20260226
    run_20260220
    (No Model)
  ─────────────────
  Reload Model                               ✅ DONE
  Check Retrain                              ✅ DONE
─────────────────────
Status                  (notification)       ✅ DONE
Open Data Folder                             ✅ DONE
─────────────────────
Quit
```
