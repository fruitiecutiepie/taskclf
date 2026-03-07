# TODO: User Error Reporting

Make it easy for users to report errors when something goes wrong in taskclf.

**Repo:** `https://github.com/fruitiecutiepie/taskclf`
**Issue tracker:** `https://github.com/fruitiecutiepie/taskclf/issues`

---

## Current State

- **CLI** prints errors to stderr via `typer.echo(..., err=True)` and exits with code 1. Logs are also persisted to `<TASKCLF_HOME>/logs/taskclf.log` via a `RotatingFileHandler` with `SanitizingFilter`.
- **Tray app** shows desktop notifications for some errors (export/import/model failures) via `_notify()` → `_send_desktop_notification()`. Logs are persisted to the same rotating log file (set up in `TrayLabeler.run()`). A "Report Issue" menu item opens the GitHub issue tracker pre-filled with version/OS info.
- **Frontend** shows flash/status messages for label CRUD errors. `SuggestionBanner` only logs to `console.error`.
- **Server** raises `HTTPException` with structured details (e.g., 409 for label overlap).
- **Logging** uses Python `logging` module with `SanitizingFilter` (redacts PII). Console output goes to stderr; file output goes to `<TASKCLF_HOME>/logs/taskclf.log` with rotation (5 MB, 3 backups) at DEBUG level.
- **GitHub issue templates** are available at `.github/ISSUE_TEMPLATE/` (bug report and feature request YAML forms).
- **No** diagnostics command, crash handler, or frontend "Report Issue" link yet.

---

## ~~TODO 1: Persist logs to a file~~ DONE

**Status:** Implemented.

**What was done:**
- Added `"logs"` to `_SUBDIRS` in `core/paths.py` so `ensure_taskclf_dirs()` creates the logs directory.
- Added `DEFAULT_LOG_DIR` constant to `core/defaults.py`.
- Added `setup_file_logging()` utility in `core/logging.py` — creates a `RotatingFileHandler` (5 MB, 3 backups, DEBUG level) with `SanitizingFilter`, attached to the root logger. Idempotent (safe to call from both CLI and tray).
- CLI `main()` callback calls `setup_file_logging()` after `ensure_taskclf_dirs()`.
- `TrayLabeler.run()` calls `setup_file_logging()` for standalone tray launches.
- Tests added to `test_core_paths.py`, `test_core_defaults.py`, `test_core_logging.py`, and `test_cli_main.py`.

**Log file paths** (platform-dependent):
- macOS: `~/Library/Application Support/taskclf/logs/taskclf.log`
- Linux: `~/.local/share/taskclf/logs/taskclf.log`
- Windows: `%LOCALAPPDATA%/taskclf/logs/taskclf.log`

---

## TODO 2: Add a `taskclf diagnostics` CLI command

**Priority:** High — gives users a copy-pasteable block for bug reports.

**What:** A command that collects environment info and prints it to stdout (or writes to a file).

**Where to change:**

- `src/taskclf/cli/main.py` — add a new top-level command `diagnostics` (after the existing command groups starting at line 66).

**Output should include:**
- `taskclf` version (from `importlib.metadata.version("taskclf")`)
- Python version (`sys.version`)
- OS and architecture (`platform.platform()`, `platform.machine()`)
- `TASKCLF_HOME` resolved path (from `core/paths.taskclf_home()`)
- Whether ActivityWatch is reachable (GET `<aw_host>/api/0/info`, timeout 5s)
- Available model bundles (from `model_registry.list_bundles()`)
- Config summary (`UserConfig(data_dir).as_dict()`, with `user_id` redacted)
- Last 50 lines of the log file (from `<TASKCLF_HOME>/logs/taskclf.log`)
- Disk usage of `data/`, `models/`, `logs/` directories

**Flags:**
- `--json` — output as JSON instead of human-readable
- `--include-logs` — append the last N log lines (default: 50)
- `--out PATH` — write to a file instead of stdout

---

## ~~TODO 3: Add GitHub issue templates~~ DONE

**Status:** Implemented.

**What was done:**
- Created `.github/ISSUE_TEMPLATE/bug_report.yml` — YAML form with fields: Description, Steps to reproduce, Expected behavior, Diagnostics output, Log excerpt, Screenshots, Additional context.
- Created `.github/ISSUE_TEMPLATE/feature_request.yml` — YAML form with fields: Problem/motivation, Proposed solution, Alternatives considered, Additional context.
- Created `.github/ISSUE_TEMPLATE/config.yml` — template chooser config with a link to discussions.

---

## ~~TODO 4: Add "Report Issue" to the tray menu~~ DONE

**Status:** Implemented.

**What was done:**
- Added `_build_report_issue_url()` method to `TrayLabeler` that constructs a GitHub new-issue URL pre-filled with the `bug_report.yml` template and version/OS diagnostics as query parameters.
- Added `_report_issue()` method that opens the URL with `webbrowser.open()`.
- Added `pystray.MenuItem("Report Issue", self._report_issue)` to `_build_menu_items()` before the final separator/Quit group.
- Added 9 tests in `TestReportIssue` class in `test_tray.py`: menu presence, URL structure, version/OS inclusion, well-formedness, browser opening, unknown version fallback, and menu ordering.

**Updated tray menu structure:**

```
Open Dashboard        (default)
Resume/Pause
─────────────
Label Stats
Import Labels
Export Labels
─────────────
Model >               (submenu)
Status
Open Data Folder
Edit Config
Report Issue          ← NEW
─────────────
Quit
```

---

## TODO 5: Add "Report Issue" link to the web frontend

**Priority:** Medium — users in browser mode need an equivalent path.

**What:** Add a small help/feedback link in the frontend UI that opens the same GitHub issue URL.

**Where to change:**

- `src/taskclf/ui/frontend/src/App.tsx` — add a link/button in the top bar or footer. In browser mode (`isBrowserMode()`, line 28), render a subtle "Report Issue" anchor that opens the GitHub URL in a new tab.
- Optionally add a `GET /api/diagnostics` endpoint in `src/taskclf/ui/server.py` so the frontend can fetch version/OS info to pre-fill the report.

---

## TODO 6: Top-level crash handler

**Priority:** Low — catches truly unexpected failures.

**What:** Wrap the CLI and tray entry points in a top-level `try/except` that writes a crash report file and prints the file path to stderr.

**Where to change:**

- `src/taskclf/cli/main.py` — wrap the Typer `app()` invocation (or use Typer's exception handler hook) to catch unhandled exceptions, write a timestamped crash file to `<TASKCLF_HOME>/logs/crash_<timestamp>.txt`, and print a message like `"taskclf crashed. Details saved to <path>. Please report at <issue URL>."`.
- `src/taskclf/ui/tray.py` — wrap `TrayLabeler.run()` similarly. On crash, also attempt a desktop notification with the crash file path.

**Crash file contents:**
- Timestamp
- Exception type, message, full traceback
- `taskclf` version, Python version, OS
- Last 20 log lines (if log file exists)

---

## TODO 7: Surface errors better in SuggestionBanner

**Priority:** Low — currently errors are silently swallowed.

**What:** `SuggestionBanner.tsx` only does `console.error("Failed to accept suggestion", err)` with no user-facing feedback.

**Where to change:**

- `src/taskclf/ui/frontend/src/components/SuggestionBanner.tsx` — show a flash/toast message when accepting a suggestion fails, consistent with how `LabelGrid.tsx` and `LabelHistory.tsx` handle errors.

---

## Implementation Order

1. ~~**TODO 1** (log file) — foundation for everything else~~ DONE
2. **TODO 2** (diagnostics command) — immediately useful, references log file
3. ~~**TODO 3** (issue templates) — references diagnostics output~~ DONE
4. ~~**TODO 4** (tray menu item) — links to issue template~~ DONE
5. **TODO 5** (frontend link) — links to issue template
6. **TODO 6** (crash handler) — writes to log dir
7. **TODO 7** (SuggestionBanner) — independent, low priority

---

## Privacy Considerations

All error reporting features must respect the project's privacy rules (see `AGENTS.md`):

- Log files must use `SanitizingFilter` — no raw window titles, keystrokes, or PII on disk.
- `taskclf diagnostics` must redact `user_id` and any sensitive config values.
- Crash files must not include raw event payloads or window titles.
- The "Report Issue" link opens GitHub in the browser — the user controls what they submit. No automatic telemetry.
