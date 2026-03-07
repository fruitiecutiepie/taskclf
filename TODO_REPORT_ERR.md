# TODO: User Error Reporting

Make it easy for users to report errors when something goes wrong in taskclf.

**Repo:** `https://github.com/fruitiecutiepie/taskclf`
**Issue tracker:** `https://github.com/fruitiecutiepie/taskclf/issues`

---

## Current State

- **CLI** prints errors to stderr via `typer.echo(..., err=True)` and exits with code 1. No log file is written.
- **Tray app** shows desktop notifications for some errors (export/import/model failures) via `_notify()` → `_send_desktop_notification()`. Errors are logged to stderr only.
- **Frontend** shows flash/status messages for label CRUD errors. `SuggestionBanner` only logs to `console.error`.
- **Server** raises `HTTPException` with structured details (e.g., 409 for label overlap).
- **Logging** uses Python `logging` module with `SanitizingFilter` (redacts PII). All output goes to stderr — nothing is persisted to disk.
- **No** "Report Bug" button, GitHub issue templates, crash handler, or diagnostics command.

---

## TODO 1: Persist logs to a file

**Priority:** High — without this, error context is lost when the terminal closes.

**What:** Add a `FileHandler` that writes logs to `<TASKCLF_HOME>/logs/taskclf.log` with rotation.

**Where to change:**

- `src/taskclf/core/paths.py` — add `"logs"` to `_SUBDIRS` tuple (line 26) so the directory is created by `ensure_taskclf_dirs()`.
- `src/taskclf/core/defaults.py` — add `DEFAULT_LOG_DIR` constant derived from `_HOME / "logs"`.
- `src/taskclf/cli/main.py` — in the `main()` callback (lines 56–63), add a `RotatingFileHandler` alongside the existing `StreamHandler`. Apply the `SanitizingFilter` from `core/logging.py` to the file handler so PII is never written to disk.
- `src/taskclf/ui/tray.py` — ensure the tray's `TrayLabeler.run()` (line 1246) also sets up the file handler if it hasn't been initialized by the CLI path (tray can be launched standalone).

**Log file paths** (platform-dependent, from `core/paths.py`):
- macOS: `~/Library/Application Support/taskclf/logs/taskclf.log`
- Linux: `~/.local/share/taskclf/logs/taskclf.log`
- Windows: `%LOCALAPPDATA%/taskclf/logs/taskclf.log`

**Details:**
- Use `logging.handlers.RotatingFileHandler` with `maxBytes=5_000_000` (5 MB), `backupCount=3`.
- Log level for file handler: `DEBUG` always (regardless of `--verbose` flag) so errors are captured even in normal operation.
- Apply `SanitizingFilter` (from `src/taskclf/core/logging.py`) to the file handler.
- Format: `"%(asctime)s %(levelname)s %(name)s %(pathname)s:%(lineno)d — %(message)s"`.

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
- Last 50 lines of the log file (if TODO 1 is implemented)
- Disk usage of `data/`, `models/`, `logs/` directories

**Flags:**
- `--json` — output as JSON instead of human-readable
- `--include-logs` — append the last N log lines (default: 50)
- `--out PATH` — write to a file instead of stdout

---

## TODO 3: Add GitHub issue templates

**Priority:** Medium — structured issue forms reduce back-and-forth.

**What:** Create `.github/ISSUE_TEMPLATE/bug_report.yml` (YAML form) and `feature_request.yml`.

**Where to create:**

- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/ISSUE_TEMPLATE/config.yml` (optional: template chooser config)

**Bug report template fields:**
1. **Description** — what happened
2. **Steps to reproduce** — numbered list
3. **Expected behavior** — what should have happened
4. **Diagnostics output** — paste output of `taskclf diagnostics`
5. **Log excerpt** — paste relevant lines from `<TASKCLF_HOME>/logs/taskclf.log`
6. **Screenshots** — optional
7. **Additional context** — optional

---

## TODO 4: Add "Report Issue" to the tray menu

**Priority:** Medium — single-click path from the app to the issue tracker.

**What:** Add a tray menu item that opens the GitHub new-issue URL in the browser, pre-filled with version and OS info as query params.

**Where to change:**

- `src/taskclf/ui/tray.py` — in `_build_menu_items()` (line 625), add a `pystray.MenuItem("Report Issue", self._report_issue)` before the `Quit` item, inside the last separator group (after "Edit Config", line 649).
- Add a `_report_issue` method to `TrayLabeler` that:
  1. Builds a URL like `https://github.com/fruitiecutiepie/taskclf/issues/new?template=bug_report.yml&title=...`
  2. Appends version/OS as URL query params or body text
  3. Opens it with `webbrowser.open()`
- `tests/test_tray.py` — add a test that verifies the menu item exists and the URL is well-formed.

**Current tray menu structure** (for reference):

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
─────────────         ← insert "Report Issue" here
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
- `src/taskclf/ui/tray.py` — wrap `TrayLabeler.run()` (line 1246) similarly. On crash, also attempt a desktop notification with the crash file path.

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

1. **TODO 1** (log file) — foundation for everything else
2. **TODO 2** (diagnostics command) — immediately useful, references log file
3. **TODO 3** (issue templates) — references diagnostics output
4. **TODO 4** (tray menu item) — links to issue template
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
