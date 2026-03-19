# TODO: User Error Reporting

Make it easy for users to report errors when something goes wrong in taskclf.

**Repo:** `https://github.com/fruitiecutiepie/taskclf`
**Issue tracker:** `https://github.com/fruitiecutiepie/taskclf/issues`

---

## Current State

- **CLI** prints errors to stderr via `typer.echo(..., err=True)` and exits with code 1. Logs are also persisted to `<TASKCLF_HOME>/logs/taskclf.log` via a `RotatingFileHandler` with `SanitizingFilter`.
- **Tray app** shows desktop notifications for some errors (export/import/model failures) via `_notify()` → `_send_desktop_notification()`. Logs are persisted to the same rotating log file (set up in `TrayLabeler.run()`). A "Report Issue" menu item opens the GitHub issue tracker pre-filled with version/OS info.
- **Frontend** shows flash/status messages for label CRUD errors. `SuggestionBanner` shows inline error messages with auto-dismiss on accept failure.
- **Server** raises `HTTPException` with structured details (e.g., 409 for label overlap).
- **Logging** uses Python `logging` module with `SanitizingFilter` (redacts PII). Console output goes to stderr; file output goes to `<TASKCLF_HOME>/logs/taskclf.log` with rotation (5 MB, 3 backups) at DEBUG level.
- **GitHub issue templates** are available at `.github/ISSUE_TEMPLATE/` (bug report and feature request YAML forms).
- **`taskclf diagnostics`** CLI command collects version, OS, AW reachability, config, model bundles, disk usage, and log tail. Supports `--json`, `--include-logs`, `--log-lines`, and `--out` flags.
- **Crash handler** wraps both CLI (`cli_main()`) and tray (`TrayLabeler.run()`) entry points. On unhandled exceptions, writes a timestamped crash report to `<TASKCLF_HOME>/logs/crash_<timestamp>.txt` with environment info and sanitized log tail. CLI prints the path and issue URL to stderr; tray attempts a desktop notification.

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

## ~~TODO 2: Add a `taskclf diagnostics` CLI command~~ DONE

**Status:** Implemented.

**What was done:**
- Added `_collect_diagnostics()` helper in `cli/main.py` — gathers version, Python version, OS/arch, `TASKCLF_HOME` path, ActivityWatch reachability (GET with 5s timeout), model bundles via `model_registry.list_bundles()`, config summary (with `user_id` redacted), disk usage of `data/`, `models/`, `logs/`, and optionally the last N log lines.
- Added `_format_diagnostics_text()` for human-readable output.
- Added `diagnostics` top-level Typer command with flags: `--json`, `--include-logs/--no-include-logs`, `--log-lines N`, `--out PATH`.
- AW unreachable is handled gracefully (reports `reachable: false` with error string, does not fail the command).
- `user_id` is always redacted in config output. Log lines are already sanitized by `SanitizingFilter`.
- 21 tests added in `TestDiagnostics` class in `test_cli_main.py` (TC-E2E-DIAG): exit code, human/JSON output, all expected keys, AW unreachable handling, `user_id` redaction, privacy checks, empty model bundles, disk usage, log tail with `--include-logs`/`--log-lines`, missing/empty log file, `--out` file write, and `--json --out` combination.

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
Toggle Dashboard      (default)
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

## ~~TODO 5: Top-level crash handler~~ DONE

**Status:** Implemented.

**What was done:**
- Added `src/taskclf/core/crash.py` with `write_crash_report(exc, *, log_dir, log_tail_lines)` — writes a timestamped crash file to `<TASKCLF_HOME>/logs/crash_<YYYYMMDD_HHMMSS>.txt` containing: timestamp, exception type/message/traceback, taskclf version, Python version, OS, and the last N sanitized log lines (via `redact_message()`). Handles missing/empty log files gracefully and avoids overwriting on same-second collisions.
- Added `cli_main()` wrapper in `cli/main.py` — wraps `app()` in `try/except Exception`, writes crash report, prints crash file path and issue URL to stderr, and exits with code 1. `SystemExit` and `KeyboardInterrupt` pass through. Updated `pyproject.toml` entry point from `app` to `cli_main`.
- Refactored `TrayLabeler.run()` in `ui/tray.py` — split into `run()` (crash wrapper) and `_run_inner()` (original logic). On crash, writes crash report and attempts a desktop notification with the crash file path before re-raising. `SystemExit` and `KeyboardInterrupt` pass through.
- 12 tests added in `TestWriteCrashReport` class in `test_core_crash.py` (TC-CRASH-001 through TC-CRASH-010): file creation, filename format, timestamp, exception details, version/OS info, issue URL, log tail inclusion, missing/empty log file, privacy/sanitization, collision avoidance, and directory creation.
- 4 tests added in `TestCrashHandler` class in `test_cli_main.py` (TC-CRASH-CLI): crash report + exit code 1, stderr output with path and URL, `SystemExit` pass-through, `KeyboardInterrupt` pass-through.
- 5 tests added in `TestTrayCrashHandler` class in `test_tray.py` (TC-CRASH-TRAY): crash report creation, desktop notification attempt, exception re-raise, `SystemExit` pass-through, `KeyboardInterrupt` pass-through.

**Crash file path** (platform-dependent):
- macOS: `~/Library/Application Support/taskclf/logs/crash_<YYYYMMDD_HHMMSS>.txt`
- Linux: `~/.local/share/taskclf/logs/crash_<YYYYMMDD_HHMMSS>.txt`
- Windows: `%LOCALAPPDATA%/taskclf/logs/crash_<YYYYMMDD_HHMMSS>.txt`

---

## ~~TODO 7: Surface errors better in SuggestionBanner~~ DONE

**Status:** Implemented.

**What was done:**
- Added `createSignal<string | null>` for error state in `SuggestionBanner.tsx`.
- On `accept()` catch: sets error signal with the error message and auto-clears after 4 seconds (matching the pattern used in `LabelGrid.tsx`).
- Error cleared at the start of each accept attempt.
- Error rendered inline below the Accept/Dismiss buttons via `<Show when={error()}>`, styled with `color: var(--error)`.
- `console.error` retained for developer visibility.

---

## Implementation Order

1. ~~**TODO 1** (log file) — foundation for everything else~~ DONE
2. ~~**TODO 2** (diagnostics command) — immediately useful, references log file~~ DONE
3. ~~**TODO 3** (issue templates) — references diagnostics output~~ DONE
4. ~~**TODO 4** (tray menu item) — links to issue template~~ DONE
5. ~~**TODO 5** (crash handler) — writes to log dir~~ DONE
7. ~~**TODO 7** (SuggestionBanner) — independent, low priority~~ DONE

---

## Privacy Considerations

All error reporting features must respect the project's privacy rules (see `AGENTS.md`):

- Log files must use `SanitizingFilter` — no raw window titles, keystrokes, or PII on disk.
- `taskclf diagnostics` must redact `user_id` and any sensitive config values.
- Crash files must not include raw event payloads or window titles.
- The "Report Issue" link opens GitHub in the browser — the user controls what they submit. No automatic telemetry.
