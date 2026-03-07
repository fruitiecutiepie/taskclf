# core.crash

Top-level crash handler that writes a report file on unhandled exceptions.

Crash reports include environment info and sanitized log tail so the user
can attach them to GitHub issue reports without manual data gathering.

## Crash file contents

Each crash report contains:

- UTC timestamp
- Exception type, message, and full traceback
- `taskclf` version, Python version, and OS
- Last N sanitized log lines (if log file exists)
- Link to the GitHub issue tracker

## Crash file location

Files are written to `<TASKCLF_HOME>/logs/crash_<YYYYMMDD_HHMMSS>.txt`.

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/taskclf/logs/crash_*.txt` |
| Linux | `~/.local/share/taskclf/logs/crash_*.txt` |
| Windows | `%LOCALAPPDATA%/taskclf/logs/crash_*.txt` |

## Privacy

- Log tail lines are sanitized via `redact_message()` before inclusion.
- Tracebacks contain only source code references, not data payloads.
- No automatic telemetry — the user controls what they submit.

::: taskclf.core.crash
