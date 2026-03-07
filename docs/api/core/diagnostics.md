# core.diagnostics

Collect environment and runtime diagnostics for bug reports.

Used by both `taskclf diagnostics` (CLI) and the tray "Report Issue" menu
item so that users never have to gather this information manually.

## Collected information

- `taskclf` version, Python version, OS, and architecture
- `TASKCLF_HOME` path
- ActivityWatch reachability
- Model bundles (ID, validity, creation date)
- User config (with `user_id` redacted)
- Disk usage for `data/`, `models/`, and `logs/`
- Optional log tail (last N lines)

## Privacy

- `user_id` is always redacted before inclusion.
- Log tail lines are sanitized via `redact_message()` when read through `core.crash._read_log_tail`.
- No data leaves the machine unless the user explicitly submits a bug report.

## Tray integration

When the user clicks **Report Issue** in the system tray, the diagnostics
output and a sanitized log excerpt are automatically collected and pre-filled
into the GitHub issue template (`diagnostics` and `logs` fields).

If the resulting URL exceeds the browser length limit, the log excerpt is
dropped to keep the URL valid.

::: taskclf.core.diagnostics
