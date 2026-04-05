# ui.electron_port_conflict

Electron main-process helpers for detecting a TCP listener on the embedded UI
port, classifying whether it looks like a stale **taskclf** sidecar, and
terminating it before a new sidecar starts.

## Source

Implementation lives in [`electron/port_conflict.ts`](../../../electron/port_conflict.ts).

## Behavior summary

- Uses OS-native probes (`lsof` / `ss` on Unix, PowerShell on Windows) to find
  the listening PID and command line.
- **`classifyListenerCommandLine`** applies conservative rules: auto-recovery
  only when evidence matches the packaged `backend/entry` layout or
  `python -m taskclf.cli.main tray …` with Electron sidecar flags.
- **`killPidAndWaitForPortFree`** sends `SIGTERM` (Unix) or `taskkill` (Windows),
  polls until the port is free, then escalates to `SIGKILL` / `/F` if needed.

User-visible startup flow (dialogs, launcher log lines) is documented in
[`electron_shell`](electron_shell.md#packaged-app-ui-port-preflight).

## Integration

- Consumed by [`electron/main.ts`](../../../electron/main.ts).
- Covered by `electron/port_conflict.test.js` (run `make electron-test`).
