# ui.shell_warm

Small Electron main-process helpers for loading a lightweight `data:` placeholder
into the pill `BrowserWindow` while the FastAPI sidecar starts, so renderer
startup overlaps PyInstaller cold-start without eagerly creating label/panel popup
windows.

## Source

[`electron/shell_warm.ts`](../../../electron/shell_warm.ts)

## API

- **`shellLoadingDataUrl()`** — returns a `data:text/html;charset=utf-8,…` URL
  with a minimal “Loading…” page.
- **`warmPillWindow(pill)`** — `loadURL` that placeholder into the given pill
  window only.

## Integration

- Used by [`electron/main.ts`](../../../electron/main.ts) during startup, in
  parallel with `waitForShell`.
- Startup UX for the full shell is documented in [`electron_shell`](electron_shell.md).

## Tests

`electron/shell_warm.test.js` (run `make electron-test`).
