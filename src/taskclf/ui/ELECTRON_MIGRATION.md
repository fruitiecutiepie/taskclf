# Electron Shell Status

This document tracks the current Electron-based desktop shell that now
coexists with the legacy pywebview shell.

## Current status

The repo now has an **opt-in Electron shell** launched with:

```bash
taskclf electron
```

The Electron app lives in `electron/` and keeps the Python backend as a
sidecar process. The sidecar is the existing tray backend running in
headless browser mode:

```bash
taskclf tray --browser --no-tray --no-open-browser
```

That keeps the existing `EventBus`, transition detection, model-backed
suggestions, and REST/WebSocket API intact while moving the native tray
and window management into Electron.

## Runtime architecture

```text
taskclf electron
├── Electron main process
│   ├── Tray icon + menu
│   ├── Single frameless BrowserWindow
│   └── IPC bridge (preload.ts)
└── Python sidecar
    └── taskclf tray --browser --no-tray --no-open-browser
        ├── FastAPI server
        ├── ActivityMonitor
        └── WebSocket / tray-state publishing
```

## Key implementation points

- `electron/main.ts` owns the tray icon, BrowserWindow lifecycle, and
  Python sidecar startup.
- `electron/preload.ts` exposes `window.electronHost.invoke(...)` to the
  renderer.
- `src/taskclf/ui/frontend/src/lib/host.ts` now detects
  `window.electronHost` and routes host commands through Electron IPC.
- The Electron renderer uses a **single-window layout** instead of the
  pywebview shell's three native windows.
- The renderer reports semantic window states (`compact`, `label`,
  `panel`, `dashboard`) so Electron can resize the BrowserWindow without
  snapping back to the primary display.

## Why the sidecar uses `tray`

The Electron shell deliberately spawns `taskclf tray`, not `taskclf ui`,
because the tray backend already owns:

- pause/resume state
- tray-state publishing
- label/model counters
- model suggestion lifecycle

Running it with `--no-tray --no-open-browser` strips away only the native
Python shell pieces that Electron replaces.

## What remains in pywebview

The pywebview shell still exists as a fallback path:

- `taskclf tray` keeps using the pystray + pywebview shell by default
- `taskclf ui` still launches the legacy pywebview floating window
- `src/taskclf/ui/window.py` and `src/taskclf/ui/window_run.py` remain
  the implementation for that legacy path

## Current limitations

- Electron packaging/signing is not implemented yet; the shell currently
  expects a repo checkout with `electron/node_modules/` installed.
- The Electron tray menu is intentionally minimal and does not yet mirror
  every pystray menu action.
- The pywebview shell remains available because the Electron path is an
  incremental migration, not a hard cutover.

## Validation checklist

### Automated checks

- `uv run ruff check src/taskclf/cli/main.py src/taskclf/ui/tray.py src/taskclf/ui/electron_shell.py tests/test_cli_commands.py tests/test_tray.py tests/test_ui_electron_shell.py`
- `uv run pytest tests/test_ui_electron_shell.py tests/test_tray.py tests/test_cli_commands.py -k "ui_electron_shell or embedded_mode_can_skip_browser_launch or DesktopShellCommands"`
- `pnpm exec vitest run src/App.test.tsx src/lib/host.test.ts` from `src/taskclf/ui/frontend/`
- `pnpm run typecheck` from `src/taskclf/ui/frontend/`
- `pnpm run typecheck` and `pnpm run build` from `electron/`
- `pnpm exec electron --version` from `electron/`

### Manual multi-display checks

1. Run `taskclf electron`.
2. Drag the compact pill horizontally across displays and verify it does not snap back to the primary display.
3. Repeat with vertically stacked displays and verify the top edge stays stable near the menu bar boundary.
4. Hover the label badge and status dot so the window resizes through `compact`, `label`, `panel`, and `dashboard` modes while preserving the window's current display and right edge.
5. Use the Electron tray menu to toggle the dashboard, open the browser fallback, and toggle pause.
6. Quit from the Electron tray and verify the Python sidecar exits with it.
