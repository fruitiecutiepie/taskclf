# Electron Shell Status

This document tracks the current Electron-based desktop shell that now
coexists with the legacy pywebview shell.

## Agent Guide for Implementing TODOs

When an AI agent is tasked with implementing TODOs listed in this document, the agent MUST follow these steps:

1. **Read the Context**: Review the "Runtime architecture" and "Key implementation points" sections to understand how the Electron shell interacts with the Python sidecar.
2. **Analyze the Task**: Understand the requirements of the specific TODO item. Identify whether it requires changes in the Electron main process (`electron/main.ts`), the React frontend (`src/taskclf/ui/frontend/`), or the Python backend.
3. **Execute**: Make the necessary code changes, adhering to the architecture described below.
4. **Validate**: Run the relevant tests from the "Validation checklist" to ensure both automated and manual checks pass. Ensure no existing functionality is broken.
5. **Mark as Complete**: When a TODO item is fully implemented and validated, change its markdown checkbox from `[ ]` to `[x]`. **Do not delete** the item or use strikethroughs, as checkboxes provide a clean historical record of completed work.

## TODOs

- [x] Verify packaging/signing for the Electron app is possible and automate it
- [ ] Mirror all necessary pystray menu actions to the Electron tray menu
- [ ] Integrate full desktop icon instead of defaulting to a generic Electron icon
- [ ] Remove legacy pywebview shell when full migration to Electron is stable and complete

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
│   ├── Compact pill BrowserWindow (150×30)
│   ├── Label popup BrowserWindow (280×330, hidden by default)
│   ├── Panel popup BrowserWindow (280×520, hidden by default)
│   └── IPC bridge (preload.ts)
└── Python sidecar
    └── taskclf tray --browser --no-tray --no-open-browser
        ├── FastAPI server
        ├── ActivityMonitor
        └── WebSocket / tray-state publishing
```

## Key implementation points

- `electron/main.ts` owns the tray icon, BrowserWindow lifecycle, child
  window state machine, and Python sidecar startup.
- `electron/preload.ts` exposes `window.electronHost.invoke(...)` to the
  renderer.
- `src/taskclf/ui/frontend/src/lib/host.ts` now detects
  `window.electronHost` and routes host commands through Electron IPC.
- The Electron shell uses **three native windows** matching the pywebview
  shell: a compact pill, a label popup (`?view=label`), and a state
  panel popup (`?view=panel`).
- Child windows are anchored below the pill, right-aligned, with
  hover-show, click-to-pin, delayed hide (300 ms), and drag detection
  mirroring the `WindowChild` state machine in `window.py`.

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

- `uv run ruff check src/taskclf/cli/main.py src/taskclf/ui/tray.py src/taskclf/ui/electron_shell.py tests/test_cli_commands.py tests/test_ui_tray.py tests/test_ui_electron_shell.py`
- `uv run pytest tests/test_ui_electron_shell.py tests/test_ui_tray.py tests/test_cli_commands.py -k "ui_electron_shell or embedded_mode_can_skip_browser_launch or DesktopShellCommands"`
- `pnpm exec vitest run src/App.test.tsx src/lib/host.test.ts` from `src/taskclf/ui/frontend/`
- `pnpm run typecheck` from `src/taskclf/ui/frontend/`
- `pnpm run typecheck` and `pnpm run build` from `electron/`
- `pnpm exec electron --version` from `electron/`

### Manual multi-display checks

1. Run `taskclf electron`.
2. Drag the compact pill horizontally across displays and verify it does not snap back to the primary display.
3. Repeat with vertically stacked displays and verify the top edge stays stable near the menu bar boundary.
4. Hover the label badge to open the label popup window; hover the status dot to open the panel popup window. Verify both appear anchored below the pill and follow when the pill is dragged.
5. Use the Electron tray menu to toggle the dashboard, open the browser fallback, and toggle pause.
6. Quit from the Electron tray and verify the Python sidecar exits with it.
