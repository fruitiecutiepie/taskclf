# Electron Migration Checklist

This document records the steps required to migrate the native floating window
from pywebview to Electron, if/when that becomes desirable (e.g. for
transparent click-through overlays, built-in tray integration, or
cross-platform packaging).

## Current architecture (pywebview)

```
taskclf ui  (single Python process)
├── FastAPI server       (background thread, port 8741)
├── ActivityMonitor      (background thread, polls AW)
└── pywebview window     (main thread, loads http://127.0.0.1:8741)
    └── SolidJS frontend
        ├── api.ts       (REST fetch — host-agnostic)
        ├── ws.ts        (WebSocket — host-agnostic)
        └── host.ts      (window control — ONLY pywebview-specific file)
```

The **Host API abstraction** (`src/lib/host.ts`) is the migration seam.
Components call `host.invoke({ cmd: "setCompact" })` — never
`window.pywebview.api.*` directly. Data operations (labels, queue,
features) use standard HTTP/WebSocket via `api.ts` and `ws.ts`.

## What ports cleanly (zero changes)

- **All SolidJS components** — they only use `host.invoke()`, `api.ts`, `ws.ts`
- **`api.ts`** — pure `fetch()` against the FastAPI server
- **`ws.ts`** — pure `WebSocket` against the FastAPI server
- **FastAPI server** (`server.py`) — stays as the Python backend
- **All Python business logic** — models, features, labels, inference

## What needs to change

### 1. Create `electron/` directory

```
electron/
├── main.ts          — app lifecycle, BrowserWindow, Tray, global hotkeys
├── preload.ts       — contextBridge exposing ipcRenderer.invoke
├── package.json     — electron + electron-builder deps
└── tsconfig.json
```

### 2. Add `ElectronHost` adapter in `host.ts`

```typescript
class ElectronHost implements Host {
  readonly isNativeWindow = true;

  async invoke(command: HostCommand): Promise<void> {
    // contextBridge exposes this from preload.ts
    return (window as any).electronHost.invoke(command);
  }
}
```

Update `detectHost()`:

```typescript
function detectHost(): Host {
  if ((window as any).electronHost) return new ElectronHost();
  if (window.pywebview) return new PyWebViewHost();
  return new BrowserHost();
}
```

### 3. Implement IPC in Electron main process

```typescript
// main.ts
ipcMain.handle("host", (_event, command: HostCommand) => {
  switch (command.cmd) {
    case "setCompact":
      mainWindow.setSize(280, 52);
      break;
    case "setExpanded":
      mainWindow.setSize(420, 560);
      break;
    case "hideWindow":
      mainWindow.hide();
      break;
  }
});
```

### 4. Python backend as a sidecar process

Electron spawns the Python server as a child process:

```typescript
const pythonProcess = spawn("taskclf", ["ui", "--browser", "--port", "8741"]);
```

The `--browser` flag (already implemented) makes the CLI skip pywebview
and just run the FastAPI server. Electron loads `http://127.0.0.1:8741`
in its BrowserWindow.

### 5. Tray integration moves into Electron

Replace the separate `taskclf tray` process. Electron's `Tray` API
handles the menu bar icon natively, with show/hide/quit built in.
The HTTP-based `/api/window/toggle` endpoint becomes unnecessary
(Electron controls its own window directly).

### 6. Window features to implement in Electron

- `alwaysOnTop: true` on the BrowserWindow
- `frame: false` for frameless
- `transparent: true` if click-through overlay is desired
- `setIgnoreMouseEvents(true, { forward: true })` for passthrough mode
- Global keyboard shortcut via `globalShortcut.register()`

## Migration order (minimal risk)

1. **Scaffold Electron shell** — `electron/`, `main.ts`, `preload.ts`
2. **Add `ElectronHost`** to `host.ts` (one adapter, ~15 lines)
3. **Implement IPC** in `main.ts` for window control commands
4. **Spawn Python sidecar** from Electron main process
5. **Move tray** from pystray to Electron `Tray`
6. **Overlay features** last — transparency, click-through, per-OS quirks
7. **Package** with electron-builder for `.dmg` / `.exe` / `.AppImage`

## Files involved

| Scope | File | Change |
|-------|------|--------|
| Frontend | `src/lib/host.ts` | Add `ElectronHost` adapter (~15 lines) |
| Frontend | `src/lib/host.ts` | Update `detectHost()` (~3 lines) |
| Electron | `electron/main.ts` | New: app lifecycle, BrowserWindow, Tray, IPC |
| Electron | `electron/preload.ts` | New: contextBridge for secure IPC |
| Electron | `electron/package.json` | New: deps + build config |
| Python | `src/taskclf/cli/main.py` | No changes (`--browser` flag already exists) |
| Python | `src/taskclf/ui/server.py` | No changes (HTTP API stays) |
| Components | All `.tsx` files | **No changes** |
| Data clients | `api.ts`, `ws.ts` | **No changes** |

## Estimated effort

- **Scenario A** (simple floating window): 1-2 days. Scaffold Electron,
  add adapter, spawn Python sidecar.
- **Scenario B** (with transparent overlay / click-through): 3-4 days.
  Add transparency, mouse-event forwarding, per-OS testing.
- **Scenario C** (full desktop app with auto-updates): 1 week.
  Add electron-builder, code signing, auto-updater, installer.
