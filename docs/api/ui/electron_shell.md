# ui.electron_shell

Electron desktop-shell launcher for taskclf.

## Overview

`taskclf.ui.electron_shell` is the Python-side launcher for the optional
Electron desktop shell. It does not run Electron-specific business logic
itself; instead, it prepares a small environment contract and then starts
the Electron app in the foreground.

The Electron main process uses that environment to spawn the existing
Python tray backend in headless browser mode:

```bash
taskclf tray --browser --no-tray --no-open-browser
```

That keeps the existing FastAPI server, `EventBus`, activity monitor, and
WebSocket feeds intact while moving native tray/window responsibilities
into Electron.

## ElectronLaunchConfig

`ElectronLaunchConfig` captures the runtime values forwarded from the
Python CLI to Electron:

- model selection (`model_dir`, `models_dir`)
- ActivityWatch connection (`aw_host`)
- polling and transition timing (`poll_seconds`, `transition_minutes`)
- privacy-sensitive runtime paths (`data_dir`, `title_salt`)
- shell mode (`dev`, `ui_port`)
- optional user-facing metadata (`username`, `retrain_config`)

The dataclass is intentionally small and shell-focused; it does not
expose Electron internals to the rest of the Python codebase.

## launch_electron_shell

```python
launch_electron_shell(
    config: ElectronLaunchConfig,
    *,
    python_executable: str | None = None,
) -> None
```

Builds and launches the Electron shell from the repo's `electron/`
directory using `pnpm run start`.

Behavior:

- verifies that the repo checkout contains `electron/package.json`
- requires `electron/node_modules/` to exist
- resolves `pnpm` from the current PATH
- forwards the current Python executable by default so Electron can spawn
  a compatible sidecar backend
- blocks until the Electron process exits

## Payload release manifest

GitHub releases must ship `manifest.json` (and matching `payload-<triple>.zip`
files) so `https://github.com/<org>/<repo>/releases/latest/download/manifest.json`
resolves for the **latest** release. The **Electron launcher** workflow
(`.github/workflows/electron-release.yml`, tags `launcher-v*`) builds payloads
on each OS, then publishes `manifest.json` plus installers and zips in one
release. The **Python-only payload** workflow (`.github/workflows/payload-release.yml`,
tags `v*`) publishes the same manifest shape for library users who only need the
sidecar zip.

The manifest has a `platforms` object. Keys are **LLVM-style target triples**, for example
`x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`, matching
`scripts/host_target_triple.py` and the Electron updater’s
`hostTargetTriple()`. Each entry points at `payload-<triple>.zip` for that
architecture and OS.

The zip contains a **PyInstaller one-folder** sidecar: after extraction, the
Electron app runs `backend/entry` (Unix) or `backend/entry.exe` (Windows),
alongside `_internal/` and bundled data. Builds are produced with
`make build-payload` (see [`Payload build (PyInstaller)`](../scripts/payload_build.md)).

## Packaged app: payload download progress

In a **packaged** Electron build (`app.isPackaged`), the main process downloads
the payload zip from the release manifest when needed (first launch without a
local core, or when the user accepts an in-app update). While the zip is
downloading, a small native progress window shows **percentage** when the
server sends `Content-Length`, plus byte counts; after download, it shows
**Verifying** and **Extracting** stages. Implementation lives in
`electron/updater.ts` (streaming download + SHA-256) and `electron/main.ts`
(progress window). Optional override: `TASKCLF_MANIFEST_URL` for the manifest
location (see `checkForUpdate()` in `electron/updater.ts`).

## Integration

- Used by the `taskclf electron` CLI command in
  [`docs/api/cli/main.md`](../cli/main.md)
- Read by the Electron app in `electron/main.ts`
- Works alongside the legacy pywebview shell documented in
  [`window.md`](window.md)

:::: taskclf.ui.electron_shell
