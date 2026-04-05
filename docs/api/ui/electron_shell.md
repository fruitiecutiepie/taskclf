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

The Electron build also generates separate icon assets under `electron/build/`:

- `icon.png` for the app/window icon on all platforms
- `trayTemplate.png` and `trayTemplate@2x.png` for the macOS menu-bar tray icon

The tray icon is intentionally separate from the app icon because macOS expects
a monochrome template image for status-bar rendering; using the colored app icon
can leave the tray glyph invisible until the item is highlighted.

See also [`electron_tray_icon`](electron_tray_icon.md).

## Tray interaction policy

The Electron shell intentionally treats tray interactions differently
depending on source:

- clicking the tray icon shows or focuses the dashboard and does not hide
  an already-open shell
- the tray menu keeps an explicit **Toggle Dashboard** action for hide/show
  behavior
- after the sidecar is ready, the full tray menu mirrors the Python tray labels
  (**Show Status**, **Today's Labels**, **Prediction Model** with **Refresh Model**
  and **Retrain Status**, **Open Data Folder**, **Edit Config**, **Advanced** with **Edit Inference Policy**, etc.); packaged builds also include **Backend Versions** and
  **Check for Updates** after **Prediction Model**

That keeps the Electron shell aligned with the documented pystray UX
where a primary tray click opens the UI, while still preserving a clear
way to hide it on demand.

## Launcher and payload release manifests

Launcher and payload releases are separate GitHub release tags:

- **`launcher-v*`** — desktop installers plus a **launcher** `manifest.json` (compatibility policy and payload index URL). Built by `.github/workflows/electron-release.yml`.
- **`v*`** — PyInstaller sidecar zips plus a **payload** `manifest.json` (per-platform URLs and SHA-256). Built by `.github/workflows/payload-release.yml`.

By default, the packaged Electron app fetches a **launcher manifest** from its own launcher tag:

```text
https://github.com/<org>/<repo>/releases/download/launcher-v<appVersion>/manifest.json
```

The launcher manifest now declares:

- `launcher_version`
- `payload_index_url`
- `default_payload_selection`
- `compatible_payloads`

That keeps compatibility policy with the launcher release while allowing the
app to boot the **latest compatible** `v*` payload instead of forcing an exact
launcher-version match.

The **Electron launcher** workflow (`.github/workflows/electron-release.yml`,
tags `launcher-v*`) publishes the launcher manifest plus installers only (no
payload zips on that tag). To **re-publish** those assets for an existing tag
without creating a new tag, run the workflow manually (**Actions → Electron
Launcher Release → Run workflow**) and set **target launcher tag** (for example
`launcher-v0.4.0`). The **payload** workflow
(`.github/workflows/payload-release.yml`, tags `v*`) publishes per-version
**payload manifests** with `platforms` entries pointing at
`payload-<triple>.zip` assets for that payload release.

The payload index is hosted on GitHub Pages at:

```text
https://fruitiecutiepie.github.io/taskclf/payload-index.json
```

That index lists available payload versions and the `manifest.json` URL for each
`v*` payload release.  The payload release workflow is the writer for this file;
regular docs deploys preserve the currently published copy so a `master` push
cannot republish stale release metadata over a newer payload tag.

The zip contains a **PyInstaller one-folder** sidecar: after extraction, the
Electron app runs `backend/entry` (Unix) or `backend/entry.exe` (Windows),
alongside `_internal/` and bundled data. Builds are produced with
`make build-payload` (see [`Payload build (PyInstaller)`](../scripts/payload_build.md)).

## Packaged app: payload download progress

In a **packaged** Electron build (`app.isPackaged`), the main process fetches
the launcher manifest, then the GitHub Pages payload index, then the payload
manifest for the chosen compatible version. If the desired payload is already
installed, the launcher switches `active.json`; otherwise it downloads the
payload zip from the chosen payload release manifest. Before any first-run
network call or sidecar boot work begins, the launcher shows a small native
**startup status** window so the app does not appear idle while it is checking
for local core files, waiting on release metadata, or waiting for the local UI
server to answer. While polling for the sidecar HTTP server, the main process
also loads a lightweight `data:` placeholder into the pill and child webviews so
renderer startup overlaps with PyInstaller cold-start.

The Python sidecar also defers importing heavy data dependencies (for example
`pandas` and parquet reads) until REST routes that need them run, so the local
HTTP port can answer sooner than a full training/feature-build import graph
would allow.

When a payload zip is actually downloading, that same startup UX switches to a
native progress window with **percentage** when the server sends
`Content-Length`, plus byte counts; after download, it shows **Verifying** and
**Extracting** stages. Implementation lives in `electron/updater.ts`
(streaming download + SHA-256, timed manifest fetch, and `net.fetch()` via
Electron's Chromium network stack for packaged-app updater requests) and `electron/main.ts`
(startup/progress windows). Optional overrides:

- `TASKCLF_MANIFEST_URL` -- alternate launcher manifest location
- `TASKCLF_MANIFEST_TIMEOUT_MS` -- manifest fetch timeout in milliseconds
  (default: `15000`; set `0` to disable the timeout)
- `TASKCLF_ELECTRON_SHELL_WAIT_MS` -- max time to wait for the local FastAPI
  sidecar after spawn (default: `120000` packaged, `30000` dev; PyInstaller
  cold-starts can exceed 30s on slower disks)

## Packaged app: UI port preflight

Before spawning the Python sidecar, the Electron main process checks whether
`TASKCLF_ELECTRON_UI_PORT` (default `8741`) already has a TCP listener (typically
`127.0.0.1`). Implementation: [`electron/port_conflict.ts`](../../../electron/port_conflict.ts)
(see [`electron_port_conflict`](electron_port_conflict.md)), invoked from
`electron/main.ts` immediately before `spawnSidecar()`.

Behavior:

- **Port free** — startup continues as before.
- **Listener looks like a stale taskclf sidecar** — for example the packaged
  PyInstaller binary path (`…/taskclf-electron/…/backend/entry` with the usual
  `tray --browser --no-tray …` arguments, or `python -m taskclf.cli.main tray`
  with those flags) — the launcher stops that process, waits until the port is
  free, then starts the new sidecar.
- **Listener does not match those patterns (or classification is uncertain)** —
  a **Port Busy** dialog offers **Report Issue** (opens the GitHub issue URL with
  context), **Kill Port and Start taskclf** (terminate the listener and
  continue if the port clears), or **Quit**.

The launcher log records a `port preflight` line with PID, classification, and a
truncated command line for diagnostics.

## Packaged app: optional payload version chooser

When more than one compatible payload version exists in the payload index,
**Initial Setup** / **Core Update Required** dialogs (first download or required
update before start) and the **Core Update Available** background prompt
include an extra **Choose Version** action. That opens a small picker window
listing compatible versions (newest-first, consistent with the tray **Backend Versions**
menu). The version chosen there applies only to that install or update step and
does **not** persist as the tray **Selected** pin (`selected.json` remains for
explicit tray-driven pinning via **Use Recommended** / **Use Installed**).

See also [`electron_payload_choice`](electron_payload_choice.md).

## Packaged app: manual update check

The tray menu includes **Check for Updates** in packaged builds. That action
fetches the launcher manifest, payload index, and latest compatible payload on
demand, then:

- shows **Up to Date** when the active payload already matches the latest
  compatible release
- shows **Core Update Available** with **Update and Restart** (and
  **Choose Version** when multiple compatible payloads exist) when a newer
  compatible payload should be activated
- shows **Update Check Failed** when release metadata cannot be fetched

Unlike normal startup/background resolution, the manual check compares against
the latest compatible payload even when the tray **Selected** payload is pinned
to an older version, so users can explicitly discover and apply newer payload
releases. Applying the recommended update from this dialog clears the tray
selection pin and restarts in recommended/auto mode.

## Packaged app: debugging the main process

The Electron main process appends structured lines to
`<userData>/logs/electron-launcher.log` (spawn line, captured sidecar
stdout/stderr, sidecar exit code/signal, `waitForShell` timeout details,
manifest timeout/fetch failures, first-run failures). Run the `.app` from
Terminal and set **`TASKCLF_ELECTRON_DEBUG=1`** for extra console output
(periodic `waitForShell` progress and successful shell-ready confirmation).
Failed manifest fetches record HTTP status / URL in `lastManifestCheckFailure`
(see `electron/updater.ts`) and surface in a fatal launcher dialog together
with recent backend output, the launcher log path, and actions to open the log
folder or pre-fill a GitHub bug report.

## Integration

- Used by the `taskclf electron` CLI command in
  [`docs/api/cli/main.md`](../cli/main.md)
- Read by the Electron app in `electron/main.ts`
- Uses the tray interaction helper in
  [`electron_tray_dashboard`](electron_tray_dashboard.md)
- Uses the launcher payload policy documented in
  [`electron_update_policy.md`](electron_update_policy.md)
- Uses compatible list helpers in [`electron_payload_choice`](electron_payload_choice.md)
- Works alongside the legacy pywebview shell documented in
  [`window.md`](window.md)

:::: taskclf.ui.electron_shell
