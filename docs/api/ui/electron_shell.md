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

GitHub releases must ship `manifest.json`. By default, the packaged Electron
app fetches a **launcher manifest** from its own launcher tag:

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
tags `launcher-v*`) publishes the launcher manifest plus installers and payload
zips in one release. To **re-publish** those assets for an existing tag without
creating a new tag, run the workflow manually (**Actions → Electron Launcher
Release → Run workflow**) and set **target launcher tag** (for example
`launcher-v0.4.0`). The **Python-only payload** workflow
(`.github/workflows/payload-release.yml`, tags `v*`) still publishes
per-version **payload manifests** with `platforms` entries pointing at
`payload-<triple>.zip` assets for that payload release.

The payload index is hosted on GitHub Pages at:

```text
https://fruitiecutiepie.github.io/taskclf/payload-index.json
```

That index lists available payload versions and the `manifest.json` URL for each
`v*` payload release.

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
server to answer.

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

## Packaged app: optional payload version chooser

When more than one compatible payload version exists in the payload index,
**Initial Setup** / **Core Update Required** dialogs (first download or required
update before start) and the **Core Update Available** background prompt
include an extra **Choose Version** action. That opens a small picker window
listing compatible versions (newest-first, consistent with the tray **Payload**
menu). The version chosen there applies only to that install or update step and
does **not** persist as the tray **Selected** pin (`selected.json` remains for
explicit tray-driven pinning via **Use Recommended** / **Use Installed**).

See also [`electron_payload_choice`](electron_payload_choice.md).

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
- Uses the launcher payload policy documented in
  [`electron_update_policy.md`](electron_update_policy.md)
- Uses compatible list helpers in [`electron_payload_choice`](electron_payload_choice.md)
- Works alongside the legacy pywebview shell documented in
  [`window.md`](window.md)

:::: taskclf.ui.electron_shell
