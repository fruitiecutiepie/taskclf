# scripts/payload_build.py

Build helper for the **Electron backend sidecar** distributed as `payload-<triple>.zip` on GitHub releases (tags `v*`).

## Role

- Runs **PyInstaller** in **one-folder** mode (`--onedir`) with `--name entry`.
- Bundles the built web UI static assets under `taskclf/ui/static` (requires `make ui-build` first).
- Collects **`taskclf` Python code** via `--collect-submodules taskclf` only. It does **not** use `--collect-all taskclf`, because that would sweep the entire `src/taskclf` tree (including `ui/frontend/node_modules`, hundreds of MB of dev tooling) into the sidecar.
- After PyInstaller runs, the build **strips** `taskclf/ui/frontend` from the one-folder output if present, and **fails** if a raw dev frontend tree or `node_modules` paths leak into `_internal/`.
- Stages the one-folder output into `build/payload/backend/` (executable `entry` or `entry.exe` plus `_internal/`).
- Writes `build/payload-<triple>.zip` with a top-level `backend/` directory.

**Shipped UI:** only the compiled static assets under `taskclf/ui/static`. **Not shipped:** SolidJS sources, `package.json`, lockfiles, or `node_modules` under `taskclf/ui/frontend`.

The layout must stay compatible with `getActivePayloadBackendPath()` in `electron/updater.ts` (`backend/entry` or `backend/entry.exe`).

## Usage

From the repo root, after `make ui-build`:

```bash
uv sync --group bundle
uv run --group bundle python scripts/payload_build.py
```

Or:

```bash
make build-payload
```

`--triple` overrides the zip basename (defaults to `scripts/host_target_triple.py`).

## See also

- [`ui/electron_shell`](../ui/electron_shell.md) — manifest and launcher behavior
- Repository file `scripts/host_target_triple.py` — LLVM-style triple strings (must match `electron/updater.ts`)
