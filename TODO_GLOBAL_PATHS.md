# TODO — Global Path Defaults for Tool-Installed Usage

## Status: **Complete**

All items have been implemented.

## Problem

All default paths in `src/taskclf/core/defaults.py` were **relative**:

- `DEFAULT_DATA_DIR` = `"data/processed"`
- `DEFAULT_MODELS_DIR` = `"models"`
- `DEFAULT_OUT_DIR` = `"artifacts"`
- `DEFAULT_RAW_AW_DIR` = `"data/raw/aw"`
- `DEFAULT_TELEMETRY_DIR` = `"artifacts/telemetry"`

This works when running from the project root (`cd ~/taskclf && taskclf tray`),
but breaks when the tool is installed globally via `uv tool install taskclf` or
`pipx install taskclf` and invoked from an arbitrary directory.

## Solution

Introduced a **`TASKCLF_HOME`** environment variable (with XDG/platform defaults)
that anchors all data/model/artifact paths.

### Resolution order

1. `TASKCLF_HOME` env var (if set)
2. `~/Library/Application Support/taskclf/` (macOS)
3. `~/.local/share/taskclf/` (Linux, following XDG_DATA_HOME)
4. `%LOCALAPPDATA%/taskclf/` (Windows, if ever supported)

### Default directory layout under `TASKCLF_HOME`

```
~/.local/share/taskclf/       # or equivalent
├── data/
│   ├── raw/aw/
│   └── processed/
├── models/
├── artifacts/
│   └── telemetry/
└── configs/
    └── retrain.yaml
```

### Changes completed

- [x] Add `taskclf_home()` helper to a new `core/paths.py`
      that resolves the base directory per the resolution order above.
- [x] Change each `DEFAULT_*` path constant to be computed from `taskclf_home()`.
- [x] Ensure CLI `--data-dir`, `--models-dir`, etc. still override the defaults
      (absolute paths take precedence — Typer defaults are overridden by explicit args).
- [x] Auto-create `TASKCLF_HOME` subdirectories on first use (with a log message)
      via `ensure_taskclf_dirs()`.
- [x] Wire `ensure_taskclf_dirs()` into the CLI entrypoint (Typer `@app.callback()`).
- [x] Update `AGENTS.md` repo map (`core/paths.py` added).
- [x] Update CLI help text to mention `TASKCLF_HOME`.
- [x] Add tests for path resolution (env var set, env var unset, platform detection).
- [x] Update docs (installation guide, CLI reference, new `docs/api/core/paths.md`).
