# TODO — Global Path Defaults for Tool-Installed Usage

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

## Proposed Solution

Introduce a **`TASKCLF_HOME`** environment variable (or XDG-style default) that
anchors all data/model/artifact paths.

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

### Changes required

- [x] Add `taskclf_home()` helper to a new `core/paths.py`
      that resolves the base directory per the resolution order above.
- [x] Change each `DEFAULT_*` path constant to be computed from `taskclf_home()`.
- [x] Ensure CLI `--data-dir`, `--models-dir`, etc. still override the defaults
      (absolute paths take precedence — Typer defaults are overridden by explicit args).
- [x] Auto-create `TASKCLF_HOME` subdirectories on first use (with a log message)
      via `ensure_taskclf_dirs()`.
- [x] Update `AGENTS.md` repo map (`core/paths.py` added).
- [x] Update CLI help text to mention `TASKCLF_HOME`.
- [x] Add tests for path resolution (env var set, env var unset, platform detection).
- [x] Update docs (installation guide, CLI reference, new `docs/api/core/paths.md`).

### Remaining work

- [ ] Wire `ensure_taskclf_dirs()` into the CLI entrypoint (e.g. a Typer callback)
      so directories are auto-created on first command invocation.
- [ ] Migration: if `TASKCLF_HOME` is not set and the CWD contains a `data/` or
      `models/` directory, consider using CWD-relative paths as a fallback (with a
      deprecation warning suggesting `TASKCLF_HOME`).  Alternatively, print a
      one-time hint on first run if no `TASKCLF_HOME` is set and the default
      platform path doesn't exist.

### Affected commands

Every CLI command that uses the default path constants, including but not limited to:

- `taskclf tray` (`--data-dir`, `--models-dir`, `--retrain-config`)
- `taskclf ui` (`--data-dir`, `--model-dir`)
- `taskclf train *` (`--models-dir`, `--data-dir`)
- `taskclf ingest *` (`--data-dir`)
- `taskclf features *` (`--data-dir`)
- `taskclf report *` (`--out-dir`)
