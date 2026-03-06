# core.paths

Platform-aware base directory resolution for taskclf.

## TASKCLF_HOME resolution order

All data, model, and artifact paths are anchored to a single root
directory resolved by `taskclf_home()`:

| Priority | Source | Example |
|---|---|---|
| 1 | `TASKCLF_HOME` env var | `export TASKCLF_HOME=~/my-taskclf` |
| 2 | macOS default | `~/Library/Application Support/taskclf/` |
| 3 | XDG default (Linux) | `~/.local/share/taskclf/` |
| 4 | Windows default | `%LOCALAPPDATA%/taskclf/` |

## Directory layout

```text
<TASKCLF_HOME>/
├── data/
│   ├── raw/aw/
│   └── processed/
├── models/
├── artifacts/
│   └── telemetry/
└── configs/
```

## API reference

::: taskclf.core.paths
    options:
      show_if_no_docstring: true
