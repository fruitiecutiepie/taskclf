# infer.resolve

Model resolution for inference: map CLI arguments to a concrete model
bundle directory, with automatic fallback and hot-reload support.

## Overview

The resolve module bridges the gap between the CLI `--model-dir` option
and the [model registry](../model_registry.md).  At inference startup
the caller may or may not specify an explicit model directory.
`resolve_model_dir` applies a deterministic precedence chain to
guarantee either a valid bundle path or a descriptive error:

```
explicit --model-dir ──► validate path exists ──► return
        │ (None)
        ▼
  active.json exists? ──► yes ──► use active pointer
        │ (no / stale)
        ▼
  find_best_model() ──► best eligible bundle ──► self-heal active.json ──► return
        │ (none eligible)
        ▼
  raise ModelResolutionError (with exclusion reasons)
```

For long-running online inference loops, `ActiveModelReloader` watches
`active.json` for mtime changes and transparently reloads the model
bundle without restarting the process.

## Resolution precedence

| Priority | Condition | Behaviour |
|----------|-----------|-----------|
| 1 | `model_dir` argument provided | Validate the path exists; return it directly |
| 2 | `models/active.json` present and valid | Use the active pointer from the registry |
| 3 | No active pointer but eligible bundles exist | Select best by `macro_f1`; self-heal `active.json` |
| 4 | No eligible bundles | Raise `ModelResolutionError` with per-bundle exclusion reasons |

## ModelResolutionError

Custom exception raised when no model can be resolved.

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | str | Human-readable error with actionable guidance |
| `report` | `SelectionReport` or `None` | Full selection report including `excluded` records with per-bundle reasons (attached when resolution went through the registry) |

## `resolve_model_dir`

```python
def resolve_model_dir(
    model_dir: str | None,
    models_dir: Path,
    policy: SelectionPolicy | None = None,
) -> Path
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | `str` or `None` | — | Explicit `--model-dir` value from CLI; `None` triggers automatic resolution |
| `models_dir` | `Path` | — | Base directory containing promoted model bundles |
| `policy` | `SelectionPolicy` or `None` | `None` | Selection policy override; when `None`, the registry uses policy v1 |

**Returns:** `Path` to the resolved model bundle directory.

**Raises:** `ModelResolutionError` when no model can be resolved.  The
error message includes the list of excluded bundles and their exclusion
reasons when available.

## `ActiveModelReloader`

Lightweight mtime-based watcher designed for the online inference loop.
Polls `active.json` at a configurable interval and, when a change is
detected, loads the new model bundle via `load_model_bundle`.  On
failure the current model is kept and a warning is logged.

### Constructor

```python
ActiveModelReloader(
    models_dir: Path,
    check_interval_s: float = 60.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models_dir` | `Path` | — | Directory containing `active.json` |
| `check_interval_s` | `float` | `60.0` | Minimum seconds between mtime checks; prevents excessive `stat` calls |

### `check_reload`

```python
def check_reload(self) -> tuple[Booster, ModelMetadata, dict[str, LabelEncoder] | None] | None
```

Returns `(model, metadata, cat_encoders)` when a reload succeeds.
Returns `None` in three cases:

1. The check interval has not elapsed since the last check.
2. The `active.json` mtime is unchanged.
3. The mtime changed but the reload failed (a warning is logged and the
   caller should keep the current model).

After a successful reload the internal mtime is updated, so a second
immediate call returns `None`.

## Usage

### Resolve at startup

```python
from pathlib import Path
from taskclf.infer.resolve import resolve_model_dir, ModelResolutionError

try:
    bundle_path = resolve_model_dir(
        model_dir=None,           # let the registry decide
        models_dir=Path("models/"),
    )
except ModelResolutionError as exc:
    print(exc)
    if exc.report and exc.report.excluded:
        for rec in exc.report.excluded:
            print(f"  {rec.model_id}: {rec.reason}")
    raise SystemExit(1)
```

### Hot-reload in an online loop

```python
from pathlib import Path
from taskclf.infer.resolve import ActiveModelReloader

reloader = ActiveModelReloader(Path("models/"), check_interval_s=30.0)

# inside the polling loop
result = reloader.check_reload()
if result is not None:
    model, metadata, cat_encoders = result
    # swap to the new model for subsequent predictions
```

## See also

- [`model_registry`](../model_registry.md) — bundle scanning, ranking,
  and active pointer management
- [`core.model_io`](../core/model_io.md) — `load_model_bundle` and
  `ModelMetadata`
- [`infer.online`](online.md) — online inference loop that uses
  `resolve_model_dir` and `ActiveModelReloader`

::: taskclf.infer.resolve
