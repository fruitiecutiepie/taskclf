# core.inference_policy

Versioned inference-policy artifact: model + calibration + reject
threshold.

## Overview

The `InferencePolicy` is the canonical deployment descriptor for
inference.  It binds a specific model bundle to an optional calibrator
store and a reject threshold that was tuned on the (potentially
calibrated) score distribution.

```
model bundle + calibrator store + reject threshold → InferencePolicy
                                                       ↓
                                          inference_policy.json
                                                       ↓
                                         resolve_inference_config()
                                                       ↓
                                    OnlinePredictor / batch / tray
```

The policy file lives at `models/inference_policy.json` and is the
first thing inference resolution checks.

## Canonical template (repository)

Unlike per-user `config.toml`, `models/inference_policy.json` is **not**
auto-written on first tray startup. The repository holds a stable
checked-in example at
[`configs/inference_policy.template.json`](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/inference_policy.template.json)
that matches `render_default_inference_policy_template_json()` (fixed
`created_at` / `git_commit` for reproducibility). See the
[Inference policy template](../../guide/inference_policy_template.md) guide.

Runtime placeholder files use the same shape via
`write_inference_policy_starter_template()`, with live provenance and
`_help.paths_are_relative_to` set to your install’s TASKCLF_HOME.

## Editing from the tray

The tray menu **Edit Inference Policy** opens this file in the default
editor. If the file is missing, the tray creates it first. When a model
bundle can be resolved, the starter policy is seeded from that bundle's
`metadata.json`, reusing its advisory `reject_threshold` and attaching a
matching calibrator store when `artifacts/**/store.json` explicitly
binds to the same model/schema. When no model can be resolved, the tray
writes a placeholder starter with an inline `_help` block showing the
preferred `taskclf policy create` commands and path conventions. Prefer
`taskclf policy create` for a validated policy. Hand-edited JSON that
fails validation is ignored by `load_inference_policy`, so inference
falls back to `active.json` resolution until the file is fixed.

## Resolution precedence

When inference starts:

1. Explicit `--model-dir` CLI override — bypasses policy entirely.
2. `models/inference_policy.json` — canonical deployment descriptor.
3. `models/active.json` + code defaults — deprecated legacy fallback.
4. Best-model selection + code defaults — no-config fallback.

## InferencePolicy fields

| Field | Type | Description |
|-------|------|-------------|
| `policy_version` | `Literal["v1"]` | Schema version of the policy |
| `model_dir` | `str` | Path to model bundle, relative to `models_dir.parent` |
| `model_schema_hash` | `str` | Must match the bundle's `metadata.schema_hash` |
| `model_label_set` | `list[str]` | Must match the bundle's `metadata.label_set` |
| `calibrator_store_dir` | `str \| None` | Path to calibrator store, relative to `models_dir.parent` |
| `calibration_method` | `str \| None` | `"temperature"` or `"isotonic"` |
| `reject_threshold` | `float` | Tuned on calibrated scores when calibrator is present |
| `created_at` | `str` | ISO-8601 timestamp |
| `source` | `str` | `"manual"`, `"tune-reject"`, `"retrain"`, `"calibrate"`, `"tray-edit"`, `"tray-template"`, etc. |
| `git_commit` | `str` | Git commit SHA at creation time |

## Functions

### build_inference_policy

```python
build_inference_policy(
    *,
    model_dir: str,
    model_schema_hash: str,
    model_label_set: list[str],
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    calibrator_store_dir: str | None = None,
    calibration_method: str | None = None,
    source: str = "manual",
) -> InferencePolicy
```

Convenience builder that fills in `created_at` and `git_commit`
automatically.

### save_inference_policy

```python
save_inference_policy(policy: InferencePolicy, models_dir: Path) -> Path
```

Atomically persist the policy as `models_dir/inference_policy.json`.
Uses temp-file + `os.replace` so readers never see a partial write.

### load_inference_policy

```python
load_inference_policy(models_dir: Path) -> InferencePolicy | None
```

Read the policy file.  Returns `None` on missing, invalid JSON, or
validation failure (no exception raised).

### remove_inference_policy

```python
remove_inference_policy(models_dir: Path) -> bool
```

Delete the policy file.  Returns `True` if removed, `False` if it
did not exist.

### render_default_inference_policy_template_json

```python
render_default_inference_policy_template_json() -> str
```

Return the canonical JSON text for `configs/inference_policy.template.json`
(stable timestamps for the checked-in file).

### write_inference_policy_starter_template

```python
write_inference_policy_starter_template(models_dir: Path) -> Path
```

Atomically write a placeholder `inference_policy.json` for manual editing
(same shape as the canonical template, with live `created_at` /
`git_commit` and `_help.paths_are_relative_to`).

### validate_policy

```python
validate_policy(policy: InferencePolicy, models_dir: Path) -> None
```

Validate that the policy's references resolve to compatible artifacts
on disk.  Raises `PolicyValidationError` on failure.

Checks:

1. `model_dir` exists with `metadata.json`.
2. `model_schema_hash` matches the bundle.
3. `model_label_set` matches the bundle.
4. `calibrator_store_dir` (if set) exists with `store.json`.
5. Calibrator store's `model_schema_hash` (if set) matches the policy.

## Usage

### After tuning

```bash
taskclf train tune-reject \
  --model-dir models/run_001 \
  --calibrator-store artifacts/calibrator_store \
  --from 2026-01-01 --to 2026-01-31 \
  --write-policy
```

### Manual creation

```bash
taskclf policy create \
  --model-dir models/run_001 \
  --calibrator-store artifacts/calibrator_store \
  --reject-threshold 0.55
```

### Inspect / remove

```bash
taskclf policy show
taskclf policy remove
```

::: taskclf.core.inference_policy
