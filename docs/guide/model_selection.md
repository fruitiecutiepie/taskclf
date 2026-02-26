# Model selection

This document defines how TaskCLF selects a “default” model for inference (when `--model-dir` is omitted), how to rank candidate models, and what “compatible” means.

## Terms

- **Model bundle**: a directory containing `model.txt`, `metadata.json`, `metrics.json`, and other bundle files (see `docs/model_bundle_layout.md`).
- **Promoted model**: a model bundle that lives under `models/` (the `models_dir`).
- **Rejected model**: a model bundle moved to `<out_dir>/rejected_models/` by retrain.
- **Compatible model**: a promoted bundle that matches the *current* runtime schema and label set (defined below).
- **Rankable model**: a promoted + compatible bundle that has the files required to compute ranking metrics (always includes `metrics.json` today).
- **Default model**: the model used by inference when `--model-dir` is not provided.

## Compatibility (hard requirement)

A model bundle is **compatible** with the current code if BOTH hold:

1. `metadata.schema_hash == FeatureSchemaV1.SCHEMA_HASH` (exact match)
2. `sorted(metadata.label_set) == sorted(LABEL_SET_V1)` (exact match)

This matches the behavior of `load_model_bundle(..., validate_schema=True, validate_labels=True)`.

If a bundle is not compatible, it MUST NOT be used as the default model (or auto-selected), and tooling should surface a clear reason.

## Current state (today)

- Inference requires explicit `--model-dir` (no auto-selection).
- Retrain uses `find_latest_model()` *only* to pick a “champion” by recency for regression-gate comparison; it does not reflect quality.
- `metrics.json` inside the bundle contains only `macro_f1`, `weighted_f1`, `confusion_matrix`, `label_names`.

Acceptance gates (BreakIdle precision, “no class below 0.50 precision”, etc.) are computed during evaluation and stored in `evaluation.json` written under `--out-dir` (not inside the bundle).

## Goal state (to implement)

### Default resolution precedence

When `--model-dir` is omitted, inference resolves the model directory by the following precedence:

1. If `models/active.json` exists and is valid: use the bundle it points to.
2. Else: select the best bundle from `models/` by the selection policy (below).
3. If no eligible bundle exists: fail loudly with a message listing why bundles were excluded, and require explicit `--model-dir`.

`--model-dir` always overrides the default resolution.

### Why an active pointer file

A pointer file makes “continuous best model” explicit and stable:

- training/promotion updates the pointer when a challenger overthrows the previous best
- inference reads a single small file, rather than re-scanning and re-ranking on every call
- updates can be atomic and auditable

## Selection policy v1 (ranking)

### Candidate set

Start from all promoted bundles under `models/`.

Filter to bundles that are:
- **valid bundles** (have required files)
- **compatible** (schema_hash + label_set match)

### Score and tie-breakers

Rank compatible bundles using:

1. higher `metrics.macro_f1` wins
2. tie-break: higher `metrics.weighted_f1` wins
3. tie-break: newer `metadata.created_at` wins (ISO8601 datetime with UTC offset)

Notes:
- `metadata.created_at` is produced by `datetime.now(UTC).isoformat()` (includes offset).
- `metrics.json` uses floats for F1 metrics.

### Constraints / gates

Policy v1 does NOT re-check acceptance gates by default, because acceptance checks are not stored in bundle `metrics.json` today.

If you want selection to apply gates, you must either:
- (preferred) persist a minimal `acceptance.json` (or equivalent) inside the bundle at promotion time, OR
- teach the selector to locate the corresponding `evaluation.json` in the `--out-dir` history (hard, because it is not colocated with the bundle and not part of the bundle contract).

Until that exists, *promotion-time gating* is the control point:
- retrain already rejects bundles that fail gates (moved to rejected_models)
- selection chooses the best among promoted bundles by F1

## Active pointer file: `models/active.json`

### Purpose

`models/active.json` is a small file that defines the default model bundle for inference when `--model-dir` is omitted.

### Schema

```json
{
  "model_dir": "models/<bundle_dir_name>",
  "model_id": "<optional stable id; may equal bundle dir name>",
  "selected_at": "2026-02-26T12:34:56.789123+00:00",
  "policy_version": 1,
  "reason": {
    "metric": "macro_f1",
    "macro_f1": 0.8123,
    "weighted_f1": 0.9012
  }
}
````

Required keys:

* `model_dir` (string): path to the promoted bundle directory. Prefer a relative path (from repo root or from `models/`), but be consistent.
* `selected_at` (string): ISO8601 datetime string.
* `policy_version` (int): selection policy version.

Optional keys:

* `model_id` (string)
* `reason` (object)

### Validation rules

On read, inference MUST:

* ensure file is valid JSON and required keys exist
* ensure `model_dir` exists and is a valid bundle
* ensure bundle is compatible (schema_hash + label_set match)

If the pointer file is invalid, inference falls back to selecting best-by-policy.

### Atomic update requirement

Updates to `active.json` MUST be atomic to avoid partial reads:

* write to `active.json.tmp`
* `os.replace("active.json.tmp", "active.json")` on the same filesystem

### Audit log (recommended)

Append a line to `models/active_history.jsonl` on every active change:

```json
{"at":"...","old":{...},"new":{...}}
```

This enables rollback and debugging of “why did we switch models?”

## Overthrow flow (continuous best)

When a new model is promoted to `models/`:

1. Scan and rank promoted compatible bundles using policy v1.
2. If the best bundle differs from the currently active bundle:

   * atomically update `models/active.json`
   * append to `models/active_history.jsonl`

In retrain, prefer comparing candidates against the current `active.json` model (if present) rather than “latest by timestamp”.

## CLI expectations (recommended)

* `taskclf train list`:

  * lists promoted bundles
  * shows ranking metrics and compatibility status
  * highlights the active bundle (if pointer exists)
  * supports `--json`

* `taskclf model set-active --model-dir <...>`:

  * writes `active.json` atomically after validation
  * logs to history
