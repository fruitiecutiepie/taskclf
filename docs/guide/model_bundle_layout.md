# Model bundle layout

This document defines the on-disk layout for a model bundle directory used by TaskCLF.

A “model bundle” is the unit that is promoted to `models/` and loaded by inference via `load_model_bundle()`.

## Directories

- Promoted bundles:
  - `models/<bundle_dir>/...`

- Rejected bundles:
  - `<out_dir>/rejected_models/<bundle_dir>/...`
  - In practice `<out_dir>` is typically `artifacts/`, so rejected bundles often live under:
    - `artifacts/rejected_models/<bundle_dir>/...`

There are no other promotion/staging directories today.

## Bundle directory contents

A valid bundle directory contains:

| File | Always? | Purpose |
|---|---|---|
| `model.txt` | yes | LightGBM model artifact in text format. Loaded by `lgb.Booster(model_file=...)`. |
| `metadata.json` | yes | Schema + params + provenance. Parsed as `ModelMetadata`. |
| `metrics.json` | yes | Macro/weighted F1 and confusion matrix. |
| `confusion_matrix.csv` | yes | Confusion matrix as CSV (human-friendly / tooling). |
| `categorical_encoders.json` | conditional | Only present if categorical encoders were provided at save time. |

Important: evaluation artifacts produced by `write_evaluation_artifacts()` (e.g., `evaluation.json`, `calibration.json`, `calibration.png`) are written to the evaluation output directory (`--out-dir`, typically `artifacts/`) and are NOT stored inside the bundle directory.

## Artifact filenames (hard requirements)

- The LightGBM model file is **exactly** `model.txt`.
  - Even if other docs mention `model.bin`, the current code path saves and loads only `model.txt`.

Tooling MUST treat `model.txt` as required for a loadable bundle.

## metadata.json contract (current)

`metadata.json` is a JSON object matching the `ModelMetadata` Pydantic model:

```json
{
  "schema_version": "v1",
  "schema_hash": "<FeatureSchemaV1.SCHEMA_HASH>",
  "label_set": ["BreakIdle", "..."],

  "train_date_from": "YYYY-MM-DD",
  "train_date_to": "YYYY-MM-DD",

  "params": { "learning_rate": 0.05, "...": "..." },

  "git_commit": "<rev-parse HEAD or 'unknown'>",
  "dataset_hash": "<first16hex>",
  "reject_threshold": 0.5,
  "data_provenance": "real",

  "created_at": "2026-02-26T12:34:56.789123+00:00"
}
````

Notes:

* `created_at` is produced by `datetime.now(UTC).isoformat()` and includes UTC offset and microseconds.
* Compatibility checks in `load_model_bundle()` require:

  * `schema_hash` exact match vs `FeatureSchemaV1.SCHEMA_HASH`
  * `label_set` exact match vs `LABEL_SET_V1` (sorted equality)

## metrics.json contract (current)

See `docs/metrics_contract.md`.

## Valid bundle definition

A directory is a valid model bundle if:

* required files exist (`model.txt`, `metadata.json`, `metrics.json`, `confusion_matrix.csv`)
* `metadata.json` and `metrics.json` parse as JSON
* required keys exist and types are correct (per contracts above)

A valid bundle is compatible if:

* `metadata.schema_hash` matches current `FeatureSchemaV1.SCHEMA_HASH`
* `metadata.label_set` matches current `LABEL_SET_V1` (sorted equality)

Selection tooling should distinguish:

* invalid bundle (missing/corrupt files)
* incompatible bundle (schema/labels mismatch)
* valid + compatible bundle (candidate for selection)
