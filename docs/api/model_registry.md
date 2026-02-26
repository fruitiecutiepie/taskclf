# model_registry

Model registry: scan, validate, rank, and filter model bundles.

Provides a pure, testable API for discovering promoted model bundles,
checking compatibility with the current schema and label set, and
ranking candidates by the selection policy.

## BundleMetrics Fields

| Field | Type | Description |
|-------|------|-------------|
| `macro_f1` | float | Macro-averaged F1 across all classes |
| `weighted_f1` | float | Support-weighted F1 across all classes |
| `confusion_matrix` | list[list[int]] | NxN confusion matrix |
| `label_names` | list[str] | Label ordering for confusion matrix rows/cols |

## ModelBundle Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | Bundle directory name |
| `path` | Path | Absolute path to the bundle directory |
| `valid` | bool | `False` if parsing or validation failed |
| `invalid_reason` | str or None | Why the bundle is invalid (if applicable) |
| `metadata` | ModelMetadata or None | Parsed `metadata.json` |
| `metrics` | BundleMetrics or None | Parsed `metrics.json` |
| `created_at` | datetime or None | Parsed `metadata.created_at` timestamp |

## SelectionPolicy Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Policy version (default `1`) |

## ExclusionRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | Bundle directory name |
| `path` | Path | Path to the excluded bundle directory |
| `reason` | str | Human-readable exclusion reason (e.g. `"invalid: missing metrics.json"`, `"incompatible: schema_hash mismatch"`) |

## SelectionReport Fields

| Field | Type | Description |
|-------|------|-------------|
| `best` | ModelBundle or None | Highest-ranked eligible bundle, or `None` if no bundle qualifies |
| `ranked` | list[ModelBundle] | Eligible bundles in score-descending order (best first) |
| `excluded` | list[ExclusionRecord] | Every bundle that was filtered out, with reason |
| `policy` | SelectionPolicy | Policy used for this selection |
| `required_schema_hash` | str | Schema hash that bundles were required to match |

## Functions

- `list_bundles(models_dir)` — scan a directory for bundle subdirectories; returns valid and invalid bundles sorted by `model_id`.
- `is_compatible(bundle, required_schema_hash, required_label_set)` — check schema hash + label set match.
- `passes_constraints(bundle, policy)` — hard constraint gate (policy v1: valid bundle with metrics).
- `score(bundle, policy)` — sortable ranking tuple `(macro_f1, weighted_f1, created_at)`.
- `find_best_model(models_dir, policy, required_schema_hash, required_label_set)` — scan, filter, rank, and select the best bundle; returns a `SelectionReport`.

::: taskclf.model_registry
