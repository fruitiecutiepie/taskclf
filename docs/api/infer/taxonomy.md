# infer.taxonomy

User-specific taxonomy mapping: core labels to user-defined buckets.

## Overview

The taxonomy layer sits between the model's core 8-class predictions
and the user-facing display.  It maps one or more core labels into
user-defined **buckets** via a YAML config, without altering the
underlying core predictions.

```
core label + core_probs → TaxonomyResolver → mapped_label + mapped_probs
```

See the [taxonomy guide](../../guide/taxonomy.md) and
`configs/user_taxonomy_example.yaml` for configuration details.

## Config model hierarchy

### TaxonomyConfig

Top-level config loaded from YAML.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | `str` | `"1.0"` | Config schema version |
| `label_schema_version` | `str` | `"labels_v1"` | Expected label schema |
| `user_id` | `str \| None` | `None` | Optional user scope |
| `display` | `TaxonomyDisplay` | *(defaults)* | Display preferences |
| `reject` | `TaxonomyReject` | *(defaults)* | Rejection display settings |
| `buckets` | `list[TaxonomyBucket]` | *(required)* | At least one bucket |
| `advanced` | `TaxonomyAdvanced` | *(defaults)* | Tuning knobs |

### TaxonomyBucket

A user-facing task category that aggregates one or more core labels.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique display name |
| `description` | `str` | Human-readable description |
| `core_labels` | `list[str]` | Core labels mapped to this bucket (must be valid `LABEL_SET_V1` entries) |
| `color` | `str` | Hex color for display (`#RRGGBB`) |

### TaxonomyDisplay

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `show_core_labels` | `bool` | `False` | Show underlying core labels in UI |
| `default_view` | `"mapped" \| "core"` | `"mapped"` | Default view mode |
| `color_theme` | `str` | `"default"` | Color theme name |

### TaxonomyReject

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mixed_label_name` | `str` | `"Mixed/Unknown"` | Label shown for rejected predictions |
| `include_rejected_in_reports` | `bool` | `False` | Include rejected buckets in reports |

### TaxonomyAdvanced

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `probability_aggregation` | `"sum" \| "max"` | `"sum"` | How core-label probs are combined per bucket |
| `min_confidence_for_mapping` | `float` | `0.55` | Minimum confidence for mapping |
| `reweight_core_labels` | `dict[str, float]` | `{}` | Per-label probability multipliers |

## TaxonomyResolver

Stateless mapper from core predictions to user-defined buckets.
Precomputes index lookups at construction time for fast per-row
resolution.

```python
from pathlib import Path
from taskclf.infer.taxonomy import load_taxonomy, TaxonomyResolver

config = load_taxonomy(Path("configs/user_taxonomy.yaml"))
resolver = TaxonomyResolver(config)
result = resolver.resolve(core_label_id, core_probs)
print(result.mapped_label, result.mapped_probs)
```

`resolve_batch` maps an entire batch at once:

```python
results = resolver.resolve_batch(pred_indices, proba_matrix)
mapped_labels = [r.mapped_label for r in results]
```

## Aggregation modes

When a bucket contains multiple core labels, their probabilities are
combined using the configured aggregation mode:

- **`sum`** (default) -- probabilities are summed, then the full
  vector is renormalized.
- **`max`** -- the maximum probability among the bucket's core labels
  is used, then renormalized.

## Fallback bucket

Core labels not assigned to any user bucket are automatically collected
into an `"Other"` fallback bucket.  A log message lists the unmapped
labels when this occurs.

## Reweighting

`advanced.reweight_core_labels` allows adjusting core-label
probabilities before mapping.  Each entry is a `label: weight`
multiplier applied to the probability vector, which is then
renormalized.  This can bias the mapping toward or away from
specific core labels without retraining.

## I/O helpers

- `load_taxonomy(path)` -- load and validate a YAML config.
- `save_taxonomy(config, path)` -- serialize a config to YAML.
- `default_taxonomy()` -- create an identity mapping (one bucket per
  core label) as a starting point for customisation.

::: taskclf.infer.taxonomy
