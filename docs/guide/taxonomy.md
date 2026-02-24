# User-Specific Taxonomy Mapping

Version: 1.0
Status: Stable
Last Updated: 2026-02-24

This document explains how users can map core classifier labels to
their own task categories without retraining the model.

---

## 1. Overview

The global classifier always predicts one of 8 core labels (see
`docs/guide/labels_v1.md`).  Different users may want to see different
categories in their time reports.  The taxonomy mapping layer sits
**after** model prediction and converts core labels + probability
vectors into user-defined buckets with aggregated probabilities.

Key properties:

- Core predictions are never modified.
- Mapping is deterministic given the same config and input.
- Multiple core labels can map to the same user bucket (many-to-one).
- Unmapped core labels fall into an automatic "Other" bucket.
- Rejected predictions remain `Mixed/Unknown` regardless of mapping.

---

## 2. Config Format

Taxonomy configs are YAML files.  See `configs/user_taxonomy_example.yaml`
for a complete example.

### Minimal example

```yaml
version: "1.0"
label_schema_version: labels_v1

buckets:
  - name: "Deep Work"
    core_labels: [Build, Debug, Review, Write]
    color: "#2E86DE"

  - name: "Research"
    core_labels: [ReadResearch]
    color: "#9B59B6"

  - name: "Communication"
    core_labels: [Communicate, Meet]
    color: "#27AE60"

  - name: "Break"
    core_labels: [BreakIdle]
    color: "#7F8C8D"
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `version` | Yes | Config format version (currently `"1.0"`) |
| `label_schema_version` | No | Must be `labels_v1` |
| `user_id` | No | Ties config to a specific user |
| `display` | No | Display preferences (show_core_labels, default_view) |
| `reject` | No | Reject label name, whether to include in reports |
| `buckets` | Yes | List of user-facing categories (at least one) |
| `advanced` | No | Aggregation mode, reweighting, min confidence |

### Bucket fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique display name |
| `description` | No | Human-readable description |
| `core_labels` | Yes | List of core label names (at least one) |
| `color` | No | Hex color `#RRGGBB` (default: `#808080`) |

---

## 3. Validation Rules

The following rules are enforced when loading a taxonomy config:

1. Every `core_label` in a bucket must be a valid `CoreLabel` member.
2. Bucket names must be unique.
3. Colors must be valid hex (`#RRGGBB`).
4. Reweight keys must be valid `CoreLabel` values with weight > 0.
5. At least one bucket is required.

If a core label is not in any bucket, it is automatically assigned to
an implicit "Other" fallback bucket.

If validation fails, the system falls back to core labels only.

---

## 4. Probability Aggregation

When multiple core labels map to one bucket, their probabilities are
combined.

### Sum mode (default)

```
bucket_prob = sum(core_probs[i] for i in bucket.core_labels)
```

Example: if "Deep Work" maps Build + Debug + Write, and their core
probabilities are 0.4 + 0.1 + 0.2, then `bucket_prob = 0.7`.

### Max mode

```
bucket_prob = max(core_probs[i] for i in bucket.core_labels)
```

Set in the config:

```yaml
advanced:
  probability_aggregation: "max"
```

After aggregation, bucket probabilities are normalized to sum to 1.0.

---

## 5. Reweighting

Advanced users can adjust the contribution of individual core labels
without retraining:

```yaml
advanced:
  reweight_core_labels:
    Build: 2.0
    BreakIdle: 0.5
```

Reweights are multiplied into `core_probs` before aggregation, then
re-normalized.  This lets users bias the mapping toward or away from
specific activities.

---

## 6. CLI Commands

### Validate a config

```bash
taskclf taxonomy validate --config configs/user_taxonomy.yaml
```

### Show the mapping table

```bash
taskclf taxonomy show --config configs/user_taxonomy.yaml
```

### Generate a default config

```bash
taskclf taxonomy init --out configs/user_taxonomy.yaml
```

This creates an identity mapping (one bucket per core label) as a
starting point for customisation.

### Use with batch inference

```bash
taskclf infer batch \
  --model-dir models/run_dir \
  --from 2025-06-10 --to 2025-06-15 \
  --taxonomy configs/user_taxonomy.yaml
```

Adds a `mapped_label` column to `predictions.csv`.

### Use with online inference

```bash
taskclf infer online \
  --model-dir models/run_dir \
  --taxonomy configs/user_taxonomy.yaml
```

---

## 7. Integration Points

The taxonomy mapping layer fits into the inference pipeline as follows:

```
Model -> core_probs -> Reject check -> Taxonomy mapping -> mapped_label
```

- **Batch inference**: `run_batch_inference()` accepts an optional
  `taxonomy` parameter.  When set, `BatchInferenceResult` includes
  `mapped_labels` and `mapped_probs`.
- **Online inference**: `OnlinePredictor` accepts an optional
  `taxonomy` parameter.  `predict_bucket()` returns the mapped label
  when set.
- Core label smoothing always operates on core labels; taxonomy
  mapping is applied afterwards.

---

## 8. Versioning

The taxonomy config has its own `version` field.  Changes to:

- Bucket definitions
- Aggregation semantics
- Reweight behavior

should bump the version.  Taxonomy configs are user-editable and
should be versioned alongside other project configs.
