# Inference Pipeline Overview

Version: 1.1
Status: Stable
Last Updated: 2026-03-03

End-to-end reference for how a feature row becomes a labeled prediction.
Each stage links to its authoritative documentation.

---

## Pipeline stages

```
Feature Row → Model → core_probs → Calibration → Reject Check → Taxonomy Mapping → Final Output
```

| Stage | What happens | Details |
|-------|-------------|---------|
| **1. Input** | A single feature row enters the model. Must conform to the [Feature Schema](../api/core/schema.md) and include all columns from `FEATURE_COLUMNS`. | [Core Types](../api/core/types.md), [Configuration](../api/core/defaults.md) |
| **2. Model prediction** | LightGBM produces `core_probs: float[8]`, ordered by label ID, summing to 1.0. | [Task Labels (v1)](labels_v1.md), [LightGBM Trainer](../api/train/lgbm.md) |
| **3. Calibration** | Per-user or global calibrator adjusts probabilities. Preserves ordering and sum constraint. | [Personalization](personalization.md) Sections 3–5, [Calibration API](../api/infer/calibration.md) |
| **4. Reject check** | If `max(calibrated_probs) < reject_threshold`, the prediction is rejected and mapped to `Mixed/Unknown`. | [Acceptance Criteria](acceptance.md) Section 4 |
| **5. Smoothing** | Rolling majority window merges noisy per-minute predictions into stable blocks. | [Prediction Smoothing](../api/infer/smooth.md) |
| **6. Taxonomy mapping** | Optional: maps core labels to user-defined buckets with aggregated probabilities. | [Custom Taxonomy](taxonomy.md) |

---

## Final output object

Each prediction produces:

| Field | Type | Source |
|-------|------|--------|
| `user_id` | string | Feature row |
| `bucket_start_ts` | timestamp | Feature row |
| `core_label_id` | int | `argmax(core_probs)` |
| `core_label_name` | string | Label set lookup |
| `core_probs` | float[8] | Model output |
| `confidence` | float | `max(core_probs)` |
| `is_rejected` | boolean | Reject check |
| `mapped_label_name` | string | Taxonomy mapping (or core label) |
| `mapped_probs` | dict[str, float] | Taxonomy mapping |
| `model_version` | string | [Model Bundle](model_bundle_layout.md) metadata |
| `schema_version` | string | Feature schema |
| `label_version` | string | Label schema |

See [Prediction Types](../api/infer/prediction.md) for the implementation.

---

## Invariants

- **Determinism**: same model + calibrator + mapping + input = identical output. No stochastic inference.
- **Error handling**: missing features, unknown categoricals, or schema mismatches produce `Mixed/Unknown` — the pipeline never crashes. See [Data Validation](../api/core/validation.md).
- **Privacy**: no raw titles, keystrokes, or URLs enter or leave the pipeline. See [Privacy Model](privacy.md).

---

## Versioning

Changes to any of the following require a version bump:

- Label ordering or count
- Output structure
- Confidence definition
- Reject semantics
- Calibration insertion point
- Taxonomy mapping semantics
