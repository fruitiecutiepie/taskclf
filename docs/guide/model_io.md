# Model Input / Output Contract v1

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines the exact interface contract between:

- Feature pipeline
- Model inference layer
- Calibration layer
- Personalization layer
- Aggregation layer
- UI layer

All components MUST adhere to this contract.

---

# 1. Model Input Contract

## 1.1 Input Type

The model consumes exactly one window feature row at a time.

Input must conform to:

```

schema/features_v1.json

```id="9kfj3m"

Required fields:

- All numerical features
- All categorical features
- `user_id`
- `bucket_start_ts`

No additional fields may be required by the model.

---

## 1.2 Preprocessing Rules

Before inference:

- Missing numeric values must be imputed as:
  - 0 for interaction rates
- Boolean values must be encoded consistently
- Categorical values must use the same encoding as training

No feature scaling is required (LightGBM handles raw numeric values).

Feature ordering must match training order.

---

# 2. Model Output Contract

## 2.1 Raw Model Output

The global model produces:

```

core_probs: float[8]

```

Where:

- Length = 8
- Ordered by label ID
- Sum(core_probs) = 1.0

Label ordering must match:

```

schema/labels_v1.json

```

---

## 2.2 Core Prediction

Derived from raw output:

```

core_label_id = argmax(core_probs)
core_label_name = labels[core_label_id]
confidence = max(core_probs)

```id="ks3v0m"

Confidence definition:

```

confidence = max(core_probs)

```

Entropy may optionally be logged but is not part of required interface.

---

# 3. Reject Policy

After raw prediction:

```

if confidence < REJECT_THRESHOLD:
is_rejected = true
else:
is_rejected = false

```

If rejected:

- core_label is still computed
- But final mapped_label becomes `Mixed/Unknown`
- Window is logged for potential relabeling

Reject threshold is configurable and versioned separately.

---

# 4. Calibration Layer

If per-user calibration is enabled:

1. Raw core_probs
2. Apply user calibrator (if exists)
3. Produce calibrated_probs

Calibrator must:

- Preserve ordering
- Preserve probability sum = 1.0
- Not change label count

If no user calibrator exists:

- Use global calibration (optional)
- Or use raw probabilities

---

# 5. Personalization Mapping Layer

Mapping takes:

```

core_label_id
core_probs
user_mapping_config

```

Outputs:

```

mapped_label_name
mapped_probs

```id="c9xw2a"

Rules:

- mapped_probs must sum to 1.0
- If multiple core labels map to same user bucket:
  - mapped_prob[bucket] = sum(core_probs of mapped labels)
- If rejected:
  - mapped_label = "Mixed/Unknown"

Mapping must not alter core_probs.

---

# 6. Final Inference Output Object

Each window must produce:

```

{
"user_id": string,
"bucket_start_ts": timestamp,
"core_label_id": int,
"core_label_name": string,
"core_probs": float[8],
"confidence": float,
"is_rejected": boolean,
"mapped_label_name": string,
"mapped_probs": { string: float },
"model_version": string,
"schema_version": "features_v1",
"label_version": "labels_v1"
}

```

This structure is stable and versioned.

---

# 7. Determinism Requirement

Given:

- Same model artifact
- Same calibrator
- Same mapping config
- Same input feature row

Inference must produce identical output.

No stochastic inference allowed.

---

# 8. Error Handling

If:

- Missing required feature
- Unknown categorical value
- Schema mismatch

Then:

- Log error
- Mark window as `Mixed/Unknown`
- Do NOT crash inference pipeline

---

# 9. Versioning

Changing any of the following requires version bump:

- Label ordering
- Output structure
- Confidence definition
- Reject semantics
- Calibration insertion point
- Mapping semantics
