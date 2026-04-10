# Personalization Guide

Version: 2.0
Status: Stable
Last Updated: 2026-03-28

This document describes the personalization strategy for taskclf:
how the global classifier adapts to individual users without retraining
or changing the core label set.

---

# 1. Approach

Personalization uses two complementary mechanisms. Historically both were
active in the default schema, but the current default (`v3`) follows the
schema-v2 model semantics described later in this guide.

1. **user_id as a categorical feature** — legacy v1 behavior where
   LightGBM learns per-user behavioural patterns directly during training.
2. **Per-user probability calibration** — post-prediction calibrators
   adjust the raw probability distribution to better reflect each
   user's label distribution.

Neither mechanism changes the core label set or requires label explosion.

---

# 2. user_id as a Model Feature (legacy v1)

`user_id` is included in `FEATURE_COLUMNS` as a categorical feature.
LightGBM treats it as a native categorical, learning splits on user
identity alongside other features (app, typing rate, etc.).

For unseen users at inference time, the categorical encoder maps the
unknown user_id to a reserved `-1` code, ensuring graceful degradation.

Privacy note: `user_id` must be a random UUID — never an email, name,
or machine identifier. The model artifact stores the encoded vocabulary,
not raw identifiers.

---

# 3. Calibration Methods

Two calibration methods are available:

## 3.1 Temperature Scaling

A single scalar parameter T applied to the model's log-probabilities:

```
calibrated = softmax(logits / T)
```

- T > 1 softens (less confident)
- T < 1 sharpens (more confident)
- T = 1 is identity (no change)

The optimal T is found by grid search minimising NLL on validation data.

Use temperature scaling when data is limited or when simplicity is
preferred.

## 3.2 Isotonic Regression

Per-class monotonic mapping from raw probability to calibrated
probability. Uses sklearn's `IsotonicRegression` with `out_of_bounds="clip"`.

After per-class transformation, probabilities are renormalized to
sum to 1.0.

Use isotonic regression when there is sufficient validation data
(recommended: >= 200 samples) and when the model's probability
estimates show per-class miscalibration.

---

# 4. Eligibility Thresholds

Per-user calibration is only enabled when a user has enough labeled
data.  Exact thresholds are defined in
[Acceptance Criteria](acceptance.md) Section 8 and coded in
`src/taskclf/core/defaults.py`.

Users who do not meet these thresholds fall back to the global calibrator.

---

# 5. CalibratorStore

The `CalibratorStore` manages the mapping from user_id to calibrator:

- **global_calibrator** — fitted on all users' validation data.
  Applied to any user without a dedicated calibrator.
- **user_calibrators** — dict of user_id to per-user calibrator,
  only for users who pass the eligibility check.

The store is serialized as a directory:

```
calibrator_store/
  store.json        # metadata: method, user count, user IDs
  global.json       # serialized global calibrator
  users/
    <user_id>.json  # serialized per-user calibrator
```

---

# 6. Pipeline Integration

## 6.1 Training-Side

```bash
# 1. Train the model (user_id is now a feature)
taskclf train lgbm --from 2026-01-01 --to 2026-02-01

# 2. Fit calibrators (now records model binding in store.json)
taskclf train calibrate \
  --model-dir models/<run_id> \
  --from 2026-01-01 --to 2026-02-01 \
  --method temperature \
  --out artifacts/calibrator_store

# 3. Tune reject threshold and write inference policy
taskclf train tune-reject \
  --model-dir models/<run_id> \
  --calibrator-store artifacts/calibrator_store \
  --from 2026-01-01 --to 2026-02-01 \
  --write-policy
```

Step 3 creates `models/inference_policy.json` which binds the model
bundle, calibrator store, and tuned reject threshold.  This policy
is the canonical deployment descriptor for inference.

## 6.2 Batch Inference

```bash
# Uses inference policy automatically (model + calibrator + threshold)
taskclf infer batch --from 2026-02-01 --to 2026-02-07

# Or with explicit overrides (backward compatible)
taskclf infer batch \
  --model-dir models/<run_id> \
  --from 2026-02-01 --to 2026-02-07 \
  --calibrator-store artifacts/calibrator_store
```

## 6.3 Online Inference

```bash
# Uses inference policy automatically; hot-reloads on policy change
taskclf infer online

# Or with explicit overrides (backward compatible)
taskclf infer online \
  --model-dir models/<run_id> \
  --calibrator-store artifacts/calibrator_store
```

---

# 7. Inference Flow

Per window, the pipeline is:

1. Raw model probabilities
2. Per-user calibration (via CalibratorStore lookup)
3. Reject decision (max prob < threshold)
4. Rolling majority smoothing
5. Taxonomy mapping (optional)

Calibration happens before the reject decision so that calibrated
confidence values drive rejection.

---

# 8. Determinism

Given the same model, calibrator store, and input feature row,
inference must produce identical output. Calibrators are deterministic.

---

# 9. Versioning

Changes to any of the following require a version bump:

- Eligibility thresholds
- Calibration insertion point in the pipeline
- Store serialization format
- Default calibration method

---

# 10. Migration Boundary

This section records the long-term direction for personalization and
the compatibility gate that prevents mixing old and new contracts.

## Schema-v1 bundles (legacy)

`user_id` serves two roles at inference time:

1. **Model feature** — `user_id` is in `FEATURE_COLUMNS` and
   `CATEGORICAL_COLUMNS` (in `train/lgbm.py`). LightGBM learns
   per-user splits directly. An incorrect or default `user_id`
   produces wrong encoded values and degrades predictions.
2. **Calibrator key** — `CalibratorStore.get_calibrator(user_id)`
   selects the per-user calibrator (or falls back to global). A
   wrong `user_id` routes to the wrong calibrator.

Both roles must receive the correct `user_id` for predictions to be
accurate.

## Schema-v2 bundles (implemented, and used by the current v3 default model semantics)

`user_id` has been removed from the core model's feature vector in
schema v2.  Personalization now relies entirely on:

- Per-user calibrators (temperature scaling or isotonic regression).
- Per-user reject thresholds (loaded from
  `InferencePolicy.per_user_reject_thresholds`).
- User-specific post-processing (taxonomy overrides).

The core model becomes user-agnostic, which simplifies training,
reduces overfitting to high-volume users, and allows new users to
receive calibrated predictions without retraining.

### Training a v2 model

```bash
taskclf train lgbm \
  --from 2026-01-01 --to 2026-02-01 \
  --schema-version v2
```

Or programmatically:

```python
from taskclf.train.lgbm import train_lgbm
from taskclf.core.model_io import build_metadata

model, metrics, cm_df, params, cat_encoders = train_lgbm(
    train_df, val_df,
    schema_version="v2",
)
metadata = build_metadata(
    label_set=metrics["label_names"],
    ...,
    schema_version="v2",
)
```

### Key differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| `user_id` in features | Yes (34 features) | No (33 features) |
| `FEATURE_COLUMNS` | includes `user_id` | `FEATURE_COLUMNS_V2` excludes it |
| `CATEGORICAL_COLUMNS` | 4 columns | `CATEGORICAL_COLUMNS_V2`: 3 columns |
| Personalization | LightGBM learns user splits | Calibrators + per-user thresholds |
| Schema hash | `FeatureSchemaV1.SCHEMA_HASH` | `FeatureSchemaV2.SCHEMA_HASH` |

### Per-user reject thresholds

The `InferencePolicy` supports an optional
`per_user_reject_thresholds` dict that maps user IDs to individual
reject thresholds.  When present, `OnlinePredictor.predict_bucket()`
applies the per-user threshold instead of the global one.  Users not
in the dict fall back to the global threshold.

## Gate

The schema hash (computed from the column registry in
`core/schema.py`) differs between v1 and v2 because v1 includes
`user_id` and v2 does not.  `load_model_bundle` validates the
bundle's `schema_hash` against the expected hash for its declared
`schema_version`, refusing to load a bundle whose hash does not match.

This prevents accidentally loading a v1 model with v2 features (or
vice versa).  Both v1 and v2 bundles can coexist in the same
`models/` directory; the hash gate routes each bundle to the correct
feature contract.

### Migration steps (completed)

1. `_COLUMNS_V2` and `FeatureSchemaV2` defined in `core/schema.py`.
2. `FEATURE_COLUMNS_V2` and `CATEGORICAL_COLUMNS_V2` defined in `train/lgbm.py`.
3. `train_lgbm`, `prepare_xy`, and `encode_categoricals` accept `schema_version`.
4. `build_metadata` and `load_model_bundle` support both v1 and v2 schema versions.
5. Old v1 bundles continue to work unchanged.
