# Personalization Guide

Version: 1.0
Status: Stable
Last Updated: 2026-02-24

This document describes the personalization strategy for taskclf:
how the global classifier adapts to individual users without retraining
or changing the core label set.

---

# 1. Approach

Personalization uses two complementary mechanisms:

1. **user_id as a categorical feature** — LightGBM learns per-user
   behavioural patterns directly during training.
2. **Per-user probability calibration** — post-prediction calibrators
   adjust the raw probability distribution to better reflect each
   user's label distribution.

Neither mechanism changes the core label set or requires label explosion.

---

# 2. user_id as a Model Feature

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

Per-user calibration is only enabled when a user has enough labeled data:

| Criterion | Threshold |
|-----------|-----------|
| Labeled windows | >= 200 |
| Distinct calendar days | >= 3 |
| Distinct core labels observed | >= 3 |

These thresholds are defined in `docs/guide/acceptance.md` Section 8
and coded in `src/taskclf/core/defaults.py`.

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

# 2. Fit calibrators
taskclf train calibrate \
  --model-dir models/<run_id> \
  --from 2026-01-01 --to 2026-02-01 \
  --method temperature \
  --out artifacts/calibrator_store
```

## 6.2 Batch Inference

```bash
taskclf infer batch \
  --model-dir models/<run_id> \
  --from 2026-02-01 --to 2026-02-07 \
  --calibrator-store artifacts/calibrator_store
```

## 6.3 Online Inference

```bash
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
