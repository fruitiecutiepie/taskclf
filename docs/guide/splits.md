# Train / Validation / Test Split Policy v1

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines how datasets are split for:

- Global model training
- Evaluation
- Personalization evaluation
- Drift monitoring

The primary goal is to prevent temporal leakage and unrealistic evaluation.

---

# 1. Core Principles

1. No future information may leak into training.
2. Evaluation must simulate real deployment.
3. Users must be evaluated both:
   - As seen users (personalized scenario)
   - As unseen users (generalization scenario)
4. Time ordering must be respected.

---

# 2. Global Model Training Split

## 2.1 Primary Split Strategy

Time-based split per user.

For each user:

- Sort windows by bucket_start_ts.
- Split chronologically:

```

Train: earliest 70%
Validation: next 15%
Test: final 15%

```

No shuffling allowed.

---

## 2.2 Cross-User Evaluation (Generalization Test)

Additionally, define:

- 10–20% of users held out entirely from training.
- These users appear only in test set.

Purpose:
- Evaluate how model performs on unseen users.
- Measure cold-start quality.

---

# 3. Personalization Evaluation

When using per-user calibration:

## 3.1 Calibration Split

For users with sufficient data:

- Train global model on global train set.
- Freeze global model.
- Use early portion of user’s train set to fit calibrator.
- Evaluate on user test set.

Do NOT use user test data for calibration.

---

# 4. Window Exclusion Rules

Exclude windows from training if:

- They overlap multiple manual label blocks.
- They partially overlap blocks.
- They are rejected windows.
- They have missing critical features.
- They belong to sessions shorter than MIN_BLOCK_DURATION.

---

# 5. Evaluation Metrics

Metrics must be computed:

1. Global aggregate
2. Per-class
3. Per-user
4. Seen users only
5. Unseen users only

Required metrics:

- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Confusion matrix
- Reject rate
- Calibration curve

---

# 6. Temporal Drift Monitoring Split

For drift detection:

Maintain rolling evaluation:

- Train on historical window up to T
- Evaluate on windows in T → T+7 days
- Slide forward

Purpose:
- Detect degradation over time.

---

# 7. No Random K-Fold

K-fold cross-validation across random windows is NOT allowed.

Reason:
- Violates temporal ordering.
- Causes leakage of session patterns.
- Inflates performance artificially.

---

# 8. Class Imbalance Handling

During training:

- Use class weights proportional to inverse frequency
OR
- Use balanced sampling

Must document chosen method in model metadata.

---

# 9. Dataset Snapshotting

Each training run must store:

- Feature schema version
- Label schema version
- Time window used
- User count
- Class distribution
- Random seed (if any)

Training must be reproducible.

---

# 10. Determinism Requirement

Given identical dataset snapshot and config:

- Train split assignment must be identical.
- Evaluation metrics must be identical.

No stochastic split variation allowed.

---

# 11. Versioning

Changing any of the following requires version bump:

- Train/val/test ratios
- Holdout user policy
- Exclusion rules
- Evaluation metrics
