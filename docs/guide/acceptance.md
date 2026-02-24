# Acceptance Criteria & Quality Gates v1

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines measurable quality thresholds required for:

- Baseline release
- Model promotion
- Personalization activation
- Retraining approval

Models must meet these criteria before deployment.

---

# 1. Baseline (Heuristic) Acceptance

Before ML deployment, heuristic baseline must satisfy:

- BreakIdle precision ≥ 0.95
- Reject rate ≤ 40%
- No catastrophic misclassification of idle as Build/Write

Baseline establishes minimum acceptable behavior.

---

# 2. Global Model Minimum Performance

Evaluated on test set.

## 2.1 Core Requirements

- Macro F1 ≥ 0.65
- Weighted F1 ≥ 0.70
- BreakIdle precision ≥ 0.95
- BreakIdle recall ≥ 0.90
- No class precision < 0.50 (except Meet in cold-start users)

If any class precision < 0.50:
- Model must not be promoted.

---

## 2.2 Seen vs Unseen Users

Seen users:

- Macro F1 ≥ 0.70

Unseen users:

- Macro F1 ≥ 0.60

If unseen user F1 < 0.55:
- Cold-start UX must default to heuristic assist mode.

---

# 3. Calibration Requirements

After calibration:

- Brier score improvement ≥ 5% over raw model
OR
- Reliability curve visually closer to diagonal

Overconfidence is not allowed.

---

# 4. Reject Rate Bounds

Reject rate must satisfy:

- ≥ 5% (avoid overconfidence)
- ≤ 30% (avoid unusable system)

If reject rate > 35%:
- Model considered underfit or feature insufficient.

---

# 5. Label Stability (Flap Rate)

Define flap rate:

```

Number of label changes / total windows

```

Acceptance:

- Flap rate ≤ 0.25 before smoothing
- ≤ 0.15 after smoothing

High flap rate indicates unstable predictions.

---

# 6. Smoothing Acceptance

After block merging:

- ≥ 80% of blocks ≥ MIN_BLOCK_DURATION
- No more than 10% of blocks shorter than 2 minutes

---

# 7. Drift Monitoring Thresholds

Trigger investigation if:

- Macro F1 drops by ≥ 10% relative to previous model
- Reject rate increases by ≥ 10%
- Feature PSI > 0.2 for any major feature
- Class distribution shift > 15%

---

# 8. Personalization Activation Criteria

Per-user calibration enabled only if:

- ≥ 200 labeled windows
- ≥ 3 separate days of data
- ≥ 3 distinct core labels observed

Otherwise:
- Use global calibration.

---

# 9. Training Reproducibility

Each promoted model must include:

- Model artifact hash
- Dataset snapshot hash
- Feature schema version
- Label schema version
- Config parameters
- Training timestamp

If training cannot be reproduced, model cannot be promoted.

---

# 10. Safety Rules

The following errors are blockers:

- BreakIdle frequently misclassified as Build
- System crashes on missing feature
- Probability vector does not sum to 1.0
- Inconsistent label ordering across inference calls

---

# 11. Production Promotion Checklist

Before release:

- All acceptance criteria met
- Drift test passed on most recent week
- Manual sanity check performed on 3 users
- Documentation updated

---

# 12. Versioning

Any change to:

- Threshold values
- Required metrics
- Reject bounds
- Personalization activation conditions

Requires version bump.
