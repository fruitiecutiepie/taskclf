# core.drift

Pure statistical functions for drift and distribution-shift detection.

## Overview

All functions operate on numerical aggregates and prediction outputs only.
No raw content (keystrokes, titles, URLs) is ever accessed or stored.

Thresholds follow [acceptance.md](../../guide/acceptance.md) section 7:

- Feature PSI > 0.2 triggers investigation
- Reject rate increase >= 10% triggers investigation
- Class distribution shift > 15% triggers investigation

## Result Models

| Model | Fields | Description |
|-------|--------|-------------|
| `KsResult` | `statistic`, `p_value` | Two-sample KS test output |
| `FeatureDriftResult` | `feature`, `psi`, `ks_statistic`, `ks_p_value`, `is_drifted` | Per-feature drift metrics |
| `FeatureDriftReport` | `results`, `flagged_features`, `timestamp` | Aggregated multi-feature report |
| `RejectRateDrift` | `ref_rate`, `cur_rate`, `increase`, `is_flagged` | Reject-rate comparison |
| `EntropyDrift` | `ref_mean_entropy`, `cur_mean_entropy`, `ratio`, `is_flagged` | Prediction entropy comparison |
| `ClassShiftResult` | `ref_dist`, `cur_dist`, `max_shift`, `shifted_classes`, `is_flagged` | Class distribution shift |

## Functions

### compute_psi

Population Stability Index between two 1-D distributions.
Uses equal-frequency (quantile) binning derived from the reference set.

```python
from taskclf.core.drift import compute_psi
psi = compute_psi(reference_array, current_array, bins=10)
```

### compute_ks

Two-sample Kolmogorov-Smirnov test via `scipy.stats.ks_2samp`.

```python
from taskclf.core.drift import compute_ks
result = compute_ks(reference_array, current_array)
print(result.statistic, result.p_value, result.is_significant(alpha=0.05))
```

### feature_drift_report

Run PSI + KS across all numerical features and flag drifted ones.

```python
from taskclf.core.drift import feature_drift_report
report = feature_drift_report(ref_df, cur_df, numerical_features)
print(report.flagged_features)
```

### detect_reject_rate_increase

Flag if reject rate increased by >= threshold (absolute).

```python
from taskclf.core.drift import detect_reject_rate_increase
result = detect_reject_rate_increase(ref_labels, cur_labels, threshold=0.10)
```

### detect_entropy_spike

Flag if mean prediction entropy spiked relative to reference.

```python
from taskclf.core.drift import detect_entropy_spike
result = detect_entropy_spike(ref_probs, cur_probs, spike_multiplier=2.0)
```

### detect_class_shift

Flag if any class proportion changed by more than threshold.

```python
from taskclf.core.drift import detect_class_shift
result = detect_class_shift(ref_labels, cur_labels, threshold=0.15)
```

::: taskclf.core.drift
