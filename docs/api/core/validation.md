# core.validation

Data validation for feature DataFrames: range checks, missing rates,
monotonic timestamps, session boundaries, and distribution warnings.

## Usage

```python
from taskclf.core.validation import validate_feature_dataframe

report = validate_feature_dataframe(df, max_missing_rate=0.5)
if not report.ok:
    for finding in report.errors:
        print(f"ERROR: {finding.message}")
for finding in report.warnings:
    print(f"WARN: {finding.message}")
```

## Hard checks (errors)

- Non-nullable columns must not contain nulls.
- Nullable columns must not exceed `max_missing_rate`.
- Numeric values must fall within declared ranges (from `features_v1.json`).
- `bucket_end_ts` must equal `bucket_start_ts + 60s`.
- `bucket_start_ts` must be strictly increasing within each `(user_id, session_id)` group.

## Soft checks (warnings)

- Constant-value columns (std == 0).
- Dominant-value columns (>90% identical).
- Class imbalance (<5% representation) if `label` column exists.
- Session boundary changes with very small gaps.

::: taskclf.core.validation
